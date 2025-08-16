from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import pathlib
import re
import tempfile
from app.analyzers.po_lint import analyze_po
import zipfile
from typing import Dict, List, Tuple, Optional, Iterable, Callable, Awaitable, Any

import httpx
import polib
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Assuming the following files exist in the same directory structure
# If not, you might need to adjust the import paths
from .providers.openai_compat import OpenAICompatProvider
from .providers.ollama import OllamaProvider
from .schemas import AnalyzeOut, Settings, SettingsOut
from .utils.audit import audit_po
from .utils.logging_config import setup_logging
from .utils.placeholders import (
    extract_all_placeholders,               # NEW: import to get placeholders for singular/plural separately
    extract_all_placeholders_from_entry,
    extract_samples_from_catalog,
    validate_translation_placeholders,
)
from .utils.plurals import get_plural_forms_header, nplurals_for_locale
from .utils.debug_buffer import recent as debug_recent

# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------
load_dotenv()
setup_logging()
log = logging.getLogger("app")

app = FastAPI(title="WP Plugin PO File AI Translator", version="0.6.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SETTINGS_FILE = os.getenv("SETTINGS_FILE", ".settings.json")
SETTINGS_PATH = pathlib.Path(__file__).resolve().parent.parent / SETTINGS_FILE

# retry behavior for provider short-returns
MAX_RETRIES = int(os.getenv("TRANSLATE_MAX_RETRIES", "3"))

# -----------------------------------------------------------------------------
# Settings persistence
# -----------------------------------------------------------------------------
def _settings_to_dict(s: Settings) -> dict:
    return s.model_dump()

def _settings_from_disk() -> Optional[Settings]:
    try:
        if SETTINGS_PATH.exists():
            data = json.load(open(SETTINGS_PATH, "r", encoding="utf-8"))
            return Settings(**data).normalized()
    except Exception as e:
        log.warning("Failed to read settings: %s", e)
    return None

def _settings_save_to_disk(s: Settings) -> None:
    try:
        SETTINGS_PATH.write_text(
            json.dumps(_settings_to_dict(s), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("Failed to persist settings: %s", e)

_disk = _settings_from_disk()
if _disk:
    EFFECTIVE_SETTINGS = _disk
else:
    EFFECTIVE_SETTINGS = Settings(
        provider=os.getenv("PROVIDER", "openai_compat"),
        system_prompt=os.getenv("SYSTEM_PROMPT", "").strip()
        or "You are a professional software translator for WordPress plugins.",
        glossary=os.getenv("GLOSSARY", ""),
        batch_size=int(os.getenv("BATCH_SIZE", "40")),  # lower default
        openai_compat=dict(
            base_url=os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.getenv("OPENAI_COMPAT_API_KEY", ""),
            model=os.getenv("OPENAI_COMPAT_MODEL", "openai/gpt-oss-20b"),
            temperature=float(os.getenv("OPENAI_COMPAT_TEMPERATURE", "0.2")),
        ),
        ollama=dict(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        ),
    )
    _settings_save_to_disk(EFFECTIVE_SETTINGS)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_po_file(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".po"):
        raise HTTPException(status_code=400, detail="Only .po files are accepted")

def _coerce_headers(raw: Any) -> Dict[str, str]:
    """
    Accept dict or list of {key,value} pairs; return flat str->str dict.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for (k, v) in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("key") is not None:
                out[str(item["key"])] = str(item.get("value", ""))
        return out
    return {}

def _provider_from_settings(s: Any):
    """
    Build a provider instance from request settings.

    Accepts either:
      • a Settings object with nested .openai_compat / .ollama blocks
      • a plain dict, either full Settings-shaped or provider-scoped

    Compatible with providers that use either:
      OpenAICompatProvider(model=..., api_key=..., ...)
      OpenAICompatProvider(settings={...})
      OpenAICompatProvider({...})  # positional dict
    And same for OllamaProvider.
    """

    def _instantiate_openai(cfg: Dict[str, Any], provider_label: str):
        # Normalize
        model = cfg.get("model") or cfg.get("model_name")
        if not model:
            raise HTTPException(status_code=400, detail="missing 'model' in provider settings")

        api_key = cfg.get("api_key") or cfg.get("apiKey")
        base_url = cfg.get("base_url") or cfg.get("baseUrl")
        if not base_url:
            base_url = (
                "https://api.openai.com/v1"
                if provider_label in ("openai", "openai_compat", "openai-compat")
                else "https://openrouter.ai/api/v1"
            )
        temperature = float(cfg.get("temperature", 0.2))
        timeout = int(cfg.get("timeout", 60))
        headers = _coerce_headers(cfg.get("headers") or cfg.get("extra_headers"))

        # new flags
        use_response_format = bool(cfg.get("use_response_format", True))
        trace = bool(cfg.get("trace", False))
        trace_truncate = int(cfg.get("trace_truncate", 2000))

        # Try keyword-style constructor
        try:
            return OpenAICompatProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
                extra_headers=headers,
                # not all builds accept these kw args; if they don't, fall through
                use_response_format=use_response_format,
                trace=trace,
                trace_truncate=trace_truncate,
            )
        except TypeError:
            # Try positional dict
            try:
                return OpenAICompatProvider(
                    {
                        "model": model,
                        "api_key": api_key,
                        "base_url": base_url,
                        "temperature": temperature,
                        "timeout": timeout,
                        "headers": headers,
                        "extra_headers": headers,
                        "use_response_format": use_response_format,
                        "trace": trace,
                        "trace_truncate": trace_truncate,
                    }
                )
            except TypeError:
                # Try settings= kwarg
                return OpenAICompatProvider(
                    settings={
                        "model": model,
                        "api_key": api_key,
                        "base_url": base_url,
                        "temperature": temperature,
                        "timeout": timeout,
                        "headers": headers,
                        "extra_headers": headers,
                        "use_response_format": use_response_format,
                        "trace": trace,
                        "trace_truncate": trace_truncate,
                    }
                )

    def _instantiate_ollama(cfg: Dict[str, Any]):
        model = cfg.get("model") or cfg.get("model_name")
        if not model:
            raise HTTPException(status_code=400, detail="missing 'model' in provider settings")

        base_url = cfg.get("base_url") or cfg.get("baseUrl") or cfg.get("host") or "http://localhost:11434"
        temperature = float(cfg.get("temperature", 0.2))
        timeout = int(cfg.get("timeout", 120))
        headers = _coerce_headers(cfg.get("headers") or cfg.get("extra_headers"))

        # new flags
        trace = bool(cfg.get("trace", False))
        trace_truncate = int(cfg.get("trace_truncate", 2000))

        # Try keyword-style constructor
        try:
            return OllamaProvider(
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
                extra_headers=headers,
                trace=trace,
                trace_truncate=trace_truncate,
            )
        except TypeError:
            # Try positional dict
            try:
                return OllamaProvider(
                    {
                        "model": model,
                        "base_url": base_url,
                        "temperature": temperature,
                        "timeout": timeout,
                        "headers": headers,
                        "extra_headers": headers,
                        "trace": trace,
                        "trace_truncate": trace_truncate,
                    }
                )
            except TypeError:
                # Try settings= kwarg
                return OllamaProvider(
                    settings={
                        "model": model,
                        "base_url": base_url,
                        "temperature": temperature,
                        "timeout": timeout,
                        "headers": headers,
                        "extra_headers": headers,
                        "trace": trace,
                        "trace_truncate": trace_truncate,
                    }
                )

    # 1) Settings object path
    if hasattr(s, "provider"):
        provider = (getattr(s, "provider", None) or "openai_compat").lower()

        if provider in ("openai", "openrouter", "openai_compat", "openai-compat"):
            cfg = dict(getattr(s, "openai_compat", {}) or {})
            return _instantiate_openai(cfg, provider)

        if provider in ("ollama", "local"):
            cfg = dict(getattr(s, "ollama", {}) or {})
            return _instantiate_ollama(cfg)

        raise HTTPException(status_code=400, detail=f"Unsupported provider '{provider}'")

    # 2) Dict path
    if not isinstance(s, dict):
        raise HTTPException(status_code=400, detail="provider settings must be an object")

    provider = (s.get("provider") or s.get("type") or "openai_compat").lower()

    if provider in ("openai", "openrouter", "openai_compat", "openai-compat"):
        # Either provider-scoped dict or full settings dict
        cfg = dict(s.get("openai_compat", {}) or s)
        return _instantiate_openai(cfg, provider)

    if provider in ("ollama", "local"):
        cfg = dict(s.get("ollama", {}) or s)
        return _instantiate_ollama(cfg)

    raise HTTPException(status_code=400, detail=f"Unsupported provider '{provider}'")

def _clone_catalog_structure(src: polib.POFile) -> polib.POFile:
    dst = polib.POFile()
    for k, v in (src.metadata or {}).items():
        dst.metadata[k] = v
    for e in src:
        new_e = polib.POEntry(
            msgid=e.msgid,
            msgctxt=e.msgctxt,
            msgid_plural=e.msgid_plural,
            occurrences=e.occurrences,
            comment=e.comment,
            tcomment=e.tcomment,
            flags=list(e.flags) if e.flags else [],
        )
        dst.append(new_e)
    return dst

def _catalog_to_po_text(cat: polib.POFile) -> str:
    return str(cat)

def _build_zip_bytes(per_locale_po: Dict[str, polib.POFile]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for locale, catalog in per_locale_po.items():
            zf.writestr(f"languages/{locale}.po", _catalog_to_po_text(catalog))
            with tempfile.TemporaryDirectory() as td:
                po_tmp = os.path.join(td, f"{locale}.po")
                mo_tmp = os.path.join(td, f"{locale}.mo")
                catalog.save(po_tmp)
                catalog.save_as_mofile(mo_tmp)
                with open(mo_tmp, "rb") as f:
                    zf.writestr(f"languages/{locale}.mo", f.read())
    mem.seek(0)
    return mem.read()

# --- language name helpers (explicit instruction for model) ---
_LANG_MAP = {
    "en": "English", "en-us": "English", "en-gb": "English",
    "es": "Spanish", "es-es": "Spanish", "es-mx": "Spanish",
    "fr": "French", "fr-fr": "French", "fr-ca": "French",
    "de": "German", "de-de": "German",
    "it": "Italian", "it-it": "Italian",
    "pt": "Portuguese", "pt-pt": "Portuguese", "pt-br": "Portuguese (Brazilian)",
    "nl": "Dutch", "nl-nl": "Dutch",
    "sv": "Swedish", "sv-se": "Swedish",
    "no": "Norwegian", "nb": "Norwegian", "nn": "Norwegian",
    "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "sk": "Slovak",
    "ru": "Russian", "uk": "Ukrainian",
    "tr": "Turkish", "el": "Greek",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
}

def language_name_for_locale(locale: str) -> str:
    if not locale:
        return ""
    key = locale.lower()
    return _LANG_MAP.get(key) or _LANG_MAP.get(key.split("-")[0], key)

# --- echo allowlist heuristics ---
_URL_RE = re.compile(r"^https?://|^www\.", re.I)

def _allow_echo(src: str, glossary: str) -> bool:
    s = (src or "").strip()
    if not s:
        return True
    if len(s) < 3:
        return True
    if _URL_RE.search(s):
        return True
    # no alphabetic letters = essentially placeholders/symbols/numbers
    if not re.search(r"[A-Za-zΑ-Ωα-ωÀ-ÿ]", s):
        return True
    # exact glossary term match (case-insensitive, trimmed)
    gl_terms = [t.strip().lower() for t in (glossary or "").replace("\n", ",").split(",") if t.strip()]
    return s.lower() in gl_terms

# -----------------------------------------------------------------------------
# Settings endpoints
# -----------------------------------------------------------------------------
@app.get("/settings", response_model=SettingsOut)
def get_settings():
    public = EFFECTIVE_SETTINGS.public_copy()
    return SettingsOut(ok=True, defaults=public)

@app.post("/settings", response_model=SettingsOut)
def set_settings(payload: Settings):
    global EFFECTIVE_SETTINGS
    EFFECTIVE_SETTINGS = payload.normalized()
    _settings_save_to_disk(EFFECTIVE_SETTINGS)
    public = EFFECTIVE_SETTINGS.public_copy()
    return SettingsOut(ok=True, defaults=public)

# -----------------------------------------------------------------------------
# Provider helpers
# -----------------------------------------------------------------------------
@app.get("/providers/ollama/models")
async def ollama_models():
    s = EFFECTIVE_SETTINGS
    host = s.ollama["host"].rstrip("/")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{host}/api/tags")
    if r.status_code >= 400:
        raise HTTPException(400, f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    try:
        tags = r.json().get("models", [])
        models = [m.get("model") or m.get("name") for m in tags if isinstance(m, dict)]
    except Exception:
        models = []
    return {"ok": True, "models": models}

@app.post("/analyze")
async def analyze_endpoint(
    po: UploadFile = File(...),
):
    try:
        content = await po.read()
        src = polib.pofile(content.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PO file: {e}")

    report = analyze_po(src, locale=None, filename=po.filename)
    return JSONResponse(report)


@app.post("/analyze/apply")
async def analyze_apply_endpoint(
    po: UploadFile = File(...),
    fixes_json: str = Form("[]"),  # array of {index, key, kind, text}
):
    """
    Apply safe auto-fixes to a PO and return the updated PO as base64.
    'kind' must be 'replace_msgstr' or 'replace_plural_i'.
    """
    try:
        content = await po.read()
        src = polib.pofile(content.decode("utf-8", errors="replace"))
        fixes = json.loads(fixes_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

    for f in fixes:
        idx = int(f["index"])
        kind = f["kind"]
        text = f["text"]
        if kind == "replace_msgstr":
            src[idx].msgstr = text
        elif kind == "replace_plural_i":
            key = str(f.get("key", ""))
            if "|p" in key:
                try:
                    p = int(key.split("|p", 1)[1])
                    src[idx].msgstr_plural[p] = text
                except Exception:
                    continue  # ignore malformed plural index

    out_bytes = src.__unicode__().encode("utf-8")
    b64 = base64.b64encode(out_bytes).decode("ascii")
    return JSONResponse({"poBase64": b64})


@app.post("/providers/verify")
async def verify_provider(payload: Settings):
    """
    Verify provider connectivity and JSON contract without importing helpers from provider modules.
    This avoids tight coupling to provider-internal utilities.
    """
    # Local, robust JSON extractor: strict -> fenced block -> largest {...} span
    def _extract_json_safely_local(text: str) -> dict:
        if isinstance(text, dict):
            return text
        if not isinstance(text, str):
            raise ValueError("non-string content")
        s = text.strip()

        # fenced ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception as e:
                raise ValueError(f"fenced non-JSON: {e}")

        # direct JSON
        try:
            return json.loads(s)
        except Exception:
            pass

        # largest {...} span
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = s[first : last + 1]
            try:
                return json.loads(candidate)
            except Exception as e:
                raise ValueError(f"non-JSON or truncated JSON: {e}")

        raise ValueError("no JSON object found")

    s = payload.normalized()

    if s.provider == "openai_compat":
        base = s.openai_compat["base_url"].rstrip("/")
        key = s.openai_compat["api_key"]
        model = s.openai_compat["model"]
        temp = float(s.openai_compat["temperature"])

        if not base:
            raise HTTPException(400, "OpenAI base_url missing")
        if "openai.com" in base and not key:
            raise HTTPException(400, "OpenAI API key missing")

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        messages = [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": 'Respond with {"items":[{"key":"k","text":"ok"}]}'},
        ]

        async with httpx.AsyncClient(timeout=30) as client:
            body = {
                "model": model,
                "temperature": temp,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            r = await client.post(f"{base}/chat/completions", headers=headers, json=body)

            # Some gateways require json_schema; try that if json_object is rejected
            if r.status_code == 400 and "response_format.type" in r.text and "json_schema" in r.text:
                json_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verify_items",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["items"],
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["key", "text"],
                                        "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
                                    },
                                }
                            },
                        },
                    },
                }
                body = {"model": model, "temperature": temp, "response_format": json_schema, "messages": messages}
                r = await client.post(f"{base}/chat/completions", headers=headers, json=body)

            # Final fallback: no response_format at all
            if r.status_code == 400 and "response_format" in r.text:
                body = {"model": model, "temperature": temp, "messages": messages}
                r = await client.post(f"{base}/chat/completions", headers=headers, json=body)

        if r.status_code >= 400:
            raise HTTPException(400, f"HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = _extract_json_safely_local(content)
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("top-level 'items' missing or not an array")
            # sanity: ensure each item is an object with key and text
            if items and not (isinstance(items[0], dict) and "key" in items[0] and "text" in items[0]):
                raise ValueError("each items[i] must be an object with 'key' and 'text'")
        except Exception as e:
            raise HTTPException(400, f"Model did not return strict JSON: {e}")

        return {"ok": True, "provider": "openai_compat", "details": {"model": model}}

    if s.provider == "ollama":
        host = s.ollama["host"].rstrip("/")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{host}/api/tags")
        if r.status_code >= 400:
            raise HTTPException(400, f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        try:
            tags = r.json().get("models", [])
            models = [m.get("model") or m.get("name") for m in tags if isinstance(m, dict)]
        except Exception:
            models = []
        return {"ok": True, "provider": "ollama", "details": {"models": models}}

    raise HTTPException(400, "Unsupported provider")


# -----------------------------------------------------------------------------
# Analyze + Audit (summary for UI)
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(po: UploadFile = File(...)):
    _ensure_po_file(po)
    try:
        content = await po.read()
        catalog = polib.pofile(content.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PO file: {e}")

    headers = dict(catalog.metadata or {})
    total = len(catalog)
    plurals = sum(1 for e in catalog if e.msgid_plural)
    untranslated = sum(1 for e in catalog if not e.translated())
    samples = extract_samples_from_catalog(catalog)
    audit = audit_po(catalog)

    summary_text = (
        f"{total} entries. {plurals} plural blocks. "
        f"{untranslated} untranslated. "
        f"Issues found: {len(audit.get('issues', []))}."
    )

    info = dict(
        headers=headers,
        counts=dict(entries=total, plurals=plurals, untranslated=untranslated),
        placeholderSamples=samples[:20],
        summaryText=summary_text,
    )
    return AnalyzeOut(ok=True, info=info, audit=audit)

# -----------------------------------------------------------------------------
# Translation core + batching with automatic retries and anti-echo
# -----------------------------------------------------------------------------
def _build_items_for_locale(
    catalog: polib.POFile,
    target_locale: str,
) -> Tuple[List[dict], Dict[str, Tuple[int, Optional[int]]]]:
    """
    Build model input items for a given locale.

    Returns:
      items: [{"key": "idx|p0", "text": "...", "expected_placeholders": {...}, "context": "..."}, ...]
      reverse: {"idx|p0": (entry_index, plural_form_index_or_None), ...}
    """
    items: List[dict] = []
    reverse: Dict[str, Tuple[int, Optional[int]]] = {}
    npl = nplurals_for_locale(target_locale)

    for idx, e in enumerate(catalog):
        # Skip obsolete entries
        if e.obsolete:
            continue

        # Plural entry
        if e.msgid_plural:
            expected_sing = extract_all_placeholders(e.msgid or "")
            expected_plur = extract_all_placeholders(e.msgid_plural or "")

            for f in range(npl):
                src_text = e.msgid if f == 0 else (e.msgid_plural or e.msgid)
                key = f"{idx}|p{f}"
                exp = expected_sing if f == 0 else expected_plur
                items.append(
                    {
                        "key": key,
                        "text": src_text,
                        "expected_placeholders": exp,
                        "context": e.msgctxt or "",
                    }
                )
                reverse[key] = (idx, f)

        # Singular entry
        else:
            key = f"{idx}"
            exp = extract_all_placeholders(e.msgid or "")
            items.append(
                {
                    "key": key,
                    "text": e.msgid or "",
                    "expected_placeholders": exp,
                    "context": e.msgctxt or "",
                }
            )
            reverse[key] = (idx, None)

    return items, reverse


# Hardened retry logic with micro-batch last pass and review list (no blind echoes).
async def _translate_items_with_retries(
    *,
    all_items: List[Dict[str, Any]],
    provider,  # must expose async translate_batch(...)
    source_lang: str,
    target_locale: str,
    batch_size: int,
    system_prompt: Optional[str] = None,
    glossary: Optional[str] = None,
    http_session=None,
    max_retries: int = 3,
    on_batch_progress=None,   # async (newly:int, echoed:int) -> None
    on_batch_detail=None,     # async (payload:dict) -> None
) -> Tuple[Dict[str, str], List[dict]]:
    """
    Robust translate with batching, retries, placeholder validation, and review output.

    Returns:
      (translated_map, review_items)
      translated_map: { key: translated_text }
      review_items: [ {key, source, candidate?, reason: "echo"|"missing"|"placeholder_error"} ]
    """

    def _chunks(seq: List[Any], n: int) -> List[List[Any]]:
        n = max(1, int(n))
        return [seq[i:i+n] for i in range(0, len(seq), n)]

    def _review_item(reason: str, key: str, source: str, candidate: Optional[str] = None) -> dict:
        item = {"key": str(key), "source": source, "reason": reason}
        if candidate is not None:
            item["candidate"] = candidate
        return item

    # Build remaining work map; keep per-item expected placeholders and context if present
    remaining: Dict[str, Dict[str, Any]] = {
        str(it["key"]): {
            "key": str(it["key"]),
            "text": str(it["text"]),
            "expected": it.get("expected_placeholders"),
            "context": it.get("context", ""),
        }
        for it in all_items
    }

    translated: Dict[str, str] = {}
    review_items: List[dict] = []

    if not remaining:
        return {}, []

    # We track echo once and do not retry echoes to avoid loops
    pending_echo: Dict[str, Dict[str, Any]] = {}

    for attempt in range(1, int(max_retries) + 1):
        if not remaining:
            break

        before_count = len(remaining)
        attempt_progress = 0  # accepted in this attempt

        for b_index, batch in enumerate(_chunks(list(remaining.values()), batch_size), start=1):
            # Construct provider batch; include context for disambiguation
            provider_batch = []
            for it in batch:
                pb = {"key": it["key"], "text": it["text"]}
                if it.get("context"):
                    pb["context"] = it["context"]
                provider_batch.append(pb)

            try:
                result_map = await provider.translate_batch(
                    batch=provider_batch,
                    source_lang=source_lang,
                    target_locale=target_locale,
                    glossary=glossary,
                    system_prompt=system_prompt,
                    http_session=http_session,
                )
            except Exception as e:
                if on_batch_detail:
                    await on_batch_detail(
                        {
                            "type": "batch_error",
                            "attempt": attempt,
                            "batch_index": b_index,
                            "batch_size": len(batch),
                            "error": str(e)[:200],
                            "locale": target_locale,
                        }
                    )
                continue

            if not isinstance(result_map, dict):
                if on_batch_detail:
                    await on_batch_detail(
                        {
                            "type": "batch_error",
                            "attempt": attempt,
                            "batch_index": b_index,
                            "batch_size": len(batch),
                            "error": "invalid batch result shape",
                            "locale": target_locale,
                        }
                    )
                continue

            newly, echoed = 0, 0

            for item in batch:
                k = str(item["key"])
                v = result_map.get(k)
                src_text = item["text"]

                if v is None:
                    # model omitted this key; keep it in remaining for possible retry
                    continue

                v = "" if v is None else str(v)

                # Ignore pure echoes unless acceptable as-is
                if v == src_text and not _allow_echo(src_text, glossary):
                    echoed += 1
                    pending_echo[k] = {"key": k, "text": src_text}
                    # Remove from remaining so we don't loop on it
                    if k in remaining:
                        del remaining[k]
                    continue

                # Validate placeholders for this item
                try:
                    validate_translation_placeholders(item.get("expected") or {}, v)
                except ValueError as ve:
                    # Record placeholder error for review; keep for later retry this attempt or micro-pass
                    review_items.append(_review_item("placeholder_error", k, src_text, candidate=v))
                    continue

                # Accept translation
                translated[k] = v
                if k in remaining:
                    del remaining[k]
                pending_echo.pop(k, None)
                newly += 1

            attempt_progress += newly

            if on_batch_detail:
                await on_batch_detail(
                    {
                        "type": "batch_result",
                        "attempt": attempt,
                        "batch_index": b_index,
                        "batch_size": len(batch),
                        "returned": len(result_map),
                        "accepted": newly,
                        "echoed": echoed,
                        "remaining": len(remaining),
                        "locale": target_locale,
                    }
                )
            if on_batch_progress:
                await on_batch_progress(newly, echoed)

        # Anti-spin: if we made zero progress and nothing shrank, break early to micro-batches
        if attempt_progress == 0 and len(remaining) == before_count:
            if on_batch_detail:
                await on_batch_detail(
                    {"type": "no_progress_break", "attempt": attempt, "remaining": len(remaining), "locale": target_locale}
                )
            break

    # Final micro-batch pass for anything still remaining
    if remaining:
        for k, item in list(remaining.items()):
            try:
                result_map = await provider.translate_batch(
                    batch=[{"key": item["key"], "text": item["text"], **({"context": item["context"]} if item.get("context") else {})}],
                    source_lang=source_lang,
                    target_locale=target_locale,
                    glossary=glossary,
                    system_prompt=system_prompt,
                    http_session=http_session,
                )
                v = result_map.get(str(item["key"]))
                if v is None:
                    review_items.append(_review_item("missing", k, item["text"]))
                    continue
                v = str(v)
                if v == item["text"] and not _allow_echo(item["text"], glossary):
                    review_items.append(_review_item("echo", k, item["text"], candidate=item["text"]))
                    continue
                validate_translation_placeholders(item.get("expected") or {}, v)
                translated[str(item["key"])] = v
                del remaining[k]
                if on_batch_progress:
                    await on_batch_progress(1, 0)
            except Exception as e:
                review_items.append(_review_item("missing", k, item["text"]))
                if on_batch_detail:
                    await on_batch_detail(
                        {"type": "micro_error", "key": k, "error": str(e)[:200], "sample_src": item["text"][:120], "locale": target_locale}
                    )

    # Any unresolved echoes were already removed from remaining; ensure they are reported
    for k, item in pending_echo.items():
        review_items.append(_review_item("echo", k, item["text"], candidate=item["text"]))

    return translated, review_items

def _fill_catalog_from_map(
    src_catalog: polib.POFile,
    out_catalog: polib.POFile,
    translated_map: Dict[str, str],
    target_locale: str,
    *,
    strict: bool = True,
) -> None:
    """
    Copy translations from translated_map into out_catalog.

    strict=True  -> raise on missing keys or placeholder errors (used for fully-automatic runs)
    strict=False -> skip missing/invalid entries and leave them untranslated (used for finalize after review)
    """
    npl = nplurals_for_locale(target_locale)

    for idx, e in enumerate(src_catalog):
        expected = extract_all_placeholders_from_entry(e)

        if e.msgid_plural:
            out_catalog[idx].msgstr_plural = {}
            for f in range(npl):
                key = f"{idx}|p{f}"
                t = translated_map.get(key)

                if t is None:
                    if strict:
                        raise HTTPException(status_code=400, detail=f"Missing translation for {key} ({target_locale}).")
                    # lenient: leave this plural form empty/untranslated
                    continue

                try:
                    validate_translation_placeholders(expected, t)
                except ValueError as ve:
                    if strict:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Placeholder error in {key} ({target_locale}): {ve}",
                        )
                    # lenient: skip invalid text, keep untranslated
                    continue

                out_catalog[idx].msgstr_plural[f] = t

        else:
            key = f"{idx}"
            t = translated_map.get(key)

            if t is None:
                if strict:
                    raise HTTPException(status_code=400, detail=f"Missing translation for {key} ({target_locale}).")
                # lenient: leave msgstr empty
                continue

            try:
                validate_translation_placeholders(expected, t)
            except ValueError as ve:
                if strict:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Placeholder error in {key} ({target_locale}): {ve}",
                    )
                # lenient: skip invalid text
                continue

            out_catalog[idx].msgstr = t


# -----------------------------------------------------------------------------
# Non-streaming translate -> returns a ZIP
# -----------------------------------------------------------------------------
@app.post("/translate")
async def translate(
    po: UploadFile = File(...),
    locales: str = Form(...),
    sourceLang: str = Form("en"),
    settings_json: Optional[str] = Form(None),
):
    _ensure_po_file(po)
    try:
        content = await po.read()
        src_catalog = polib.pofile(content.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PO file: {e}")

    s = EFFECTIVE_SETTINGS if not settings_json else Settings(**json.loads(settings_json)).normalized()
    provider = _provider_from_settings(s)
    locales_list = [x.strip() for x in (locales or "").split(",") if x.strip()]
    if not locales_list:
        raise HTTPException(status_code=400, detail="No target locales provided")

    out_per_locale: Dict[str, polib.POFile] = {}
    for loc in locales_list:
        items, _ = _build_items_for_locale(src_catalog, loc)

        # batch logging hooks (info-level)
        async def _on_progress(newly: int, echoed: int):
            log.info("translate /locale=%s accepted=%d echoed=%d", loc, newly, echoed)

        async def _on_detail(payload: dict):
            try:
                log.info("translate detail /locale=%s %s", loc, json.dumps(payload, ensure_ascii=False)[:2000])
            except Exception:
                log.info("translate detail /locale=%s %r", loc, payload)

        translated_map, _review_items = await _translate_items_with_retries(
            provider=provider,
            all_items=items,
            source_lang=sourceLang,
            target_locale=loc,
            batch_size=s.batch_size,
            system_prompt=s.system_prompt,
            glossary=s.glossary,
            on_batch_progress=_on_progress,
            on_batch_detail=_on_detail,
        )

        out = _clone_catalog_structure(src_catalog)
        out.metadata["Language"] = loc
        out.metadata["Plural-Forms"] = get_plural_forms_header(loc)
        _fill_catalog_from_map(src_catalog, out, translated_map, loc)
        out_per_locale[loc] = out

    zip_bytes = _build_zip_bytes(out_per_locale)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="translations.zip"'},
    )


# -----------------------------------------------------------------------------
# Streaming translate with NDJSON progress + base64 ZIP
# -----------------------------------------------------------------------------
@app.post("/translate/ndjson")
async def translate_ndjson(
    po: UploadFile = File(...),
    locales: str = Form(...),
    sourceLang: str = Form("en"),
    settings_json: Optional[str] = Form(None),
):
    _ensure_po_file(po)
    try:
        raw_bytes = await po.read()
        src_catalog = polib.pofile(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PO file: {e}")

    s = EFFECTIVE_SETTINGS if not settings_json else Settings(**json.loads(settings_json)).normalized()
    provider = _provider_from_settings(s)
    locales_list = [x.strip() for x in (locales or "").split(",") if x.strip()]
    if not locales_list:
        raise HTTPException(status_code=400, detail="No target locales provided")

    async def generator():
        # Total across all locales so the bar can reflect the full job
        total_strings_per_locale = {loc: len(_build_items_for_locale(src_catalog, loc)[0]) for loc in locales_list}
        grand_total = sum(total_strings_per_locale.values())

        # Tell the UI how many strings in total we expect
        yield json.dumps({"type": "meta", "total": grand_total}) + "\n"

        # Collect results for possible finalize step
        translated_by_locale: Dict[str, Dict[str, str]] = {}
        review_by_locale: Dict[str, List[dict]] = {}

        overall_done = 0
        overall_echoed = 0

        for loc in locales_list:
            progress_q: asyncio.Queue[str] = asyncio.Queue()

            async def on_progress(newly: int, echoed: int):
                nonlocal overall_done, overall_echoed
                overall_done += newly
                overall_echoed += echoed
                await progress_q.put(
                    json.dumps(
                        {"type": "progress", "locale": loc, "done": overall_done, "total": grand_total, "echoed": overall_echoed}
                    )
                    + "\n"
                )

            async def on_detail(payload: dict):
                payload = dict(payload)
                payload["locale"] = loc
                await progress_q.put(json.dumps(payload) + "\n")

            async def run_locale() -> Tuple[Dict[str, str], List[dict]]:
                all_items_for_locale, _ = _build_items_for_locale(src_catalog, loc)
                return await _translate_items_with_retries(
                    provider=provider,
                    all_items=all_items_for_locale,
                    source_lang=sourceLang,
                    target_locale=loc,
                    batch_size=s.batch_size,
                    system_prompt=s.system_prompt,
                    glossary=s.glossary,
                    on_batch_progress=on_progress,
                    on_batch_detail=on_detail,
                )

            task = asyncio.create_task(run_locale())

            # While the locale is translating, stream queued progress or heartbeat
            while True:
                if task.done() and progress_q.empty():
                    break
                try:
                    line = await asyncio.wait_for(progress_q.get(), timeout=10.0)
                    yield line
                except asyncio.TimeoutError:
                    yield json.dumps({"type": "heartbeat", "locale": loc}) + "\n"

            # End-of-locale result
            try:
                translated_map, review_items = task.result()
            except HTTPException as he:
                yield json.dumps({"type": "error", "message": he.detail, "locale": loc}) + "\n"
                return
            except Exception as e:
                yield json.dumps({"type": "error", "message": f"Provider error: {e}", "locale": loc}) + "\n"
                return

            translated_by_locale[loc] = translated_map
            if review_items:
                review_by_locale[loc] = review_items

        # If anything needs review, emit a single review packet and finish
        total_review = sum(len(v) for v in review_by_locale.values())
        if total_review > 0:
            job = {
                "srcPoB64": base64.b64encode(raw_bytes).decode("ascii"),
                "locales": locales_list,
                "translated": translated_by_locale,
                "review": review_by_locale,
            }
            yield json.dumps({"type": "review", "total": total_review, "job": job}) + "\n"
            return

        # Otherwise build the final ZIP immediately and send as done
        produced: Dict[str, polib.POFile] = {}
        for loc in locales_list:
            out = _clone_catalog_structure(src_catalog)
            out.metadata["Language"] = loc
            out.metadata["Plural-Forms"] = get_plural_forms_header(loc)
            _fill_catalog_from_map(src_catalog, out, translated_by_locale.get(loc, {}), loc)
            produced[loc] = out

        zip_bytes = _build_zip_bytes(produced)
        b64 = base64.b64encode(zip_bytes).decode("ascii")
        yield json.dumps({"type": "done", "zipBase64": b64}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")

# -----------------------------------------------------------------------------
# Diagnostics: quick test-run (language-agnostic)
# -----------------------------------------------------------------------------
@app.post("/diagnostics/test-run")
async def diagnostics_test_run(
    po: UploadFile = File(...),
    locale: str = Form(...),
    sourceLang: str = Form("en"),
    sample_size: int = Form(25),
    settings_json: Optional[str] = Form(None),
):
    """
    Run a quick sample translation and return a compact quality report.
    It does not write files; it uses the current provider settings as-is.
    """
    # parse PO
    _ensure_po_file(po)
    try:
        content = await po.read()
        src_catalog = polib.pofile(content.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PO file: {e}")

    s = EFFECTIVE_SETTINGS if not settings_json else Settings(**json.loads(settings_json)).normalized()
    provider = _provider_from_settings(s)

    all_items, _ = _build_items_for_locale(src_catalog, locale)
    if not all_items:
        return JSONResponse({"ok": False, "error": "No translatable items found"})

    # choose a simple, deterministic sample from head/middle/tail when possible
    n = max(1, min(int(sample_size), len(all_items)))
    if n <= 5:
        sample = all_items[:n]
    else:
        head = all_items[: n // 3]
        mid_start = max(0, (len(all_items) // 2) - (n // 6))
        mid = all_items[mid_start : mid_start + (n // 3)]
        tail = all_items[-(n - len(head) - len(mid)) :]
        sample = head + mid + tail

    # Build provider batch with context (if any)
    provider_batch = []
    for it in sample:
        row = {"key": it["key"], "text": it["text"]}
        if it.get("context"):
            row["context"] = it["context"]
        provider_batch.append(row)

    # call provider once
    try:
        mapping = await provider.translate_batch(
            batch=provider_batch,
            source_lang=sourceLang,
            target_locale=locale,
            glossary=s.glossary,
            system_prompt=s.system_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Provider error: {e}")

    # evaluate
    results: List[Dict[str, Any]] = []
    ok_count = 0
    echo_count = 0
    ph_errors = 0
    missing_count = 0

    for it in sample:
        k = str(it["key"])
        src = it["text"]
        out = mapping.get(k)
        rec: Dict[str, Any] = {"key": k, "source": src, "translation": out or ""}

        if out is None:
            rec["status"] = "missing"
            missing_count += 1
        else:
            out = str(out)
            if out == src and not _allow_echo(src, s.glossary):
                rec["status"] = "echo"
                echo_count += 1
            else:
                try:
                    validate_translation_placeholders(it.get("expected_placeholders") or {}, out)
                    rec["status"] = "ok"
                    ok_count += 1
                except ValueError as ve:
                    rec["status"] = "placeholder_error"
                    rec["placeholder_error"] = str(ve)
                    ph_errors += 1

        results.append(rec)

    summary = {
        "total": len(sample),
        "ok": ok_count,
        "echo": echo_count,
        "placeholder_errors": ph_errors,
        "missing": missing_count,
        "provider": s.provider,
        "model": s.openai_compat.get("model") if s.provider == "openai_compat" else s.ollama.get("model"),
        "base_url": s.openai_compat.get("base_url") if s.provider == "openai_compat" else s.ollama.get("host"),
    }

    # Note: for full wire-level visibility of what was sent/returned,
    # enable provider tracing in settings (openai_compat.trace / ollama.trace).
    return JSONResponse({"ok": True, "summary": summary, "results": results})


from fastapi import Body
from fastapi.responses import JSONResponse

def _key_to_source_map(po: polib.POFile) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for e in po:
        if e.obsolete:
            continue
        key = (e.msgctxt + "|" if e.msgctxt else "") + e.msgid
        m[key] = e.msgid
        if e.msgid_plural:
            # plural forms are handled in _build_items_for_locale, so key will include |p{index}
            pass
    return m

@app.post("/translate/finalize")
async def translate_finalize(payload: dict = Body(...)):
    """
    Finalize a translation job after user review.

    Expected payload:
    {
      "job": {
        "srcPoB64": str,
        "locales": [str],
        "translated": { "<loc>": { "<key>": "<text>", ... }, ... },
        "review": { "<loc>": [ { "key": "...", "source": "...", "candidate": "...", "reason": "echo"|"missing" }, ... ] }
      },
      "decisions": { "<loc>": [ { "key": "...", "action": "accept"|"reject"|"edit", "text": "<edited text optional>" }, ... ] }
    }
    """
    try:
        job = payload.get("job") or {}
        decisions = payload.get("decisions") or {}
        src_b64 = job["srcPoB64"]
        locales_list = job["locales"]
        translated_by_locale: Dict[str, Dict[str, str]] = job.get("translated", {})
        review_by_locale: Dict[str, List[dict]] = job.get("review", {})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid finalize payload")

    # Load source PO
    try:
        src_bytes = base64.b64decode(src_b64)
        src_catalog = polib.pofile(src_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid srcPoB64: {e}")

    produced: Dict[str, polib.POFile] = {}

    for loc in locales_list:
        # start with whatever we already translated automatically
        base_map: Dict[str, str] = dict(translated_by_locale.get(loc, {}))

        # fast lookup of review items by key for this locale
        review_lookup: Dict[str, dict] = {ri["key"]: ri for ri in review_by_locale.get(loc, []) if "key" in ri}

        # apply user decisions
        for d in decisions.get(loc, []):
            key = str(d.get("key") or "")
            action = d.get("action")
            if not key or not action:
                continue

            if action == "accept":
                # Accept echo -> prefer candidate if present else the source shown in the review
                ri = review_lookup.get(key)
                if ri:
                    text = ri.get("candidate") or ri.get("source")
                    if isinstance(text, str):
                        base_map[key] = text

            elif action == "edit":
                text = d.get("text")
                if isinstance(text, str) and text.strip() != "":
                    base_map[key] = text

            elif action == "reject":
                # leave untranslated
                pass

        # build output catalog, but leniently (skip any missing/invalid without raising)
        out = _clone_catalog_structure(src_catalog)
        out.metadata["Language"] = loc
        out.metadata["Plural-Forms"] = get_plural_forms_header(loc)
        _fill_catalog_from_map(src_catalog, out, base_map, loc, strict=False)
        produced[loc] = out

    zip_bytes = _build_zip_bytes(produced)
    b64 = base64.b64encode(zip_bytes).decode("ascii")
    return JSONResponse({"zipBase64": b64})

# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------
@app.get("/debug/provider/recent")
def provider_recent(n: int = 5):
    """Peek at the last few raw model payloads captured by provider adapters."""
    return {"ok": True, "items": debug_recent(n)}