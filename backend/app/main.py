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
import zipfile
from typing import Dict, List, Tuple, Optional, Iterable, Callable, Awaitable

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

def _provider_from_settings(s: Settings):
    if s.provider == "openai_compat":
        return OpenAICompatProvider(s)
    if s.provider == "ollama":
        return OllamaProvider(s)
    raise HTTPException(status_code=400, detail="Unsupported provider")

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

@app.post("/providers/verify")
async def verify_provider(payload: Settings):
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

            if r.status_code == 400 and "response_format" in r.text:
                body = {"model": model, "temperature": temp, "messages": messages}
                r = await client.post(f"{base}/chat/completions", headers=headers, json=body)

        if r.status_code >= 400:
            raise HTTPException(400, f"HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        from .providers.openai_compat import extract_json_safely as oa_extract
        try:
            parsed = oa_extract(content)
            assert isinstance(parsed.get("items"), list)
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
    catalog: polib.POFile, target_locale: str
) -> Tuple[List[dict], Dict[str, Tuple[int, Optional[int]]]]:
    items: List[dict] = []
    reverse: Dict[str, Tuple[int, Optional[int]]] = {}
    npl = nplurals_for_locale(target_locale)

    for idx, e in enumerate(catalog):
        expected = extract_all_placeholders_from_entry(e)
        if e.msgid_plural:
            for f in range(npl):
                src_text = e.msgid if f == 0 else (e.msgid_plural or e.msgid)
                key = f"{idx}|p{f}"
                items.append({"key": key, "text": src_text, "expected_placeholders": expected})
                reverse[key] = (idx, f)
        else:
            key = f"{idx}"
            items.append({"key": key, "text": e.msgid, "expected_placeholders": expected})
            reverse[key] = (idx, None)

    return items, reverse


async def _translate_items_with_retries(
    provider,
    all_items: List[dict],
    source_lang: str,
    target_locale: str,
    batch_size: int,
    system_prompt: str,
    glossary: str,
    on_batch_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> Dict[str, str]:
    """
    Sends items in batches; retries missing/echoed-only items up to MAX_RETRIES.
    Progress increments only when new keys are actually filled.
    """
    batch_size = max(10, min(100, int(batch_size or 40)))
    remaining: Dict[str, dict] = {it["key"]: it for it in all_items}
    translated: Dict[str, str] = {}

    def _batches(values: Iterable[dict]) -> List[List[dict]]:
        lst = list(values)
        return [lst[i: i + batch_size] for i in range(0, len(lst), batch_size)]

    for attempt in range(1, MAX_RETRIES + 2):
        if not remaining:
            break

        for batch in _batches(list(remaining.values())):
            try:
                result = await provider.translate_batch(
                    batch=batch,
                    source_lang=source_lang,
                    target_locale=target_locale,
                    system_prompt=system_prompt,
                    glossary=glossary,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Provider error: {e}")

            if not isinstance(result, dict):
                raise HTTPException(status_code=400, detail="Provider error: invalid batch result")

            newly, echoed = 0, 0
            for k, v in result.items():
                if k not in remaining:
                    continue
                src_text = remaining[k]["text"]
                if v == src_text and not _allow_echo(src_text, glossary):
                    echoed += 1
                    continue
                translated[k] = v
                del remaining[k]
                newly += 1

            if on_batch_progress and (newly > 0 or echoed > 0):
                await on_batch_progress(newly, echoed)

    if remaining:
        first_missing = next(iter(remaining.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Missing translation for {first_missing} ({target_locale}). The provider did not return all items.",
        )

    return translated


def _fill_catalog_from_map(
    src_catalog: polib.POFile,
    out_catalog: polib.POFile,
    translated_map: Dict[str, str],
    target_locale: str,
):
    npl = nplurals_for_locale(target_locale)
    for idx, e in enumerate(src_catalog):
        expected = extract_all_placeholders_from_entry(e)
        if e.msgid_plural:
            out_catalog[idx].msgstr_plural = {}
            for f in range(npl):
                key = f"{idx}|p{f}"
                t = translated_map.get(key)
                if t is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing translation for {key} ({target_locale}). The provider did not return all plural forms.",
                    )
                try:
                    validate_translation_placeholders(expected, t)
                except ValueError as ve:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Placeholder error in {key} ({target_locale}): {ve}",
                    )
                out_catalog[idx].msgstr_plural[f] = t
        else:
            key = f"{idx}"
            t = translated_map.get(key)
            if t is None:
                raise HTTPException(
                    status_code=400, detail=f"Missing translation for {key} ({target_locale})."
                )
            try:
                validate_translation_placeholders(expected, t)
            except ValueError as ve:
                raise HTTPException(
                    status_code=400,
                    detail=f"Placeholder error in {key} ({target_locale}): {ve}",
                )
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
        translated_map = await _translate_items_with_retries(
            provider=provider,
            all_items=items,
            source_lang=sourceLang,
            target_locale=loc,
            batch_size=s.batch_size,
            system_prompt=s.system_prompt,
            glossary=s.glossary,
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
        content = await po.read()
        src_catalog = polib.pofile(content.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PO file: {e}")

    s = EFFECTIVE_SETTINGS if not settings_json else Settings(**json.loads(settings_json)).normalized()
    provider = _provider_from_settings(s)
    locales_list = [x.strip() for x in (locales or "").split(",") if x.strip()]
    if not locales_list:
        raise HTTPException(status_code=400, detail="No target locales provided")

    async def generator():
        total_strings_per_locale = {loc: len(_build_items_for_locale(src_catalog, loc)[0]) for loc in locales_list}
        grand_total = sum(total_strings_per_locale.values())

        yield json.dumps({"type": "meta", "total": grand_total}) + "\n"

        produced: Dict[str, polib.POFile] = {}
        overall_done = 0
        overall_echoed = 0
        
        progress_buffer = []

        # **FIX**: `on_progress` is now a regular function that appends to a buffer.
        async def on_progress(newly: int, echoed: int):
            nonlocal overall_done, overall_echoed
            overall_done += newly
            overall_echoed += echoed
            progress_buffer.append(json.dumps(
                {"type": "progress", "done": overall_done, "total": grand_total, "echoed": overall_echoed}
            ) + "\n")

        try:
            for loc in locales_list:
                all_items_for_locale, _ = _build_items_for_locale(src_catalog, loc)
                
                translated_map = await _translate_items_with_retries(
                    provider=provider,
                    all_items=all_items_for_locale,
                    source_lang=sourceLang,
                    target_locale=loc,
                    batch_size=s.batch_size,
                    system_prompt=s.system_prompt,
                    glossary=s.glossary,
                    on_batch_progress=on_progress,
                )
                
                # **FIX**: Yield any buffered progress immediately after a batch.
                for progress_item in progress_buffer:
                    yield progress_item
                progress_buffer.clear()

                out = _clone_catalog_structure(src_catalog)
                out.metadata["Language"] = loc
                out.metadata["Plural-Forms"] = get_plural_forms_header(loc)
                _fill_catalog_from_map(src_catalog, out, translated_map, loc)
                produced[loc] = out

            zip_bytes = _build_zip_bytes(produced)
            b64 = base64.b64encode(zip_bytes).decode("ascii")
            yield json.dumps({"type": "done", "zipBase64": b64}) + "\n"

        except HTTPException as he:
            yield json.dumps({"type": "error", "message": he.detail}) + "\n"
        except Exception as e:
            log.exception("Error during translation stream")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")

# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------
@app.get("/debug/provider/recent")
def provider_recent(n: int = 5):
    """Peek at the last few raw model payloads captured by provider adapters."""
    return {"ok": True, "items": debug_recent(n)}