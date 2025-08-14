from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import Settings
from ..utils.debug_buffer import push as debug_push

# ----------------------------- JSON extraction ----------------------------- #

BEGIN = "BEGIN_JSON"
END = "END_JSON"

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = "\n".join(s.splitlines()[1:])
    if s.endswith("```"):
        s = "\n".join(s.splitlines()[:-1])
    return s.strip()

def _between_markers(text: str) -> Optional[str]:
    a = text.find(BEGIN)
    b = text.rfind(END)
    if a != -1 and b != -1 and b > a:
        return text[a + len(BEGIN):b].strip()
    return None

def _first_json_object(text: str) -> str:
    """
    Find and return the first well-formed JSON object substring in `text`.
    Brace counter that handles quotes and escapes; no regex.
    """
    s = text
    start = None
    depth = 0
    in_str = False
    esc = False
    quote = ""

    for i, ch in enumerate(s):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                esc = False
                quote = ""
            continue

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    raise ValueError("no JSON object found")

def extract_json_safely(text: str) -> Dict[str, Any]:
    """
    Extract {"items":[{"key":string,"text":string},...]} from a chat model response.

    Strategy:
      1) Prefer content between BEGIN_JSON ... END_JSON markers if present.
      2) Else strip code fences and try parsing the whole thing as JSON.
      3) Else locate the first {...} object and parse that.
      4) Validate structure strictly, but coerce numeric keys to strings.
    """
    raw = text or ""
    debug_push({"where": "openai_compat.raw", "text": raw[:4000]})

    # 1) fenced block first
    fenced = _between_markers(raw)
    if fenced:
        try:
            obj = json.loads(fenced)
        except Exception as e:
            raise ValueError(f"Model returned non-JSON inside markers: {e}")
    else:
        # 2) whole body
        t = _strip_code_fences(raw)
        try:
            obj = json.loads(t)
        except Exception:
            # 3) first object
            try:
                frag = _first_json_object(t)
                obj = json.loads(frag)
            except Exception as e:
                raise ValueError(f"Model returned non-JSON: {e}")

    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object")

    items = obj.get("items")
    if not isinstance(items, list):
        raise ValueError("Top-level object must contain an 'items' array")

    normalized: List[Dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("Invalid item in model response (not an object)")
        k = it.get("key")
        v = it.get("text", it.get("translation"))
        # be liberal: coerce to strings when feasible
        if isinstance(k, (int, float)):
            k = str(k)
        if isinstance(v, (int, float)):
            v = str(v)
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("Invalid item in model response")
        normalized.append({"key": k, "text": v})

    return {"items": normalized}

# ------------------------------- Provider ---------------------------------- #

_FEWSHOT = (
    'Example input:\n'
    '{"items":[{"key":"A|p0","text":"%d result"},{"key":"A|p1","text":"%d results"},{"key":"B","text":"Add to cart"}]}\n'
    'Example output (Spanish):\n'
    f'{BEGIN}{{"items":[{{"key":"A|p0","text":"%d resultado"}},{{"key":"A|p1","text":"%d resultados"}},{{"key":"B","text":"Añadir al carrito"}}]}}{END}'
)

class OpenAICompatProvider:
    """
    Provider for OpenAI-compatible /chat/completions APIs.
    Implements translate_batch(batch, source_lang, target_locale, system_prompt, glossary) -> {key: text}
    """

    def __init__(self, settings: Settings):
        s = settings.normalized()
        oa = s.openai_compat
        self.base_url: str = (oa.get("base_url") or "http://127.0.0.1:1234/v1").rstrip("/")
        self.api_key: str = oa.get("api_key") or ""
        self.model: str = oa.get("model") or "gpt-4o-mini"
        self.temperature: float = float(oa.get("temperature") or 0.2)
        self.session = httpx.AsyncClient(timeout=60)

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }

        # Prefer server-side JSON forcing if supported
        if response_schema is not None:
            body["response_format"] = response_schema
        else:
            body["response_format"] = {"type": "json_object"}

        r = await self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=body)

        # Fallbacks for servers that don’t support response_format
        if r.status_code == 400 and "response_format.type" in r.text and "json_schema" in r.text:
            schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "wp_po_translations",
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
                                    "properties": {
                                        "key": {"type": "string"},
                                        "text": {"type": "string"},
                                    },
                                },
                            }
                        },
                    },
                },
            }
            body["response_format"] = schema
            r = await self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=body)

        if r.status_code == 400 and "response_format" in r.text:
            # Last resort: no response_format; we’ll parse ourselves
            body.pop("response_format", None)
            r = await self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=body)

        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()
        debug_push({"where": "openai_compat.response", "json": data})
        return data

    async def translate_batch(
        self,
        batch: List[Dict[str, Any]],
        source_lang: str,
        target_locale: str,
        system_prompt: str,
        glossary: str,
    ) -> Dict[str, str]:
        """
        Input batch: [{"key": "...", "text": "...", "expected_placeholders": {...}}, ...]
        Output: dict { key: text }
        """
        sys = (system_prompt or "").strip() or "You are a translation engine for WordPress PO strings."

        rules_base = (
            "Translate every item from the source language to the target locale provided. "
            "You MUST output strict JSON and NOTHING else. "
            f"Wrap the JSON between the markers {BEGIN} and {END}. "
            'The JSON must be {"items":[{"key":"...","text":"..."}]}. '
            "Preserve placeholders exactly: %s, %1$s, %d, {name}, {{var}}, and HTML tags. "
            "If key ends with |p0, |p1, ... produce the correct plural form for that index. "
            "Do NOT echo the source text unless it is already in the target language.\n"
            f"{_FEWSHOT}"
        )
        if glossary.strip():
            rules_base += f"\nUntranslatable/fixed terms (keep exactly as-is): {glossary.strip().replace('\n', ', ')}."

        # IMPORTANT: coerce keys to strings going OUT
        items_min = [{"key": str(it["key"]), "text": it["text"]} for it in batch]
        base_payload = {
            "source_language": source_lang,
            "target_locale": target_locale,
            "items": items_min,
        }

        def make_messages(extra_rule: str = ""):
            content = json.dumps(base_payload, ensure_ascii=False)
            return [
                {"role": "system", "content": f"{sys}\n{rules_base}\n{extra_rule}".strip()},
                {"role": "user", "content": content},
            ]

        # First attempt
        data = await self._chat(make_messages())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = extract_json_safely(content)

        # Detect echoes (unchanged outputs)
        src_map = {it["key"]: it["text"] for it in items_min}
        unchanged = [it for it in parsed["items"] if src_map.get(it["key"]) == it["text"]]

        # If too many echoes (>= 20% of batch), retry once with a stricter rule
        if unchanged and len(unchanged) >= max(1, len(items_min) // 5):
            strict_rule = (
                "Never return the source text verbatim. "
                "If the source is already in the target language you may keep it, "
                "otherwise you MUST translate into the target locale."
            )
            data2 = await self._chat(make_messages(strict_rule))
            content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = extract_json_safely(content2)

        out: Dict[str, str] = {}
        for it in parsed["items"]:
            out[str(it["key"])] = it["text"]
        return out
