from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import httpx

# Optional debug recorder used by /debug/provider/recent
try:
    from app.utils.debug_buffer import record as dbg_record
except Exception:  # pragma: no cover
    def dbg_record(*args, **kwargs):  # type: ignore
        pass


def extract_json_safely(content: str) -> dict:
    """
    Find the outermost {...} in 'content' and parse it.
    Raise ValueError if no single JSON object can be found.
    """
    if not isinstance(content, str):
        raise ValueError("Provider content is not a string")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in provider content")
    return json.loads(content[start : end + 1])


def _json_schema_for_keys(keys: List[str]) -> dict:
    """
    JSON Schema to force:
      - top-level object with 'items'
      - items is an array with EXACTLY len(keys) elements
      - each element is {key,text} where key ∈ keys
      - unique keys
    LM Studio accepts response_format.type = 'json_schema'.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "wp_translation_items",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["items"],
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": len(keys),
                        "maxItems": len(keys),
                        "uniqueItems": True,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["key", "text"],
                            "properties": {
                                "key": {"type": "string", "enum": keys},
                                "text": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
    }


class OpenAICompatProvider:
    """
    Back-compatible constructor:

      A) OpenAICompatProvider(settings_obj)
      B) OpenAICompatProvider(base_url, api_key, model, temperature=0.2, timeout_seconds=900)
    """

    def __init__(
        self,
        base_url_or_settings: Any,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout_seconds: int = 900,
    ):
        # Pattern A: settings object
        if api_key is None and model is None and not isinstance(base_url_or_settings, str):
            s = base_url_or_settings
            oc = getattr(s, "openai_compat", None)
            if oc is None:
                raise ValueError("Settings object missing 'openai_compat'")
            def get(obj, key, default=None):
                return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
            base_url = get(oc, "base_url")
            api_key = get(oc, "api_key") or None
            model = get(oc, "model")
            temperature = float(get(oc, "temperature", temperature))
        else:
            # Pattern B: explicit args
            base_url = base_url_or_settings

        if not base_url:
            raise ValueError("OpenAI-compatible base_url is required")
        if not model:
            raise ValueError("OpenAI-compatible model is required")

        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.model = str(model)
        self.temperature = float(temperature)

        # Generous timeouts to avoid "Client disconnected" on local servers
        self.timeout = httpx.Timeout(
            timeout=None,
            connect=30.0,
            read=float(timeout_seconds),
            write=120.0,
            pool=120.0,
        )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.headers = headers

    async def translate_batch(
        self,
        batch: List[dict],
        source_lang: str,
        target_locale: str,
        system_prompt: str,
        glossary: str,
    ) -> Dict[str, str]:
        if len(batch) > 300:
            raise ValueError("Batch too large (>300)")

        # Stable keys for this batch (global indices or ctx|id)
        expected_keys: List[str] = [str(it["key"]) for it in batch]
        payload_items = [{"key": it["key"], "text": it["text"]} for it in batch]

        rules = (
            "Return STRICT JSON only. Do not add commentary. "
            "Return an object with one field: items. "
            "items is an array with EXACTLY the same number of elements as you received. "
            "Each element is {key,text}. Use the SAME key values you received. "
            "Preserve placeholders like %s, {name}, and HTML tags."
        )

        base_body = {
            "model": self.model,
            "temperature": self.temperature,
            # many local servers accept max_tokens to allow long outputs
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system_prompt or "You translate WordPress strings. " + rules},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instructions": rules + " Return items in the SAME ORDER as input.",
                            "source_lang": source_lang,
                            "target_locale": target_locale,
                            "glossary": glossary or "",
                            "items": payload_items,
                            "keys": expected_keys,  # redundancy helps the model
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        dbg_record({"provider": "openai_compat", "dir": "request", "n": len(payload_items), "model": self.model})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Try JSON Schema with exact keys
            body = dict(base_body)
            body["response_format"] = _json_schema_for_keys(expected_keys)
            try:
                r = await client.post(url, headers=self.headers, json=body)
            except httpx.ReadTimeout as e:
                raise ValueError(f"Read timeout contacting provider: {e}") from e
            except httpx.ConnectError as e:
                raise ValueError(f"Cannot connect to provider: {e}") from e
            except httpx.HTTPError as e:
                raise ValueError(f"HTTP error contacting provider: {e}") from e

            # Fallback to text mode if the server rejects json_schema
            if r.status_code == 400:
                schema_err_snippet = r.text[:200]
                r = await client.post(url, headers=self.headers, json=base_body)
                if r.status_code >= 400:
                    raise ValueError(
                        f"Provider HTTP 400 with schema, then HTTP {r.status_code}: {r.text[:300]} | First error: {schema_err_snippet}"
                    )

        if r.status_code >= 400:
            raise ValueError(f"Provider HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()
        # Try multiple common shapes
        content: Optional[str] = None
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = data.get("choices", [{}])[0].get("text") or data.get("response")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Unexpected provider schema or empty content. Body snippet: {str(data)[:300]}")

        dbg_record({"provider": "openai_compat", "dir": "response", "snippet": content[:200]})

        # Parse JSON; unwrap {"response":"..."} if necessary
        try:
            obj = extract_json_safely(content)
            unwrap_guard = 0
            while "items" not in obj and isinstance(obj.get("response"), str) and unwrap_guard < 2:
                obj = extract_json_safely(obj["response"])
                unwrap_guard += 1
        except Exception as e:
            raise ValueError(f"Provider returned non-JSON or truncated JSON: {e}. Content snippet: {content[:200]}") from e

        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError("JSON missing 'items' array")

        # Build output map with two fallbacks:
        #  1) prefer exact keys as returned (if they match expected_keys)
        #  2) if keys are wrong but lengths match, remap by position
        out: Dict[str, str] = {}

        returned_keys = [str(it.get("key")) for it in items if isinstance(it, dict)]
        dbg_record({"provider": "openai_compat", "dir": "parsed", "count": len(items), "keys_sample": returned_keys[:5]})

        # Case A: accept any correctly keyed subset
        for it in items:
            if not isinstance(it, dict):
                continue
            k = it.get("key")
            v = it.get("text")
            if k is None or v is None:
                continue
            k = str(k)
            if k in expected_keys:
                out[k] = str(v)

        # Case B: lengths match but keys are wrong or repeated -> remap by position
        if len(out) < len(items) and len(items) == len(expected_keys):
            remapped = 0
            for idx, it in enumerate(items):
                v = it.get("text")
                if v is None:
                    continue
                key_for_position = expected_keys[idx]
                if key_for_position not in out:
                    out[key_for_position] = str(v)
                    remapped += 1
            if remapped:
                dbg_record({"provider": "openai_compat", "dir": "remap_by_position", "remapped": remapped})

        return out
