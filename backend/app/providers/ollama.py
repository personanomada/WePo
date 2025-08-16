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


def _extract_json_object(content: str) -> dict:
    if not isinstance(content, str):
        raise ValueError("Provider content is not a string")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in provider content")
    return json.loads(content[start : end + 1])


class OllamaProvider:
    """
    Back-compatible constructor:

      A) OllamaProvider(settings_obj)
      B) OllamaProvider(host, model, temperature=0.2, timeout_seconds=900)
    """

    def __init__(
        self,
        host_or_settings: Any,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout_seconds: int = 900,
    ):
        if model is None and not isinstance(host_or_settings, str):
            s = host_or_settings
            oc = getattr(s, "ollama", None)
            if oc is None:
                raise ValueError("Settings object missing 'ollama'")
            def get(obj, key, default=None):
                return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
            host = get(oc, "host") or "http://localhost:11434"
            model = get(oc, "model") or "llama3"
            temperature = float(get(oc, "temperature", temperature))
        else:
            host = host_or_settings or "http://localhost:11434"

        self.host = str(host).rstrip("/")
        self.model = str(model)
        self.temperature = float(temperature)

        self.timeout = httpx.Timeout(
            timeout=None,
            connect=30.0,
            read=float(timeout_seconds),
            write=120.0,
            pool=120.0,
        )

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

        expected_keys: List[str] = [str(it["key"]) for it in batch]
        payload_items = [{"key": it["key"], "text": it["text"]} for it in batch]
        rules = (
            "Return STRICT JSON only. Do not add commentary. "
            "Return an object with one field: items. "
            "items is an array with EXACTLY the same number of elements as you received. "
            "Each element is {key,text}. Use the SAME key values you received. "
            "Return items in the SAME ORDER as input. Preserve placeholders and HTML tags."
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You translate WordPress strings. " + rules},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instructions": rules,
                            "source_lang": source_lang,
                            "target_locale": target_locale,
                            "glossary": glossary or "",
                            "items": payload_items,
                            "keys": expected_keys,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "options": {"temperature": self.temperature},
            "stream": False,
        }

        url = f"{self.host}/api/chat"
        dbg_record({"provider": "ollama", "dir": "request", "n": len(payload_items), "model": self.model})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                r = await client.post(url, json=body)
            except httpx.ReadTimeout as e:
                raise ValueError(f"Ollama read timeout: {e}") from e
            except httpx.ConnectError as e:
                raise ValueError(f"Ollama connect error: {e}") from e
            except httpx.HTTPError as e:
                raise ValueError(f"Ollama HTTP error: {e}") from e

        if r.status_code >= 400:
            raise ValueError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()
        try:
            content = data["message"]["content"]
        except Exception as e:
            raise ValueError(f"Ollama unexpected schema: {e}. Body snippet: {str(data)[:300]}") from e

        dbg_record({"provider": "ollama", "dir": "response", "snippet": content[:200]})

        try:
            obj = _extract_json_object(content)
        except Exception as e:
            raise ValueError(f"Ollama returned non-JSON or truncated JSON: {e}. Content snippet: {content[:200]}") from e

        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError("JSON missing 'items' array")

        # Build output with the same key-rescue strategy
        out: Dict[str, str] = {}
        returned_keys = [str(it.get("key")) for it in items if isinstance(it, dict)]
        dbg_record({"provider": "ollama", "dir": "parsed", "count": len(items), "keys_sample": returned_keys[:5]})

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
                dbg_record({"provider": "ollama", "dir": "remap_by_position", "remapped": remapped})

        return out
