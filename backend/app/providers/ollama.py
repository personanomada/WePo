from __future__ import annotations
import json
from typing import Dict, List, Optional, Any

import httpx


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
    Backward-compatible constructor:
      - OllamaProvider(settings_obj)
      - OllamaProvider(host, model, temperature=0.2, timeout_seconds=900)
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

        user_payload = [{"key": it["key"], "text": it["text"]} for it in batch]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "Return ONLY strict JSON."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_lang": source_lang,
                            "target_locale": target_locale,
                            "glossary": glossary or "",
                            "items": user_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "options": {"temperature": self.temperature},
            "stream": False,
        }

        url = f"{self.host}/api/chat"
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

        try:
            obj = _extract_json_object(content)
        except Exception as e:
            raise ValueError(f"Ollama returned non-JSON or truncated JSON: {e}. Content snippet: {content[:200]}") from e

        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError("JSON missing 'items' array")

        out: Dict[str, str] = {}
        for it in items:
            k = it.get("key")
            v = it.get("text")
            if k is None or v is None:
                raise ValueError("Each item must have 'key' and 'text'")
            out[str(k)] = str(v)

        return out
