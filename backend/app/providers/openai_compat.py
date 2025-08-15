# app/providers/openai_compat.py
from __future__ import annotations
import json
from typing import Dict, List, Optional, Any

import httpx


def extract_json_safely(content: str) -> dict:
    """
    Tolerant JSON extractor used by /providers/verify and batch parsing.
    Finds the outermost { ... } in 'content' and parses it.
    Raises ValueError if not found or invalid.
    """
    if not isinstance(content, str):
        raise ValueError("Provider content is not a string")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in provider content")
    return json.loads(content[start : end + 1])


class OpenAICompatProvider:
    """
    Backward-compatible constructor:
      - Pattern A (your main.py): OpenAICompatProvider(settings_obj)
      - Pattern B: OpenAICompatProvider(base_url, api_key, model, temperature=0.2, timeout_seconds=900)
    """

    def __init__(
        self,
        base_url_or_settings: Any,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout_seconds: int = 900,  # generous for local servers
    ):
        # Pattern A: first arg is a settings object
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

        # Prevent LM Studio “Client disconnected” by allowing long generations
        self.timeout = httpx.Timeout(
            timeout=None,      # total
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

        user_payload = [{"key": it["key"], "text": it["text"]} for it in batch]

        # DO NOT set response_format to avoid LM Studio 400:
        # {"error":"'response_format.type' must be 'json_schema' or 'text'"}
        body = {
            "model": self.model,
            "temperature": self.temperature,
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
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                r = await client.post(url, headers=self.headers, json=body)
            except httpx.ReadTimeout as e:
                raise ValueError(f"Read timeout contacting provider: {e}") from e
            except httpx.ConnectError as e:
                raise ValueError(f"Cannot connect to provider: {e}") from e
            except httpx.HTTPError as e:
                raise ValueError(f"HTTP error contacting provider: {e}") from e

        if r.status_code >= 400:
            # Surface server text to the UI
            raise ValueError(f"Provider HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Unexpected provider schema: {e}. Body snippet: {str(data)[:300]}") from e

        try:
            obj = extract_json_safely(content)
        except Exception as e:
            raise ValueError(
                f"Provider returned non-JSON or truncated JSON: {e}. Content snippet: {content[:200]}"
            ) from e

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
