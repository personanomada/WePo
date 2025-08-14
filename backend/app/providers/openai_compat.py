# backend/app/providers/openai_compat.py
from __future__ import annotations
import json
from typing import Dict, List, Optional

import httpx


def _extract_json_object(s: str) -> dict:
    """
    Be tolerant to extra prose. Grab the outermost { ... }.
    """
    if not isinstance(s, str):
        raise ValueError("Provider 'content' is not a string")
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in provider content")
    return json.loads(s[start:end + 1])


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        model: str,
        temperature: float = 0.2,
        timeout_seconds: int = 900,  # 15 minutes read timeout for slow local models
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        # Generous timeouts prevent "Client disconnected" on LM Studio
        self.timeout = httpx.Timeout(
            timeout=None,            # total
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
        system = system_prompt or "You return ONLY strict JSON."

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
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
            "response_format": {"type": "json_object"},  # ignored by some servers, harmless
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

        # If server replied with HTTP error, include body
        if r.status_code >= 400:
            detail = r.text[:500]
            raise ValueError(f"Provider HTTP {r.status_code}: {detail}")

        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Unexpected provider schema: {e}. Body snippet: {str(data)[:300]}") from e

        try:
            obj = _extract_json_object(content)
        except Exception as e:
            raise ValueError(f"Provider returned non-JSON or truncated JSON: {e}. Content snippet: {content[:200]}") from e

        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError("JSON missing 'items' array")

        out: Dict[str, str] = {}
        for it in items:
            k, v = it.get("key"), it.get("text")
            if k is None or v is None:
                raise ValueError("Each item must have 'key' and 'text'")
            out[str(k)] = str(v)
        return out
