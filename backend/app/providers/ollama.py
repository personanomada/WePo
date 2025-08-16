# backend/app/providers/ollama.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import aiohttp


def _extract_json_safely(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError("Provider returned non-string content")

    s = text.strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
        return json.loads(candidate)

    try:
        return json.loads(s)
    except Exception:
        pass

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first : last + 1]
        return json.loads(candidate)

    raise ValueError(f"Provider returned non-JSON. Content snippet: {s[:240]}")


def _items_list_to_kv(d: Dict[str, Any]) -> Dict[str, str]:
    if isinstance(d, dict) and "items" in d and isinstance(d["items"], list):
        out: Dict[str, str] = {}
        for it in d["items"]:
            if not isinstance(it, dict):
                continue
            k = str(it.get("key", "")).strip()
            v = it.get("text", "")
            if k:
                out[k] = "" if v is None else str(v)
        return out

    if isinstance(d, dict):
        out: Dict[str, str] = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float)):
                out[str(k)] = str(v)
        if out:
            return out

    return {}


class OllamaProvider:
    """
    Ollama chat provider (local).
    Defaults to http://localhost:11434 but can be pointed elsewhere.
    Expects STRICT JSON output identical to OpenAICompatProvider.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        timeout: int = 120,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.extra_headers = extra_headers or {}

    async def translate_batch(
        self,
        *,
        batch: List[Dict[str, Any]],
        source_lang: str,
        target_locale: str,
        system_prompt: Optional[str] = None,
        glossary: Optional[str] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, str]:
        expected_keys = [str(it["key"]) for it in batch]

        items = []
        for it in batch:
            item = {"key": str(it["key"]), "text": str(it["text"])}
            if it.get("context"):
                item["context"] = str(it["context"])
            items.append(item)

        rules = (
            "Return STRICT JSON only. Do not add commentary. "
            "Return an object with one field: items. "
            "items is an array with EXACTLY the same number of elements as you received. "
            "Each element is an object with fields key and text. "
            "Use the SAME key values you received. "
            "Return items in the SAME ORDER as input. "
            "Preserve placeholders like %s, %d, %1$s, {name}, and keep HTML tags unchanged. "
            "Do NOT add or remove placeholders. "
            "If an input item has a 'context' field, use it for disambiguation, but do NOT include it in the output."
        )

        user_payload = {
            "instructions": rules,
            "source_lang": source_lang,
            "target_locale": target_locale,
            "glossary": glossary or "",
            "items": items,
            "keys": expected_keys,
        }

        headers = {"Content-Type": "application/json"}
        if self.extra_headers:
            headers.update(self.extra_headers)

        # Ollama chat endpoint
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt or "You translate WordPress strings accurately."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

        close_session = False
        if http_session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            http_session = aiohttp.ClientSession(timeout=timeout)
            close_session = True

        try:
            async with http_session.post(url, headers=headers, json=body) as resp:
                payload_text = await resp.text()
                if resp.status >= 400:
                    raise ValueError(f"HTTP {resp.status}: {payload_text[:240]}")
                try:
                    data = json.loads(payload_text)
                except json.JSONDecodeError:
                    # Some Ollama builds return raw content; normalize
                    data = {"message": {"content": payload_text}}
        finally:
            if close_session:
                await http_session.close()

        # Extract assistant content from Ollama
        content = ""
        if isinstance(data, dict):
            # Newer Ollama chat API:
            #   {"message":{"role":"assistant","content":"..."}, "done": true}
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                content = data["message"]["content"]
            # Older generate API compatibility:
            elif "response" in data and isinstance(data["response"], str):
                content = data["response"]
            else:
                content = json.dumps(data)
        else:
            content = str(data)

        parsed = _extract_json_safely(content)
        mapping = _items_list_to_kv(parsed)

        if not mapping:
            # last-chance parser
            try:
                maybe_map = json.loads(content)
                if isinstance(maybe_map, dict):
                    mapping = {str(k): str(v) for k, v in maybe_map.items() if isinstance(v, (str, int, float))}
            except Exception:
                pass

        return mapping

# --- public aliases for external imports ---
extract_json_safely = _extract_json_safely
items_list_to_kv = _items_list_to_kv
