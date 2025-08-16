# backend/app/providers/openai_compat.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import aiohttp


def _extract_json_safely(text: Any) -> Dict[str, Any]:
    """
    Try to robustly extract a JSON object from model output.
    Accepts:
      - a Python dict already
      - a pure JSON string
      - markdown-fenced ```json ... ``` blocks
      - strings with leading/trailing prose, where we take the largest {...} block
    Raises ValueError if we cannot recover a valid JSON object.
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        raise ValueError("Provider returned non-string content")

    s = text.strip()

    # 1) Look for fenced json code block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
        try:
            return json.loads(candidate)
        except Exception as e:
            raise ValueError(f"Provider returned fenced non-JSON: {e}. Content snippet: {candidate[:240]}")

    # 2) Try direct json
    try:
        return json.loads(s)
    except Exception:
        pass

    # 3) Try to grab the largest {...} span
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            raise ValueError(
                f"Provider returned non-JSON or truncated JSON: {e}. "
                f"Content snippet: {candidate[:240]}"
            )

    # 4) Give up
    raise ValueError(f"Provider returned non-JSON without any brace block. Content snippet: {s[:240]}")


def _items_list_to_kv(d: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize a model response into {key: text}.

    Acceptable shapes:
      - {"items": [{"key": "...","text":"..."}, ...]}
      - {"0":"...", "1":"..."}  not preferred, but we will accept
    """
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

    # fallback if model returns a raw map
    if isinstance(d, dict):
        out: Dict[str, str] = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float)):
                out[str(k)] = str(v)
        if out:
            return out

    return {}


class OpenAICompatProvider:
    """
    Generic OpenAI-compatible chat.completions provider.
    Works with:
      - OpenAI API
      - OpenRouter OpenAI-compatible endpoint
      - Other gateways that mirror /v1/chat/completions

    Expected output is STRICT JSON:
      {"items":[{"key":"<same as input>", "text":"<translated>"} ...]}
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
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
        """
        Translate a batch of items.

        batch = [
          {"key":"0","text":"Settings","context":"noun"},
          {"key":"1","text":"%1$s file","context":""},
          ...
        ]

        Returns:
          { "0": "Ajustes", "1": "%1$s archivo", ... }
        """
        expected_keys = [str(it["key"]) for it in batch]

        payload_items = []
        for it in batch:
            item = {"key": str(it["key"]), "text": str(it["text"])}
            if it.get("context"):
                # Context included only in the input. The model must NOT echo it back.
                item["context"] = str(it["context"])
            payload_items.append(item)

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
            "items": payload_items,
            "keys": expected_keys,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update(self.extra_headers)

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt or "You translate WordPress strings accurately."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"

        # Manage session lifecycle correctly: close if we created it
        close_session = False
        if http_session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            http_session = aiohttp.ClientSession(timeout=timeout)
            close_session = True

        data: Dict[str, Any]
        try:
            async with http_session.post(url, headers=headers, json=body) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise ValueError(f"HTTP {resp.status}: {txt[:240]}")
                # Try parsing the full response as JSON first
                try:
                    data = json.loads(txt)
                except json.JSONDecodeError:
                    # Some servers return raw assistant content only
                    data = {"choices": [{"message": {"content": txt}}]}
        finally:
            if close_session:
                await http_session.close()

        # Extract assistant content
        content = ""
        try:
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                # OpenAI style
                content = data["choices"][0]["message"]["content"]
            elif isinstance(data, dict) and "output_text" in data:
                # Some proxies return a flat output_text
                content = data["output_text"]
            else:
                # Fallback to entire payload as content
                content = json.dumps(data)
        except Exception:
            content = json.dumps(data)

        # Parse JSON from content
        parsed = _extract_json_safely(content)
        mapping = _items_list_to_kv(parsed)

        # As a last resort, if mapping is empty try parse content as {"key":"text",...}
        if not mapping:
            try:
                maybe_map = json.loads(content)
                if isinstance(maybe_map, dict):
                    mapping = {str(k): str(v) for k, v in maybe_map.items() if isinstance(v, (str, int, float))}
            except Exception:
                pass

        return mapping


# --- public aliases for external imports (keep at bottom of file) ---
extract_json_safely = _extract_json_safely
items_list_to_kv = _items_list_to_kv
