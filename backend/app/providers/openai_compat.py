from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("wepo.providers.openai_compat")

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_safely(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    s = text.strip()
    m = _JSON_BLOCK.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        pass
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(s[first : last + 1])
        except Exception:
            return {}
    return {}


def _items_list_to_kv(data: Any, expected_keys: Optional[List[str]] = None) -> Dict[str, str]:
    kv: Dict[str, str] = {}

    def norm(v: Any) -> str:
        return "" if v is None else str(v)

    # canonical
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
        for it in items:
            if isinstance(it, dict) and "key" in it and "text" in it:
                kv[str(it["key"])] = norm(it["text"])
        if expected_keys and kv and not any(k in expected_keys for k in kv.keys()):
            if len(items) == len(expected_keys):
                kv = {str(expected_keys[i]): norm(items[i].get("text")) for i in range(len(items))}
        return kv

    # map
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float)) or v is None:
                kv[str(k)] = norm(v)
        if kv and expected_keys is not None:
            kv = {k: v for k, v in kv.items() if k in expected_keys}
        return kv

    # bare array
    if isinstance(data, list):
        items = data
        for it in items:
            if isinstance(it, dict) and "key" in it and "text" in it:
                kv[str(it["key"])] = norm(it["text"])
        if expected_keys and (not kv or not any(k in expected_keys for k in kv.keys())):
            if len(items) == len(expected_keys):
                kv = {str(expected_keys[i]): norm(items[i].get("text")) for i in range(len(items))}
        return kv

    return kv


class OpenAICompatProvider:
    """
    OpenAI-compatible chat completions provider.
    Constructor matches your codebase: a single `settings: dict`.
    """

    def __init__(self, settings: dict):
        s = settings or {}
        self.base_url = (s.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = s.get("api_key") or ""
        self.model = s.get("model") or "gpt-4o-mini"
        self.temperature = float(s.get("temperature", 0.2))
        self.timeout = int(s.get("timeout", 60))
        self.extra_headers = s.get("extra_headers") or s.get("headers") or {}
        # tracing controls
        self.trace = bool(s.get("trace", False))
        self.trace_truncate = int(s.get("trace_truncate", 2000))
        # response_format controls
        self.use_response_format = bool(s.get("use_response_format", True))
        # cached response_format mode: "json_object" | "json_schema" | "none"
        self.rf_mode: str = "json_object"

    def _t(self, event: str, **fields: Any) -> None:
        if not self.trace:
            return
        # Truncate large blobs
        trunc = int(self.trace_truncate)
        payload = {}
        for k, v in fields.items():
            if isinstance(v, str) and len(v) > trunc:
                payload[k] = v[:trunc] + f"... [truncated {len(v)-trunc}]"
            else:
                payload[k] = v
        try:
            logger.info("oa_compat %s %s", event, json.dumps(payload, ensure_ascii=False))
        except Exception:
            # do not let logging break requests
            logger.info("oa_compat %s %r", event, payload)

    async def translate_batch(
        self,
        *,
        batch: List[Dict[str, Any]],
        source_lang: str,
        target_locale: str,
        glossary: Optional[str] = None,
        system_prompt: Optional[str] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, str]:
        expected_keys = [str(it["key"]) for it in batch]

        def _normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for it in items:
                row = {"key": str(it["key"]), "text": str(it["text"])}
                if it.get("context"):
                    row["context"] = str(it["context"])
                out.append(row)
            return out

        payload = {
            "instructions": (
                "Return STRICT JSON only. Do not add commentary. "
                "Return an object with one field: items. "
                "items is an array with EXACTLY the same number of elements as you received. "
                "Each element is an object with fields key and text. "
                "Use the SAME key values you received. "
                "Return items in the SAME ORDER as input. "
                "Preserve placeholders like %s, %d, %1$s, {name}, and keep HTML tags unchanged. "
                "Do NOT add or remove placeholders. "
                "If an input item has a 'context' field, use it for disambiguation, but do NOT include it in the output."
            ),
            "source_lang": source_lang,
            "target_locale": target_locale,
            "glossary": glossary or "",
            "items": _normalize_items(batch),
            "keys": expected_keys,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update({str(k): str(v) for k, v in self.extra_headers.items()})

        def _base_body():
            return {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt or "You translate WordPress strings accurately."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "stream": False,
            }

        JSON_ITEMS_SCHEMA = {
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

        async def _post_with_mode(session: aiohttp.ClientSession, mode: str) -> Dict[str, Any]:
            body = _base_body()
            if self.use_response_format:
                if mode == "json_object":
                    body["response_format"] = {"type": "json_object"}
                elif mode == "json_schema":
                    body["response_format"] = JSON_ITEMS_SCHEMA
            url = f"{self.base_url}/chat/completions"

            # trace request
            self._t(
                "request",
                mode=mode,
                model=self.model,
                base_url=self.base_url,
                items=len(payload["items"]),
                user_bytes=len(body["messages"][1]["content"]),
                preview_user=body["messages"][1]["content"],
            )

            async with session.post(url, headers=headers, json=body) as resp:
                txt = await resp.text()
                # trace response
                self._t("response", mode=mode, status=resp.status, preview_raw=txt)
                if resp.status >= 400:
                    raise ValueError(f"HTTP {resp.status}: {txt}")
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    return {"choices": [{"message": {"content": txt}}]}

        # manage session lifecycle
        close_session = False
        if http_session is None:
            http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            close_session = True

        try:
            data: Optional[Dict[str, Any]] = None
            last_err = None

            # try cached mode first, then degrade
            modes = [self.rf_mode] if self.use_response_format else ["none"]
            if self.use_response_format:
                if self.rf_mode == "json_object":
                    modes += ["json_schema", "none"]
                elif self.rf_mode == "json_schema":
                    modes += ["none"]

            for mode in modes:
                try:
                    data = await _post_with_mode(http_session, mode)
                    self.rf_mode = mode  # cache winner
                    break
                except ValueError as e:
                    msg = str(e)
                    last_err = msg
                    if "'response_format.type' must be 'json_schema' or 'text'" in msg and mode != "json_schema":
                        continue
                    if "response_format" in msg and mode != "none":
                        continue
                    raise

            if data is None:
                raise ValueError(last_err or "no data")

        finally:
            if close_session:
                await http_session.close()

        # extract assistant content
        content = ""
        try:
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = json.dumps(data, ensure_ascii=False)

        parsed = _extract_json_safely(content)

        # trace parsed content before normalization
        self._t("assistant_content", mode=self.rf_mode, preview_content=content)

        # handle echo of our envelope
        if isinstance(parsed, dict) and {"instructions", "source_lang", "target_locale"}.issubset(parsed.keys()):
            parsed = {"items": parsed.get("items", [])}

        mapping = _items_list_to_kv(parsed, expected_keys=expected_keys)
        if mapping and expected_keys:
            mapping = {k: v for k, v in mapping.items() if k in expected_keys}

        # trace final mapping sample
        sample = list(mapping.items())[:3]
        self._t("parsed_mapping", mode=self.rf_mode, count=len(mapping), sample=json.dumps(sample, ensure_ascii=False))

        return mapping


# public aliases if other modules import them
extract_json_safely = _extract_json_safely
items_list_to_kv = _items_list_to_kv
