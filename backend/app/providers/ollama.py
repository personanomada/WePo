import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("wepo.providers.ollama")


def _extract_json_safely(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _items_list_to_kv(
    data: Any,
    expected_keys: Optional[List[str]] = None,
) -> Dict[str, str]:
    kv: Dict[str, str] = {}

    def norm(v: Any) -> str:
        return "" if v is None else str(v)

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
        for it in items:
            if isinstance(it, dict) and "key" in it and "text" in it:
                kv[str(it["key"])] = norm(it["text"])
        if expected_keys and kv and not any(k in expected_keys for k in kv.keys()):
            if len(items) == len(expected_keys):
                kv = {str(expected_keys[i]): norm(items[i].get("text")) for i in range(len(items))}
        return kv

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float)) or v is None:
                kv[str(k)] = norm(v)
        if kv and expected_keys is not None:
            kv = {k: v for k, v in kv.items() if k in expected_keys}
        return kv

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


class OllamaProvider:
    def __init__(self, settings: dict):
        self.base_url = (settings or {}).get("base_url") or (settings or {}).get("host") or "http://127.0.0.1:11434"
        self.model = (settings or {}).get("model", "llama3")
        self.use_json_format = True
        self.trace = bool((settings or {}).get("trace", False))
        self.trace_truncate = int((settings or {}).get("trace_truncate", 2000))

    def _t(self, event: str, **fields: Any) -> None:
        if not self.trace:
            return
        trunc = int(self.trace_truncate)
        payload = {}
        for k, v in fields.items():
            if isinstance(v, str) and len(v) > trunc:
                payload[k] = v[:trunc] + f"... [truncated {len(v)-trunc}]"
            else:
                payload[k] = v
        try:
            logger.info("ollama %s %s", event, json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.info("ollama %s %r", event, payload)

    async def translate_batch(
        self,
        *,
        batch: List[Dict[str, str]],
        source_lang: str,
        target_locale: str,
        glossary: Optional[str],
        system_prompt: Optional[str] = None,
        timeout_s: int = 60,
        http_session: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, str]:
        expected_keys = [str(it["key"]) for it in batch]

        user_payload = {
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
            "items": [
                {"key": str(it["key"]), "text": str(it["text"]), **({"context": it["context"]} if it.get("context") else {})}
                for it in batch
            ],
            "keys": expected_keys,
        }

        sys = system_prompt or "You are a professional software translator."
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "stream": False,
        }
        if self.use_json_format:
            body["format"] = "json"

        close_client = False
        client = http_session
        if client is None:
            client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)
            close_client = True

        # trace request
        self._t(
            "request",
            model=self.model,
            base_url=self.base_url,
            items=len(user_payload["items"]),
            preview_user=body["messages"][1]["content"],
        )

        try:
            r = await client.post("/api/chat", json=body)
            raw = r.text
            # trace response
            self._t("response", status=r.status_code, preview_raw=raw)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {raw[:500]}")
        finally:
            if close_client:
                await client.aclose()

        try:
            outer = json.loads(raw)
            content = outer.get("message", {}).get("content", "")
            parsed = _extract_json_safely(content)
        except Exception:
            parsed = _extract_json_safely(raw)

        if isinstance(parsed, dict) and {"instructions", "source_lang", "target_locale"}.issubset(set(parsed.keys())):
            parsed = {"items": parsed.get("items", [])}

        mapping = _items_list_to_kv(parsed, expected_keys=expected_keys)
        if mapping and expected_keys:
            mapping = {k: v for k, v in mapping.items() if k in expected_keys}

        # trace mapping sample
        sample = list(mapping.items())[:3]
        self._t("parsed_mapping", count=len(mapping), sample=json.dumps(sample, ensure_ascii=False))

        return mapping


# public aliases so other modules can import for diagnostics
extract_json_safely = _extract_json_safely
items_list_to_kv = _items_list_to_kv
