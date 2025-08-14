from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from ..schemas import Settings
from ..utils.debug_buffer import push as debug_push
from .openai_compat import extract_json_safely, BEGIN, END  # reuse extractor + markers

_FEWSHOT = (
    'Example input:\n'
    '{"items":[{"key":"A|p0","text":"%d result"},{"key":"A|p1","text":"%d results"},{"key":"B","text":"Add to cart"}]}\n'
    'Example output (Spanish):\n'
    f'{BEGIN}{{"items":[{{"key":"A|p0","text":"%d resultado"}},{{"key":"A|p1","text":"%d resultados"}},{{"key":"B","text":"Añadir al carrito"}}]}}{END}'
)

class OllamaProvider:
    """
    Provider for native Ollama / LM Studio compatible endpoints.
    Implements translate_batch -> { key: text }.
    """

    def __init__(self, settings: Settings):
        s = settings.normalized()
        ol = s.ollama
        self.host: str = (ol.get("host") or "http://localhost:11434").rstrip("/")
        self.model: str = ol.get("model") or "llama3.1"
        self.temperature: float = float(ol.get("temperature") or 0.2)
        self.session = httpx.AsyncClient(timeout=60)

    async def _chat_or_generate(self, prompt_obj: Dict[str, Any]) -> str:
        """
        Try /api/chat first, then fallback to /api/generate.
        Return textual content to parse.
        """
        # --- /api/chat ---
        try:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt_obj["system"]},
                    {"role": "user", "content": prompt_obj["user"]},
                ],
                "options": {"temperature": self.temperature},
                "stream": False,
            }
            r = await self.session.post(f"{self.host}/api/chat", json=body)
            if r.status_code < 400:
                data = r.json()
                debug_push({"where": "ollama.chat.response", "json": data})
                content = data.get("message", {}).get("content", "")
                if content:
                    return content
        except Exception as e:
            debug_push({"where": "ollama.chat.error", "error": str(e)})

        # --- /api/generate ---
        body = {
            "model": self.model,
            "prompt": f"{prompt_obj['system']}\n\n{prompt_obj['user']}",
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        r = await self.session.post(f"{self.host}/api/generate", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        debug_push({"where": "ollama.generate.response", "json": data})
        return data.get("response", "")

    async def translate_batch(
        self,
        batch: List[Dict[str, Any]],
        source_lang: str,
        target_locale: str,
        system_prompt: str,
        glossary: str,
    ) -> Dict[str, str]:
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

        def make_prompt(extra_rule: str = ""):
            user = json.dumps(base_payload, ensure_ascii=False)
            return {
                "system": f"{sys}\n{rules_base}\n{extra_rule}".strip(),
                "user": user,
            }

        # First attempt
        content = await self._chat_or_generate(make_prompt())
        parsed = extract_json_safely(content)

        # Detect echoes (unchanged outputs)
        src_map = {it["key"]: it["text"] for it in items_min}
        unchanged = [it for it in parsed["items"] if src_map.get(it["key"]) == it["text"]]

        # If too many echoes (>= 20%), retry once with stricter rule
        if unchanged and len(unchanged) >= max(1, len(items_min) // 5):
            strict = (
                "Never return the source text verbatim; you MUST translate into the target locale "
                "unless it is already in that language."
            )
            content2 = await self._chat_or_generate(make_prompt(strict))
            parsed = extract_json_safely(content2)

        out: Dict[str, str] = {}
        for it in parsed["items"]:
            out[str(it["key"])] = it["text"]
        return out
