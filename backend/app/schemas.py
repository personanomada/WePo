from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class Settings(BaseModel):
    provider: str = Field(default="openai_compat")
    system_prompt: str = Field(default="")
    glossary: str = Field(default="")
    batch_size: int = Field(default=50)
    openai_compat: Dict[str, Any] = Field(default_factory=lambda: dict(base_url="", api_key="", model="", temperature=0.2))
    ollama: Dict[str, Any] = Field(default_factory=lambda: dict(host="http://localhost:11434", model="llama3.1", temperature=0.2))

    @validator("batch_size")
    def clamp_batch(cls, v):
        v = int(v or 50)
        return max(10, min(300, v))

    def normalized(self) -> "Settings":
        s = self.model_copy(deep=True)
        s.openai_compat["base_url"] = (s.openai_compat.get("base_url") or "").strip()
        s.openai_compat["model"] = (s.openai_compat.get("model") or "").strip()
        s.ollama["host"] = (s.ollama.get("host") or "").strip()
        s.ollama["model"] = (s.ollama.get("model") or "").strip()
        return s

    def public_copy(self) -> Dict[str, Any]:
        # Don’t leak API keys
        oa = dict(self.openai_compat)
        if "api_key" in oa and oa["api_key"]:
            oa["api_key"] = "****"
        return dict(
            provider=self.provider,
            system_prompt=self.system_prompt,
            glossary=self.glossary,
            batch_size=self.batch_size,
            openai_compat=oa,
            ollama=self.ollama,
        )


class SettingsOut(BaseModel):
    ok: bool
    defaults: Dict[str, Any]


class AnalyzeOut(BaseModel):
    ok: bool
    info: Dict[str, Any]
    audit: Optional[Dict[str, Any]] = None
