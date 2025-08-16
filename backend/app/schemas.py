# app/schemas.py
from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AnalyzeOut(BaseModel):
    ok: bool = True
    info: Dict[str, Any] = Field(default_factory=dict)
    audit: Dict[str, Any] = Field(default_factory=dict)


class UISettings(BaseModel):
    # Tool-level UX toggles not tied to any provider
    show_test_run: bool = False
    default_sample_size: int = 25


class Settings(BaseModel):
    provider: str = "openai_compat"

    # Global prompting controls
    system_prompt: str = ""
    glossary: str = ""

    # Batch size for translation requests
    batch_size: int = 40

    # Provider-specific blocks
    openai_compat: Dict[str, Any] = Field(default_factory=dict)
    ollama: Dict[str, Any] = Field(default_factory=dict)

    # UI / diagnostics block (new)
    ui: UISettings = Field(default_factory=UISettings)

    def normalized(self) -> "Settings":
        """Coerce defaults and expected types for nested dicts."""
        s = self.model_copy(deep=True)

        # openai_compat defaults
        oac = dict(s.openai_compat or {})
        oac.setdefault("base_url", "http://127.0.0.1:1234/v1")
        oac.setdefault("api_key", "")
        oac.setdefault("model", "openai/gpt-oss-20b")
        oac["temperature"] = float(oac.get("temperature", 0.2))
        # tracing + response_format controls (new)
        oac["trace"] = bool(oac.get("trace", False))
        oac["trace_truncate"] = int(oac.get("trace_truncate", 2000))
        oac["use_response_format"] = bool(oac.get("use_response_format", True))
        s.openai_compat = oac

        # ollama defaults
        ol = dict(s.ollama or {})
        ol.setdefault("host", "http://localhost:11434")
        ol.setdefault("base_url", ol.get("host"))  # allow either key
        ol.setdefault("model", "llama3.1")
        ol["temperature"] = float(ol.get("temperature", 0.2))
        # tracing controls (new)
        ol["trace"] = bool(ol.get("trace", False))
        ol["trace_truncate"] = int(ol.get("trace_truncate", 2000))
        s.ollama = ol

        # ui defaults (new)
        ui = s.ui or UISettings()
        ui.show_test_run = bool(getattr(ui, "show_test_run", False))
        try:
            ui.default_sample_size = int(getattr(ui, "default_sample_size", 25))
        except Exception:
            ui.default_sample_size = 25
        s.ui = ui

        # top-level defaults
        s.batch_size = int(getattr(s, "batch_size", 40))
        s.provider = (s.provider or "openai_compat").lower().replace("-", "_")

        return s

    def public_copy(self) -> Dict[str, Any]:
        """
        Strip secrets and return a JSON-safe dict for the frontend defaults.
        """
        s = self.normalized()
        oac = dict(s.openai_compat or {})
        # redact secrets
        if "api_key" in oac and oac["api_key"]:
            oac["api_key"] = "******"

        return {
            "provider": s.provider,
            "system_prompt": s.system_prompt,
            "glossary": s.glossary,
            "batch_size": s.batch_size,
            "openai_compat": oac,
            "ollama": dict(s.ollama or {}),
            "ui": s.ui.model_dump(),
        }


class SettingsOut(BaseModel):
    ok: bool = True
    defaults: Dict[str, Any] = Field(default_factory=dict)
