import React, { useEffect, useState } from "react";
import { Button } from "./ui/Button";
import { Input } from "./ui/Input";
import { Label } from "./ui/Label";
import { Textarea } from "./ui/Textarea";
import { Select } from "./ui/Select";

type Settings = {
  provider: "openai_compat" | "ollama";
  system_prompt: string;
  glossary: string;
  batch_size: number;
  openai_compat: { base_url: string; api_key: string; model: string; temperature: number };
  ollama: { host: string; model: string; temperature: number };
};

// Default system prompt for AI
const DEFAULT_SYSTEM_PROMPT = `Your task is to translate text for a WordPress plugin.
You will receive a JSON array of objects with "key" and "text" properties.
You MUST return a single, valid JSON object and nothing else.
Your JSON response MUST contain one key, "items", which is an array of objects. Each object must have a "key" and a "text" property containing the translation.
Preserve all placeholders (e.g., %s, %d, <strong>, <a>) exactly.
Do not translate terms from the glossary.

EXAMPLE INPUT:
[{"key":"msg1", "text":"Hello world"}, {"key":"msg2", "text":"Translate this!"}]

EXAMPLE OUTPUT:
{"items": [{"key": "msg1", "text": "Hallo Welt"}, {"key": "msg2", "text": "Übersetze das!"}]}`;

export default function SettingsPanel() {
  const [s, setS] = useState<Settings>({
    provider: "openai_compat",
    system_prompt: DEFAULT_SYSTEM_PROMPT,
    glossary: "",
    batch_size: 50,
    openai_compat: {
      base_url: "https://api.openai.com/v1",
      api_key: "",
      model: "gpt-4o-mini",
      temperature: 0.2,
    },
    ollama: { host: "http://localhost:11434", model: "llama3.1", temperature: 0.2 },
  });
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [verifyMsg, setVerifyMsg] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);

  // Determine API base URL from environment or default
  const API_BASE =
    (import.meta as any).env?.VITE_BACKEND_URL?.replace(/\/$/, "") ||
    "http://localhost:8000";

  // Load current settings from backend on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/settings`);
        const data = await res.json();
        if (data?.defaults) {
          // Ensure system_prompt is not empty; use default if missing
          if (!data.defaults.system_prompt) {
            data.defaults.system_prompt = DEFAULT_SYSTEM_PROMPT;
          }
          setS(data.defaults);
        }
      } catch {
        // ignore errors (e.g., if backend not available on load)
      }
    })();
  }, []);

  const save = async () => {
    setSaving(true);
    setMsg(null);
    setErr(null);
    try {
      const res = await fetch(`${API_BASE}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(s),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || "Failed to save settings");
      }
      await res.json();
      setMsg("Settings saved.");
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setSaving(false);
    }
  };

  const verify = async () => {
    setVerifyMsg("Verifying...");
    setErr(null);
    setModels([]);
    try {
      const res = await fetch(`${API_BASE}/providers/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(s),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Verification failed");
      }
      if (s.provider === "ollama") {
        const ms = data?.details?.models || [];
        setModels(ms);
        if (ms.length && !ms.includes(s.ollama.model)) {
          // If current model not in list, default to first available model
          setS((prev) => ({
            ...prev,
            ollama: { ...prev.ollama, model: ms[0] },
          }));
        }
        setVerifyMsg(`Ollama OK. ${ms.length} model(s) found.`);
      } else {
        setVerifyMsg(`OpenAI-compatible OK. Using model "${s.openai_compat.model}".`);
      }
    } catch (e: any) {
      setVerifyMsg(null);
      setErr(e.message);
    }
  };

  // If provider switched to Ollama, pre-fetch model list (optional improvement)
  useEffect(() => {
    if (s.provider !== "ollama") return;
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/providers/ollama/models`);
        const j = await r.json();
        if (j?.ok) {
          setModels(j.models || []);
        }
      } catch {
        /* ignore errors */
      }
    })();
  }, [s.provider]);

  return (
    <div className="space-y-6">
      <div className="rounded-xl border bg-white p-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <Label>Provider</Label>
            <Select
              value={s.provider}
              onChange={(v) => setS({ ...s, provider: v as Settings["provider"] })}
              options={[
                { value: "openai_compat", label: "OpenAI-compatible" },
                { value: "ollama", label: "Ollama (local)" },
              ]}
            />
          </div>
          <div>
            <Label>Batch size</Label>
            <Input
              type="number"
              min={10}
              max={300}
              value={s.batch_size}
              onChange={(e) => setS({ ...s, batch_size: Number(e.target.value) })}
            />
          </div>
          <div className="sm:col-span-2">
            <Label>System prompt</Label>
            <Textarea
              value={s.system_prompt}
              onChange={(e) => setS({ ...s, system_prompt: e.target.value })}
              rows={12}
              placeholder="Guidance for the model"
            />
          </div>
          <div className="sm:col-span-2">
            <Label>Glossary / Untranslatable terms</Label>
            <Textarea
              value={s.glossary}
              onChange={(e) => setS({ ...s, glossary: e.target.value })}
              rows={4}
              placeholder="One term per line"
            />
          </div>
        </div>
      </div>

      {s.provider === "openai_compat" && (
        <div className="rounded-xl border bg-white p-4">
          <h3 className="font-semibold mb-2">OpenAI-compatible</h3>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label>Base URL</Label>
              <Input
                value={s.openai_compat.base_url}
                onChange={(e) =>
                  setS({ ...s, openai_compat: { ...s.openai_compat, base_url: e.target.value } })
                }
              />
            </div>
            <div>
              <Label>API Key</Label>
              <Input
                type="password"
                value={s.openai_compat.api_key}
                onChange={(e) =>
                  setS({ ...s, openai_compat: { ...s.openai_compat, api_key: e.target.value } })
                }
                placeholder="sk-... (leave blank if not needed)"
              />
            </div>
            <div>
              <Label>Model</Label>
              <Input
                value={s.openai_compat.model}
                onChange={(e) =>
                  setS({ ...s, openai_compat: { ...s.openai_compat, model: e.target.value } })
                }
              />
            </div>
            <div>
              <Label>Temperature</Label>
              <Input
                type="number"
                step={0.1}
                min={0}
                max={2}
                value={s.openai_compat.temperature}
                onChange={(e) =>
                  setS({
                    ...s,
                    openai_compat: { ...s.openai_compat, temperature: Number(e.target.value) },
                  })
                }
              />
            </div>
          </div>
        </div>
      )}

      {s.provider === "ollama" && (
        <div className="rounded-xl border bg-white p-4">
          <h3 className="font-semibold mb-2">Ollama (Local AI)</h3>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label>Host URL</Label>
              <Input
                value={s.ollama.host}
                onChange={(e) =>
                  setS({ ...s, ollama: { ...s.ollama, host: e.target.value } })
                }
              />
            </div>
            <div>
              <Label>Model</Label>
              <Input
                value={s.ollama.model}
                onChange={(e) =>
                  setS({ ...s, ollama: { ...s.ollama, model: e.target.value } })
                }
              />
            </div>
            <div>
              <Label>Temperature</Label>
              <Input
                type="number"
                step={0.1}
                min={0}
                max={2}
                value={s.ollama.temperature}
                onChange={(e) =>
                  setS({
                    ...s,
                    ollama: { ...s.ollama, temperature: Number(e.target.value) },
                  })
                }
              />
            </div>
            {/* The models dropdown is populated after verify or provider switch */}
            {models.length > 0 && (
              <div className="sm:col-span-2">
                <Label>Available Models</Label>
                <Select
                  value={s.ollama.model}
                  onChange={(v) => setS({ ...s, ollama: { ...s.ollama, model: v as string } })}
                  options={models.map((m) => ({ value: m, label: m }))}
                />
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex items-center gap-3">
        <Button onClick={save} disabled={saving}>
          {saving ? "Saving..." : "Save Settings"}
        </Button>
        <Button variant="secondary" onClick={verify}>
          Verify Provider
        </Button>
        {verifyMsg && <span className="text-sm text-gray-600">{verifyMsg}</span>}
      </div>

      {msg && (
        <div className="text-green-700 bg-green-50 border border-green-300 rounded-lg p-3 text-sm">
          {msg}
        </div>
      )}
      {err && (
        <div className="text-red-700 bg-red-50 border border-red-300 rounded-lg p-3 text-sm">
          {err}
        </div>
      )}
    </div>
  );
}
