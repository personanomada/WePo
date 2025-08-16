import React, { useEffect, useMemo, useState } from "react"

type HeaderKV = { key: string; value: string }

type OpenAICompatSettings = {
  base_url: string
  api_key: string
  model: string
  temperature: number
  headers?: HeaderKV[]
  // new
  use_response_format?: boolean
  trace?: boolean
  trace_truncate?: number
}

type OllamaSettings = {
  host?: string
  base_url?: string
  model: string
  temperature: number
  headers?: HeaderKV[]
  // new
  trace?: boolean
  trace_truncate?: number
}

type UISettings = {
  // tool-level diagnostics settings
  show_test_run: boolean
  default_sample_size: number
}

type Settings = {
  provider: "openai_compat" | "ollama" | "openai" | "openrouter"
  system_prompt: string
  glossary: string
  batch_size: number
  openai_compat: OpenAICompatSettings
  ollama: OllamaSettings
  ui: UISettings
}

const blankHeaders: HeaderKV[] = []

function redactApiKey(v: string | undefined) {
  if (!v) return ""
  return v === "******" ? "" : v
}

export default function SettingsPanel() {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [defaults, setDefaults] = useState<Settings | null>(null)
  const [working, setWorking] = useState<Settings | null>(null)
  const [error, setError] = useState<string | null>(null)

  // single source of truth for "dirty"
  const dirty = useMemo(
    () => JSON.stringify(defaults) !== JSON.stringify(working),
    [defaults, working]
  )

  const fetchSettings = async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch("/settings", { method: "GET" })
      const text = await r.text()
      if (!r.ok) {
        throw new Error(`GET /settings ${r.status} ${r.statusText}: ${text.slice(0, 300)}`)
      }
      let j: any
      try {
        j = JSON.parse(text)
      } catch (e) {
        throw new Error(`Invalid JSON from /settings: ${String(e)}\nBody: ${text.slice(0, 300)}`)
      }
      if (!j || !j.defaults) {
        throw new Error("Missing 'defaults' in /settings response")
      }
      const d = j.defaults as Settings

      // normalize shape the UI expects
      d.openai_compat = d.openai_compat || ({} as any)
      d.ollama = d.ollama || ({} as any)
      d.ui = d.ui || { show_test_run: false, default_sample_size: 25 }

      d.openai_compat.api_key = redactApiKey(d.openai_compat.api_key)
      d.openai_compat.headers = d.openai_compat.headers || blankHeaders
      d.ollama.headers = d.ollama.headers || blankHeaders

      setDefaults(d)
      setWorking(JSON.parse(JSON.stringify(d)))
    } catch (e: any) {
      console.error(e)
      setError(String(e))
      // leave working as-is (likely null); render will show error instead of looping “Loading…”
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSettings()
  }, [])

  const onSave = async () => {
    if (!working) return
    setSaving(true)
    setError(null)
    try {
      const payload = JSON.parse(JSON.stringify(working))
      const r = await fetch("/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const text = await r.text()
      if (!r.ok) {
        throw new Error(`POST /settings ${r.status} ${r.statusText}: ${text.slice(0, 300)}`)
      }
      let j: any
      try {
        j = JSON.parse(text)
      } catch (e) {
        throw new Error(`Invalid JSON from POST /settings: ${String(e)}\nBody: ${text.slice(0, 300)}`)
      }
      if (!j || !j.defaults) {
        throw new Error("Missing 'defaults' in POST /settings response")
      }
      const d = j.defaults as Settings
      d.openai_compat.api_key = redactApiKey(d.openai_compat.api_key)
      d.openai_compat.headers = d.openai_compat.headers || blankHeaders
      d.ollama.headers = d.ollama.headers || blankHeaders
      d.ui = d.ui || { show_test_run: false, default_sample_size: 25 }

      setDefaults(d)
      setWorking(JSON.parse(JSON.stringify(d)))
    } catch (e: any) {
      console.error(e)
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const update = (patch: Partial<Settings>) => {
    setWorking((w) => (w ? { ...w, ...patch } : w))
  }

  const updateOpenAI = (patch: Partial<OpenAICompatSettings>) => {
    setWorking((w) => (w ? { ...w, openai_compat: { ...w.openai_compat, ...patch } } : w))
  }

  const updateOllama = (patch: Partial<OllamaSettings>) => {
    setWorking((w) => (w ? { ...w, ollama: { ...w.ollama, ...patch } } : w))
  }

  const updateUI = (patch: Partial<UISettings>) => {
    setWorking((w) => (w ? { ...w, ui: { ...w.ui, ...patch } } : w))
  }

  // --- Render states ---
  if (loading) return <div className="p-4">Loading…</div>
  if (error) {
    return (
      <div className="p-4 space-y-3">
        <div className="text-red-600 whitespace-pre-wrap">{error}</div>
        <button className="px-3 py-1 rounded bg-blue-600 text-white" onClick={fetchSettings}>
          Retry
        </button>
      </div>
    )
  }
  if (!working) {
    return (
      <div className="p-4 space-y-3">
        <div>No settings received from the server.</div>
        <button className="px-3 py-1 rounded bg-blue-600 text-white" onClick={fetchSettings}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-8">
      <h2 className="text-xl font-semibold">Settings</h2>

      <section className="space-y-2">
        <label className="block font-medium">Active provider</label>
        <select
          value={working.provider}
          onChange={(e) => update({ provider: e.target.value as Settings["provider"] })}
          className="border rounded px-2 py-1"
        >
          <option value="openai_compat">OpenAI Compatible</option>
          <option value="ollama">Ollama</option>
        </select>
      </section>

      <section className="space-y-2">
        <label className="block font-medium">System prompt</label>
        <textarea
          className="border rounded w-full p-2 h-28"
          value={working.system_prompt}
          onChange={(e) => update({ system_prompt: e.target.value })}
          placeholder="High-level translation instructions"
        />
      </section>

      <section className="space-y-2">
        <label className="block font-medium">Glossary (comma or newline separated)</label>
        <textarea
          className="border rounded w-full p-2 h-24"
          value={working.glossary}
          onChange={(e) => update({ glossary: e.target.value })}
        />
      </section>

      <section className="space-y-2">
        <label className="block font-medium">Batch size</label>
        <input
          type="number"
          className="border rounded px-2 py-1 w-32"
          min={1}
          max={200}
          value={working.batch_size}
          onChange={(e) => update({ batch_size: Number(e.target.value) || 1 })}
        />
      </section>

      <hr />

      {/* OpenAI Compatible */}
      <section className="space-y-3">
        <h3 className="text-lg font-semibold">OpenAI Compatible</h3>
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label className="block text-sm font-medium">Base URL</label>
            <input
              className="border rounded w-full px-2 py-1"
              value={working.openai_compat.base_url}
              onChange={(e) => updateOpenAI({ base_url: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium">API Key</label>
            <input
              className="border rounded w-full px-2 py-1"
              value={working.openai_compat.api_key}
              onChange={(e) => updateOpenAI({ api_key: e.target.value })}
              placeholder="sk-…"
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Model</label>
            <input
              className="border rounded w-full px-2 py-1"
              value={working.openai_compat.model}
              onChange={(e) => updateOpenAI({ model: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Temperature</label>
            <input
              type="number"
              step="0.1"
              className="border rounded w-full px-2 py-1"
              value={Number(working.openai_compat.temperature)}
              onChange={(e) => updateOpenAI({ temperature: Number(e.target.value) })}
            />
          </div>
        </div>

        {/* response_format + tracing */}
        <div className="grid gap-3 md:grid-cols-3">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={!!working.openai_compat.use_response_format}
              onChange={(e) => updateOpenAI({ use_response_format: e.target.checked })}
            />
            <span>Use response_format</span>
          </label>

          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={!!working.openai_compat.trace}
              onChange={(e) => updateOpenAI({ trace: e.target.checked })}
            />
            <span>Trace requests/responses</span>
          </label>

          <div className="flex items-center gap-2">
            <span className="text-sm">Trace truncate</span>
            <input
              type="number"
              className="border rounded px-2 py-1 w-28"
              value={Number(working.openai_compat.trace_truncate || 2000)}
              onChange={(e) => updateOpenAI({ trace_truncate: Number(e.target.value) || 2000 })}
            />
          </div>
        </div>
      </section>

      <hr />

      {/* Ollama */}
      <section className="space-y-3">
        <h3 className="text-lg font-semibold">Ollama</h3>
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <label className="block text-sm font-medium">Host / Base URL</label>
            <input
              className="border rounded w-full px-2 py-1"
              value={working.ollama.base_url || working.ollama.host || ""}
              onChange={(e) => updateOllama({ base_url: e.target.value, host: e.target.value })}
              placeholder="http://localhost:11434"
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Model</label>
            <input
              className="border rounded w-full px-2 py-1"
              value={working.ollama.model}
              onChange={(e) => updateOllama({ model: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Temperature</label>
            <input
              type="number"
              step="0.1"
              className="border rounded w-full px-2 py-1"
              value={Number(working.ollama.temperature)}
              onChange={(e) => updateOllama({ temperature: Number(e.target.value) })}
            />
          </div>
        </div>

        {/* tracing */}
        <div className="grid gap-3 md:grid-cols-3">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={!!working.ollama.trace}
              onChange={(e) => updateOllama({ trace: e.target.checked })}
            />
            <span>Trace requests/responses</span>
          </label>

          <div className="flex items-center gap-2">
            <span className="text-sm">Trace truncate</span>
            <input
              type="number"
              className="border rounded px-2 py-1 w-28"
              value={Number(working.ollama.trace_truncate || 2000)}
              onChange={(e) => updateOllama({ trace_truncate: Number(e.target.value) || 2000 })}
            />
          </div>
        </div>
      </section>

      <hr />

      {/* Diagnostics & UI */}
      <section className="space-y-3">
        <h3 className="text-lg font-semibold">Diagnostics & UI</h3>
        <label className="inline-flex items-center gap-2">
          <input
            type="checkbox"
            checked={!!working.ui.show_test_run}
            onChange={(e) => updateUI({ show_test_run: e.target.checked })}
          />
          <span>Show “Test run” control on Translate tab</span>
        </label>

        <div className="flex items-center gap-2">
          <span className="text-sm">Default test sample size</span>
          <input
            type="number"
            className="border rounded px-2 py-1 w-28"
            min={1}
            max={200}
            value={Number(working.ui.default_sample_size || 25)}
            onChange={(e) => updateUI({ default_sample_size: Number(e.target.value) || 25 })}
          />
        </div>
      </section>

      <div className="pt-4">
        <button
          className={`px-4 py-2 rounded ${dirty ? "bg-blue-600 text-white" : "bg-gray-300 text-gray-700"}`}
          onClick={onSave}
          disabled={!dirty || saving}
        >
          {saving ? "Saving…" : "Save settings"}
        </button>
        {!dirty && <span className="ml-3 text-sm text-gray-500">No changes</span>}
      </div>
    </div>
  )
}
