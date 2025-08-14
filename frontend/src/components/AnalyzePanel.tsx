import React, { useMemo, useState } from "react";

type AnalyzeResponse = {
  ok: boolean;
  info: {
    headers: Record<string, string>;
    counts: { entries: number; plurals: number; untranslated: number };
    placeholderSamples: string[];
  };
  audit?: {
    summary: {
      counts: Record<string, number>;
      language_detected: string;
      spellcheck_enabled: boolean;
    };
    issues: Array<{
      type: string;
      index: number;
      msgid: string;
      detail: string;
    }>;
  };
};

const API_BASE =
  (import.meta as any).env?.VITE_BACKEND_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

const ChevronDown = () => <span className="inline-block select-none">▼</span>;
const ChevronRight = () => <span className="inline-block select-none">▶</span>;

export default function AnalyzePanel() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  const groupedIssues = useMemo(() => {
    const g: Record<string, AnalyzeResponse["audit"]["issues"]> = {};
    if (!result?.audit?.issues) return g;
    for (const it of result.audit.issues) {
      g[it.type] = g[it.type] || [];
      g[it.type].push(it);
    }
    return g;
  }, [result]);

  async function onAnalyze() {
    if (!file) {
      setError("Choose a .po file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("po", file);
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `HTTP ${res.status}`);
      }
      const data: AnalyzeResponse = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  function toggle(group: string) {
    setOpenGroups((s) => ({ ...s, [group]: !s[group] }));
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border p-4 bg-white">
        <div className="grid gap-3 sm:grid-cols-[1fr_auto] items-end">
          <div>
            <label className="block text-sm font-medium mb-1">PO file</label>
            <input
              type="file"
              accept=".po"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full border rounded px-3 py-2"
            />
          </div>
          <div className="sm:pl-4">
            <button
              onClick={onAnalyze}
              disabled={loading || !file}
              className="inline-flex items-center rounded bg-black text-white px-4 py-2 disabled:opacity-50"
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-xl border border-red-300 bg-red-50 p-3 text-red-700 text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Analysis Summary */}
      <div>
        <h2 className="text-xl font-bold mb-2">Analysis Summary</h2>
        {!result && (
          <div className="h-6 rounded bg-gray-100" aria-hidden="true" />
        )}

        {result && (
          <div className="rounded-xl border bg-white p-4 space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg bg-gray-50 p-3">
                <div className="text-xs text-gray-600">Entries</div>
                <div className="text-lg font-semibold">
                  {result.info.counts.entries}
                </div>
              </div>
              <div className="rounded-lg bg-gray-50 p-3">
                <div className="text-xs text-gray-600">Plural strings</div>
                <div className="text-lg font-semibold">
                  {result.info.counts.plurals}
                </div>
              </div>
              <div className="rounded-lg bg-gray-50 p-3">
                <div className="text-xs text-gray-600">Untranslated</div>
                <div className="text-lg font-semibold">
                  {result.info.counts.untranslated}
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-1">Headers</h4>
              <pre className="overflow-auto rounded bg-gray-900 text-gray-100 text-xs p-3">
                {JSON.stringify(result.info.headers, null, 2)}
              </pre>
            </div>

            {result.info.placeholderSamples?.length ? (
              <div>
                <h4 className="font-medium mb-1">Placeholder samples</h4>
                <div className="flex flex-wrap gap-2 text-xs">
                  {result.info.placeholderSamples.map((t, i) => (
                    <span
                      key={i}
                      className="rounded bg-gray-100 px-2 py-1 font-mono"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>

      {/* Audit */}
      {result?.audit && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xl font-bold">PO File Audit</h2>
            <span
              className={`px-2 py-1 rounded text-xs font-semibold ${
                result.audit.summary.spellcheck_enabled
                  ? "bg-green-100 text-green-800"
                  : "bg-gray-200 text-gray-800"
              }`}
              title="Whether pyspellchecker is available on the backend"
            >
              {result.audit.summary.spellcheck_enabled
                ? "Spellcheck enabled"
                : "Spellcheck off"}
            </span>
          </div>

          <div className="rounded-xl border bg-white p-4 space-y-4">
            <div className="overflow-auto">
              <table className="min-w-[480px] border">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-600">
                      Metric
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-600">
                      Count
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.audit.summary.counts).map(
                    ([k, v]) => (
                      <tr key={k} className="border-t">
                        <td className="px-3 py-2 text-sm">{k}</td>
                        <td className="px-3 py-2 text-sm">{v}</td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>

            {/* Issue groups */}
            {Object.keys(groupedIssues).length === 0 ? (
              <div className="text-sm text-gray-600">
                No issues found by the audit.
              </div>
            ) : (
              <div className="space-y-2">
                {Object.entries(groupedIssues).map(([type, issues]) => {
                  const open = !!openGroups[type];
                  return (
                    <div key={type} className="border rounded">
                      <button
                        onClick={() => toggle(type)}
                        className="w-full flex items-center justify-between px-3 py-2 bg-gray-100 hover:bg-gray-200"
                        type="button"
                      >
                        <span className="font-medium">
                          {type} ({issues.length})
                        </span>
                        {open ? <ChevronDown /> : <ChevronRight />}
                      </button>
                      {open && (
                        <div className="p-3 space-y-2 bg-white">
                          {issues.map((it, i) => (
                            <div
                              key={i}
                              className="text-sm border rounded p-2 bg-gray-50"
                            >
                              <div className="text-gray-800">
                                {it.detail}
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                index: {it.index}
                                {it.msgid ? (
                                  <>
                                    {" "}
                                    • msgid:{" "}
                                    <span className="font-mono">
                                      {it.msgid.slice(0, 100)}
                                    </span>
                                  </>
                                ) : null}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      <p className="text-sm text-gray-600">
        Built for WordPress plugin PO and MO workflows. Preserve placeholders
        and plural forms.
      </p>
    </div>
  );
}
