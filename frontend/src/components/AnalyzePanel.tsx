import { useMemo, useState } from "react";
import { Button } from "../components/ui/Button";
import { cn } from "../lib/cn";

type LintItem = {
  index: number;
  key: string;
  ctx?: string | null;
  rule: string;
  severity: "error" | "warn" | "info";
  message: string;
  suggestion?: string | null;
  fix?: [ "replace_msgstr" | "replace_plural_i", string ];
  sample?: string | null;
  source?: string | null;
  translation?: string | null;
};

type Report = {
  detected_locale: string;
  nplurals: number;
  summary: {
    total: number;
    by_severity: { error: number; warn: number; info: number };
    by_rule: Record<string, number>;
  };
  items: LintItem[];
};

function filenameTs() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

export default function AnalyzePanel() {
  const API_BASE = (import.meta as any).env?.VITE_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

  const [poFile, setPoFile] = useState<File | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [sevFilter, setSevFilter] = useState<"all" | "error" | "warn" | "info">("all");
  const [ruleFilter, setRuleFilter] = useState<string>("all");

  const [pendingFixes, setPendingFixes] = useState<{ index: number; key: string; kind: string; text: string }[]>([]);

  const visibleItems = useMemo(() => {
    if (!report) return [];
    return report.items.filter(it => {
      if (sevFilter !== "all" && it.severity !== sevFilter) return false;
      if (ruleFilter !== "all" && it.rule !== ruleFilter) return false;
      return true;
    });
  }, [report, sevFilter, ruleFilter]);

  const runAnalyze = async () => {
    if (!poFile) { setError("Choose a PO file"); return; }
    setError(null); setLoading(true); setReport(null); setPendingFixes([]);
    const form = new FormData();
    form.append("po", poFile);
    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
    if (!res.ok) { setError(`HTTP ${res.status}: ${await res.text()}`); setLoading(false); return; }
    const data = await res.json();
    setReport(data); setLoading(false);
  };

  const addFix = (it: LintItem) => {
    if (!it.fix) return;
    const [kind, text] = it.fix;
    setPendingFixes(prev => {
      if (prev.find(p => p.index === it.index && p.key === it.key && p.kind === kind)) return prev;
      return [...prev, { index: it.index, key: it.key, kind, text }];
    });
  };

  const applyAllVisibleFixes = () => { visibleItems.forEach(it => addFix(it)); };

  const exportJSON = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `wepo-analysis_${filenameTs()}.json`;
    a.click(); URL.revokeObjectURL(url);
  };

  const exportCSV = () => {
    if (!report) return;
    const header = ["index","key","ctx","rule","severity","message","suggestion","source","translation"].join(",");
    const rows = report.items.map(it => [
      it.index, it.key, JSON.stringify(it.ctx || ""), it.rule, it.severity,
      JSON.stringify(it.message), JSON.stringify(it.suggestion || ""),
      JSON.stringify(it.source || ""), JSON.stringify(it.translation || "")
    ].join(","));
    const blob = new Blob([header+"\n"+rows.join("\n")],{type:"text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = url; a.download = `wepo-analysis_${filenameTs()}.csv`; a.click(); URL.revokeObjectURL(url);
  };

  const downloadFixedPo = async () => {
    if (!poFile) return;
    const form = new FormData();
    form.append("po", poFile);
    form.append("fixes_json", JSON.stringify(pendingFixes));
    const res = await fetch(`${API_BASE}/analyze/apply`, { method: "POST", body: form });
    if (!res.ok) { setError(`HTTP ${res.status}: ${await res.text()}`); return; }
    const data = await res.json();
    const byteString = atob(data.poBase64);
    const bytes = new Uint8Array(byteString.length);
    for (let i = 0; i < byteString.length; i++) bytes[i] = byteString.charCodeAt(i);
    const blob = new Blob([bytes], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "analyzed-fixed.po"; a.click(); URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <label className="text-sm font-medium">PO file</label>
        <input type="file" accept=".po" onChange={e => setPoFile(e.target.files?.[0] || null)} />
      </div>

      <div className="flex items-center gap-3">
        <Button onClick={runAnalyze} disabled={loading || !poFile}>{loading ? "Analyzing…" : "Analyze"}</Button>
        {report && (
          <>
            <Button onClick={exportJSON} className="bg-gray-100 text-gray-900">Export JSON</Button>
            <Button onClick={exportCSV} className="bg-gray-100 text-gray-900">Export CSV</Button>
          </>
        )}
        <div className="ml-auto text-xs text-gray-600">
          {report ? <>Detected locale: <b>{report.detected_locale}</b> • nplurals: <b>{report.nplurals}</b></> : null}
        </div>
      </div>

      {error && <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-red-700 text-sm">{error}</div>}

      {report && (
        <>
          <div className="rounded-lg border p-3 text-sm bg-white">
            <div className="flex gap-6">
              <div>Issues: <b>{report.summary.total}</b></div>
              <div className="text-red-700">Errors: <b>{report.summary.by_severity.error}</b></div>
              <div className="text-amber-700">Warnings: <b>{report.summary.by_severity.warn}</b></div>
              <div className="text-gray-600">Info: <b>{report.summary.by_severity.info}</b></div>
              <div className="ml-auto flex items-center gap-2">
                <label>Severity</label>
                <select className="border rounded px-2 py-1" value={sevFilter} onChange={e => setSevFilter(e.target.value as any)}>
                  <option value="all">All</option><option value="error">Error</option><option value="warn">Warn</option><option value="info">Info</option>
                </select>
                <label>Rule</label>
                <select className="border rounded px-2 py-1" value={ruleFilter} onChange={e => setRuleFilter(e.target.value)}>
                  <option value="all">All</option>
                  {Object.keys(report.summary.by_rule).map(r => <option key={r} value={r}>{r}</option>)}
                </select>
                <Button onClick={applyAllVisibleFixes} className="bg-gray-200 text-gray-900">Queue fixes for visible</Button>
                <Button onClick={downloadFixedPo} className="ml-2">Download with queued fixes ({pendingFixes.length})</Button>
              </div>
            </div>
          </div>

          <div className="rounded-lg border bg-white">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="p-2 text-left w-12">#</th>
                  <th className="p-2 text-left w-28">Key</th>
                  <th className="p-2 text-left w-20">Severity</th>
                  <th className="p-2 text-left w-32">Rule</th>
                  <th className="p-2 text-left w-80">Source</th>
                  <th className="p-2 text-left w-80">Translation</th>
                  <th className="p-2 text-left">Message</th>
                  <th className="p-2 text-left w-56">Suggestion</th>
                  <th className="p-2 text-left w-32">Auto-fix</th>
                </tr>
              </thead>
              <tbody>
                {visibleItems.map(it => (
                  <tr key={`${it.key}:${it.rule}`} className="border-t align-top">
                    <td className="p-2">{it.index}</td>
                    <td className="p-2 font-mono">{it.key}</td>
                    <td className={cn("p-2", it.severity === "error" ? "text-red-700" : it.severity === "warn" ? "text-amber-700" : "text-gray-600")}>{it.severity}</td>
                    <td className="p-2">{it.rule}</td>
                    <td className="p-2 whitespace-pre-wrap">{it.source || ""}</td>
                    <td className="p-2 whitespace-pre-wrap">{it.translation || ""}</td>
                    <td className="p-2">{it.message}</td>
                    <td className="p-2">{it.suggestion || "-"}</td>
                    <td className="p-2">
                      <Button
                        className={cn("px-2 py-1 text-xs", !it.fix && "opacity-40 pointer-events-none")}
                        onClick={() => it.fix && addFix(it)}
                      >
                        {it.fix ? "Apply auto-fix" : "No auto-fix"}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
