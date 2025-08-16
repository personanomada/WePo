import { useEffect, useRef, useState, useMemo } from "react";
import { LOCALES, LocaleOption } from "../lib/locales";
import { Button } from "../components/ui/Button";
import { cn } from "../lib/cn";

type Progress = { done: number; total: number; echoed: number };

type LogEntry = { ts: string; level: "info" | "warn" | "error"; text: string };

type ReviewItem = {
  key: string;
  source: string;
  candidate?: string | null; // may be same as source when reason === "echo"
  reason: "echo" | "missing";
};

type ReviewJob = {
  srcPoB64: string;
  locales: string[];
  translated: Record<string, Record<string, string>>; // locale -> {key: text}
  review: Record<string, ReviewItem[]>;               // locale -> items
};

type Decision = { key: string; action: "accept" | "reject" | "edit"; text?: string };

function nowTs() {
  const d = new Date();
  return d.toLocaleTimeString();
}
function filenameTs() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

async function streamAndParseAIResponse(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onMeta: (meta: any) => void,
  onProgress: (p: Progress, raw?: any) => void,
  onBatchDetail: (detail: any) => void,
  onDone: (done: any) => void,
  onError: (error: any) => void,
  signal: AbortSignal,
  onAny?: (raw: any) => void,
  onReview?: (reviewEvt: any) => void
) {
  const dec = new TextDecoder();
  let buffer = "";

  const read = async (): Promise<void> => {
    if (signal.aborted) {
      try { await reader.cancel(); } catch {}
      return;
    }
    const { value, done } = await reader.read();
    if (done) return;

    buffer += dec.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let eventObj: any = null;
      try {
        eventObj = JSON.parse(line);
      } catch (e) {
        onError({ message: `NDJSON parse error: ${(e as Error).message}`, raw: line });
        continue;
      }
      if (onAny) onAny(eventObj);

      switch (eventObj.type) {
        case "meta": onMeta(eventObj); break;
        case "progress":
          onProgress(
            {
              done: Number(eventObj.done || 0),
              total: Number(eventObj.total || 0),
              echoed: Number(eventObj.echoed || 0),
            },
            eventObj
          );
          break;
        case "batch":
        case "retry":
        case "fallback":
        case "heartbeat":
          onBatchDetail(eventObj);
          break;
        case "done": onDone(eventObj); break;
        case "error": onError(eventObj); break;
        case "review": onReview && onReview(eventObj); break;
        default: onBatchDetail(eventObj);
      }
    }

    await read();
  };

  await read();
}

export default function TranslatePanel() {
  const API_BASE =
    (import.meta as any).env?.VITE_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState("en");
  const [selectedLocales, setSelectedLocales] = useState<LocaleOption[]>([]);
  const [progress, setProgress] = useState<Progress>({ done: 0, total: 0, echoed: 0 });
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [consoleOpen, setConsoleOpen] = useState<boolean>(true);

  const [reviewJob, setReviewJob] = useState<ReviewJob | null>(null);
  const [decisions, setDecisions] = useState<Record<string, Record<string, Decision>>>({}); // locale -> key -> decision

  const controllerRef = useRef<AbortController | null>(null);
  const consoleEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (consoleOpen && consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, consoleOpen]);

  const log = (level: LogEntry["level"], text: string) => {
    setLogs((prev) => [...prev, { ts: nowTs(), level, text }]);
  };

  useEffect(() => { setError(null); }, [file, sourceLang, selectedLocales]);

  const onLocaleToggle = (opt: LocaleOption) => {
    setSelectedLocales((prev) => {
      const exists = prev.find((p) => p.code === opt.code);
      if (exists) return prev.filter((p) => p.code !== opt.code);
      return [...prev, opt];
    });
  };

  const b64ToBlob = (b64: string, contentType: string) => {
    const byteCharacters = atob(b64);
    const byteNumbers = new Array(byteCharacters.length)
      .fill(0)
      .map((_, i) => byteCharacters.charCodeAt(i));
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: contentType });
  };

  const cancelTranslation = () => {
    if (controllerRef.current) controllerRef.current.abort();
  };

  const clearConsole = () => setLogs([]);
  const copyConsole = async () => {
    const text = logs.map((l) => `[${l.ts}] ${l.level.toUpperCase()} ${l.text}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      log("info", "Console copied to clipboard.");
    } catch {
      log("warn", "Failed to copy console.");
    }
  };
  const exportConsole = () => {
    const text = logs.map((l) => `[${l.ts}] ${l.level.toUpperCase()} ${l.text}`).join("\n");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `wepo-translate-console_${filenameTs()}.txt`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
    log("info", "Console exported.");
  };

  const setDecision = (locale: string, key: string, decision: Decision) => {
    setDecisions((prev) => {
      const byLocale = { ...(prev[locale] || {}) };
      byLocale[key] = decision;
      return { ...prev, [locale]: byLocale };
    });
  };

  const bulkDecide = (locale: string, action: Decision["action"]) => {
    if (!reviewJob) return;
    const items = reviewJob.review[locale] || [];
    const byLocale: Record<string, Decision> = {};
    for (const it of items) {
      byLocale[it.key] = action === "edit"
        ? { key: it.key, action: "edit", text: it.source }
        : { key: it.key, action };
    }
    setDecisions((prev) => ({ ...prev, [locale]: byLocale }));
  };

  const reviewedCount = useMemo(() => {
    if (!reviewJob) return 0;
    let n = 0;
    for (const loc of reviewJob.locales) {
      const total = (reviewJob.review[loc] || []).length;
      const picked = Object.keys(decisions[loc] || {}).length;
      n += Math.min(total, picked);
    }
    return n;
  }, [reviewJob, decisions]);

  const reviewTotal = useMemo(() => {
    if (!reviewJob) return 0;
    return reviewJob.locales.reduce((sum, loc) => sum + (reviewJob.review[loc] || []).length, 0);
  }, [reviewJob]);

  const handleTranslateNDJSON = async () => {
    if (!file) { setError("Please choose a PO file first."); return; }
    if (selectedLocales.length === 0) { setError("Choose at least one target locale."); return; }

    setIsTranslating(true);
    setError(null);
    setStatus("Initializing translation...");
    setProgress({ done: 0, total: 0, echoed: 0 });
    clearConsole();
    setReviewJob(null);
    setDecisions({});

    const form = new FormData();
    form.append("po", file);
    form.append("locales", selectedLocales.map((l) => l.code).join(","));
    form.append("sourceLang", sourceLang);

    const aborter = new AbortController();
    controllerRef.current = aborter;

    try {
      log("info", `POST ${API_BASE}/translate/ndjson`);
      const res = await fetch(`${API_BASE}/translate/ndjson`, { method: "POST", body: form, signal: aborter.signal });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }
      if (!res.body) throw new Error("No response body from translation API");
      const reader = res.body.getReader();

      await streamAndParseAIResponse(
        reader,
        (meta) => { setProgress((p) => ({ ...p, total: meta.total || 0 })); setStatus(`Found ${meta.total} strings. Translating...`); log("info", `meta total=${meta.total}`); },
        (prog, raw) => {
          setProgress((prev) => ({ ...prev, ...prog }));
          setStatus(`Translated ${prog.done} of ${prog.total} strings...`);
          const loc = raw?.locale ? ` locale=${raw.locale}` : "";
          log("info", `progress done=${prog.done}/${prog.total} echoed=${prog.echoed}${loc}`);
        },
        (detail) => {
          const t = detail?.type || "event";
          if (t === "batch") {
            log("info", `batch attempt=${detail.attempt} idx=${detail.batch_index} size=${detail.batch_size} returned=${detail.returned} accepted=${detail.accepted} echoed=${detail.echoed} missing=${detail.missing_count}` + (detail.missing_sample?.length ? ` sample=${detail.missing_sample.join(",")}` : "") + (detail.locale ? ` locale=${detail.locale}` : ""));
          } else if (t === "retry") {
            log("warn", `retry key=${detail.key} mode=${detail.mode} status=${detail.status}${detail.locale ? ` locale=${detail.locale}` : ""}`);
          } else if (t === "fallback") {
            log("warn", `fallback ${detail.mode}${detail.autofilled ? ` autofilled=${detail.autofilled}` : ""}${detail.locale ? ` locale=${detail.locale}` : ""}`);
          } else if (t === "heartbeat") {
            log("info", `heartbeat${detail.locale ? ` locale=${detail.locale}` : ""}`);
          } else if (t === "error") {
            log("error", `server error: ${detail.message || "(no message)"}${detail.missing_keys_sample ? ` sample=${detail.missing_keys_sample.join(",")}` : ""}${detail.locale ? ` locale=${detail.locale}` : ""}`);
          } else {
            log("info", `${t}: ${JSON.stringify(detail).slice(0, 200)}`);
          }
        },
        (doneEvent) => {
          setStatus("Translation complete! Preparing download...");
          log("info", "done, preparing ZIP.");
          const b64 = doneEvent.zipBase64 as string;
          const blob = b64ToBlob(b64, "application/zip");
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = "translations.zip";
          document.body.appendChild(a); a.click(); document.body.removeChild(a);
          URL.revokeObjectURL(url);
        },
        (errorEvent) => {
          const msg = errorEvent.message || "An unknown error occurred during translation.";
          setError(msg); log("error", msg);
        },
        aborter.signal,
        undefined,
        (evt) => {
          const job: ReviewJob = evt.job;
          setReviewJob(job);
          setStatus(`Review required: ${evt.total} unresolved item(s).`);
          log("warn", `review required: total=${evt.total}`);
        }
      );
    } catch (e: any) {
      if (e.name !== "AbortError") { const msg = e.message || String(e); setError(msg); log("error", msg); }
      else { setStatus("Translation cancelled."); log("warn", "Translation cancelled."); }
    } finally {
      setIsTranslating(false);
      controllerRef.current = null;
    }
  };

  const finalizeWithDecisions = async () => {
    if (!reviewJob) return;
    const payload = {
      job: reviewJob,
      decisions: Object.fromEntries(
        Object.entries(decisions).map(([loc, byKey]) => [loc, Object.values(byKey)])
      ),
    };

    try {
      log("info", `POST ${API_BASE}/translate/finalize`);
      const res = await fetch(`${API_BASE}/translate/finalize`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!res.ok) { const txt = await res.text(); throw new Error(`HTTP ${res.status}: ${txt}`); }
      const data = await res.json();
      const b64 = data.zipBase64 as string;
      const blob = b64ToBlob(b64, "application/zip");
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "translations.zip";
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
      log("info", "Final ZIP downloaded.");
      setReviewJob(null);
    } catch (e: any) {
      const msg = e.message || String(e); setError(msg); log("error", msg);
    }
  };

  return (
    <div className="space-y-6">
      {/* Inputs */}
      <div className="space-y-2">
        <label className="text-sm font-medium">PO file</label>
        <input type="file" accept=".po" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Source language</label>
          <input className="w-full rounded-md border px-3 py-2 text-sm" value={sourceLang} onChange={(e) => setSourceLang(e.target.value)} placeholder="en" />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Target locales</label>
          <div className="rounded-md border p-2">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-1 max-h-48 overflow-auto">
              {LOCALES.map((opt) => {
                const active = !!selectedLocales.find((s) => s.code === opt.code);
                return (
                  <button
                    type="button"
                    key={opt.code}
                    onClick={() => onLocaleToggle(opt)}
                    className={cn("text-left text-xs rounded px-2 py-1 border", active ? "bg-black text-white border-black" : "bg-white")}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
            {selectedLocales.length > 0 && (
              <div className="mt-2 text-xs text-gray-600">Selected: {selectedLocales.map((l) => l.code).join(", ")}</div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <Button disabled={isTranslating || !file || selectedLocales.length === 0} onClick={handleTranslateNDJSON}>
          {isTranslating ? "Translating…" : "Translate"}
        </Button>
        {isTranslating && (
          <Button onClick={cancelTranslation} className="bg-gray-200 text-gray-800 hover:bg-gray-300">Cancel</Button>
        )}
        <button type="button" className="text-sm underline ml-auto" onClick={() => setConsoleOpen((v) => !v)}>
          {consoleOpen ? "Hide console" : "Show console"}
        </button>
      </div>

      {/* Progress */}
      {isTranslating && (
        <div className="mt-4 space-y-2">
          <div className="h-2 bg-gray-200 rounded overflow-hidden">
            <div className="h-2 bg-black rounded transition-all duration-300 ease-in-out"
              style={{ width: progress.total > 0 ? `${Math.min(100, (progress.done / progress.total) * 100)}%` : "100%" }} />
          </div>
          <div className="flex justify-between text-xs text-gray-600">
            <span>{status || "Waiting..."}</span>
            <span>{progress.total > 1 ? `${progress.done} / ${progress.total}` : ""}{progress.echoed > 0 && ` (echoed: ${progress.echoed})`}</span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-red-700 text-sm">{error}</div>}

      {/* Console */}
      {consoleOpen && (
        <div className="rounded-xl border bg-white">
          <div className="flex items-center justify-between px-3 py-2 border-b">
            <div className="font-medium text-sm">Console</div>
            <div className="flex gap-2">
              <button className="text-xs underline" onClick={exportConsole}>Export</button>
              <button className="text-xs underline" onClick={copyConsole}>Copy</button>
              <button className="text-xs underline" onClick={() => setLogs([])}>Clear</button>
            </div>
          </div>
          <div className="max-h-64 overflow-auto px-3 py-2 text-xs font-mono">
            {logs.length === 0 && <div className="text-gray-400">No logs yet.</div>}
            {logs.map((l, i) => (
              <div key={i} className={cn("whitespace-pre-wrap", l.level === "error" ? "text-red-700" : l.level === "warn" ? "text-amber-700" : "text-gray-800")}>
                [{l.ts}] {l.level.toUpperCase()} {l.text}
              </div>
            ))}
            <div ref={consoleEndRef} />
          </div>
        </div>
      )}

      {/* Review panel */}
      {reviewJob && (
        <div className="rounded-xl border bg-white">
          <div className="px-4 py-3 border-b flex items-center gap-3">
            <div className="font-medium text-sm">Review unresolved strings</div>
            <div className="text-xs text-gray-500">
              Decide per string. Defaults are <b>Reject</b> to prevent copying English.
            </div>
            <div className="ml-auto text-xs">
              Reviewed <b>{reviewedCount}</b> / {reviewTotal}
            </div>
          </div>

          {reviewJob.locales.map((loc) => {
            const items = reviewJob.review[loc] || [];
            if (items.length === 0) return null;
            const byKey = decisions[loc] || {};
            return (
              <div key={loc} className="px-4 py-3 border-t">
                <div className="mb-2 flex items-center gap-3">
                  <div className="text-sm font-medium">Locale {loc}</div>
                  <div className="text-xs text-gray-500">{items.length} item(s)</div>
                  <div className="ml-auto flex gap-2">
                    <Button className="bg-gray-100 text-gray-900" onClick={() => bulkDecide(loc, "reject")}>Reject all</Button>
                    <Button className="bg-gray-100 text-gray-900" onClick={() => bulkDecide(loc, "accept")}>Accept all echoes</Button>
                    <Button className="bg-gray-100 text-gray-900" onClick={() => bulkDecide(loc, "edit")}>Mark all as edit</Button>
                  </div>
                </div>

                <div className="max-h-96 overflow-auto border rounded">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="text-left p-2 w-28">Key</th>
                        <th className="text-left p-2">Source</th>
                        <th className="text-left p-2">Candidate</th>
                        <th className="text-left p-2 w-24">Reason</th>
                        <th className="text-left p-2 w-36">Decision</th>
                        <th className="text-left p-2 w-64">Edited text</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((it) => {
                        const d = byKey[it.key] || { key: it.key, action: "reject" as const };
                        const isEdit = d.action === "edit";
                        return (
                          <tr key={it.key} className="border-t align-top">
                            <td className="p-2 font-mono">{it.key}</td>
                            <td className="p-2 whitespace-pre-wrap">{it.source}</td>
                            <td className="p-2 whitespace-pre-wrap text-gray-700">
                              {it.candidate ? it.candidate : <span className="text-gray-400">—</span>}
                            </td>
                            <td className="p-2">{it.reason}</td>
                            <td className="p-2">
                              <select
                                className="border rounded px-1 py-0.5 text-xs"
                                value={d.action}
                                onChange={(e) => setDecision(loc, it.key, { key: it.key, action: e.target.value as any })}
                              >
                                <option value="reject">Reject</option>
                                <option value="accept">Accept echo</option>
                                <option value="edit">Edit</option>
                              </select>
                            </td>
                            <td className="p-2">
                              {isEdit && (
                                <textarea
                                  className="w-full border rounded px-1 py-1 text-xs"
                                  rows={2}
                                  defaultValue={it.candidate || it.source}
                                  onChange={(e) => setDecision(loc, it.key, { key: it.key, action: "edit", text: e.target.value })}
                                />
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}

          <div className="px-4 py-3 border-t flex items-center gap-3">
            <Button onClick={finalizeWithDecisions} disabled={!reviewJob}>
              Download with decisions
            </Button>
            <div className="text-xs text-gray-500">
              Any item without a decision stays untranslated. Use bulk actions to speed up.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
