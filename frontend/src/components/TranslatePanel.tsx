import { useEffect, useRef, useState } from "react";
import { LOCALES, LocaleOption } from "../lib/locales";
import { Button } from "../components/ui/Button";
import { cn } from "../lib/cn";

// Utility to stream and parse NDJSON from the translation API.
async function streamAndParseAIResponse(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onMeta: (meta: any) => void,
  onBatch: (batchItems: any[]) => void,
  onDone: (done: any) => void,
  onError: (error: any) => void,
  signal: AbortSignal,
  onProgress?: (prog: { done: number; total: number; echoed: number }) => void
) {
  const dec = new TextDecoder();
  let buffer = "";

  const read = async () => {
    if (signal.aborted) {
      reader.cancel();
      return;
    }
    const { value, done } = await reader.read();
    if (done) return;

    buffer += dec.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.trim() === "") continue;
      try {
        const serverEvent = JSON.parse(line);

        // Handle server-sent event types
        if (serverEvent.type === "meta") {
          onMeta(serverEvent);
          continue;
        }
        if (serverEvent.type === "progress") {
          // Update progress (translated count and echoed count)
          if (onProgress) {
            onProgress({
              done: serverEvent.done || 0,
              total: serverEvent.total || 0,
              echoed: serverEvent.echoed || 0,
            });
          }
          continue;
        }
        if (serverEvent.type === "done") {
          onDone(serverEvent);
          continue;
        }
        if (serverEvent.type === "error") {
          onError(serverEvent);
          continue;
        }

        // Handle raw AI response content (if any partial responses from provider)
        let contentStr: string | null = null;
        if (serverEvent.choices && serverEvent.choices[0]?.message?.content) {
          contentStr = serverEvent.choices[0].message.content;
        }
        if (contentStr) {
          try {
            const innerData = JSON.parse(contentStr);
            if (innerData.items && Array.isArray(innerData.items)) {
              onBatch(innerData.items);
            }
          } catch (e) {
            console.error("Failed to parse inner JSON from AI content:", contentStr, e);
          }
        }
      } catch (e) {
        console.error("Failed to parse NDJSON line:", line, e);
      }
    }

    await read();
  };

  await read();
}

export default function TranslatePanel() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState("en");
  const [selectedLocales, setSelectedLocales] = useState<LocaleOption[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0, echoed: 0 });
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  // Reset error when inputs change
  useEffect(() => {
    setError(null);
  }, [file, sourceLang, selectedLocales]);

  const onLocaleToggle = (opt: LocaleOption) => {
    setSelectedLocales((prev) => {
      const exists = prev.find((p) => p.code === opt.code);
      if (exists) {
        return prev.filter((p) => p.code !== opt.code);
      }
      return [...prev, opt];
    });
  };

  const handleTranslateNDJSON = async () => {
    if (!file) {
      setError("Please choose a PO file first.");
      return;
    }
    if (selectedLocales.length === 0) {
      setError("Choose at least one target locale.");
      return;
    }

    setIsTranslating(true);
    setError(null);
    setStatus("Initializing translation...");
    // Set an indeterminate progress bar until meta arrives
    setProgress({ done: 0, total: 0, echoed: 0 });

    // Prepare form data
    const form = new FormData();
    form.append("po", file);
    form.append("locales", selectedLocales.map((l) => l.code).join(","));
    form.append("sourceLang", sourceLang);

    // Determine API base URL from environment or use localhost
    const API_BASE =
      (import.meta as any).env?.VITE_BACKEND_URL?.replace(/\/$/, "") ||
      "http://localhost:8000";

    const aborter = new AbortController();
    controllerRef.current = aborter;

    try {
      const res = await fetch(`${API_BASE}/translate/ndjson`, {
        method: "POST",
        body: form,
        signal: aborter.signal,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`${res.status} ${txt}`);
      }
      if (!res.body) {
        throw new Error("No response body from translation API");
      }
      const reader = res.body.getReader();

      let currentDone = 0;
      let totalCount = 0;
      // Stream parsing callbacks
      await streamAndParseAIResponse(
        reader,
        // onMeta: initialize total count
        (meta) => {
          totalCount = meta.total || 0;
          setProgress((p) => ({ ...p, total: totalCount, done: 0, echoed: 0 }));
          setStatus(`Found ${meta.total} strings. Translating...`);
        },
        // onBatch: handle any batch of translated items (if provider streams partial JSON)
        (batchItems) => {
          currentDone += batchItems.length;
          setProgress((p) => ({ ...p, done: currentDone }));
          // Update status with current progress (using totalCount from meta)
          setStatus(`Translated ${currentDone} of ${totalCount} strings...`);
        },
        // onDone: all translations finished, provide download
        (doneEvent) => {
          setStatus("Translation complete! Preparing download...");
          const b64 = doneEvent.zipBase64 as string;
          const blob = b64ToBlob(b64, "application/zip");
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "translations.zip";
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        },
        // onError: handle errors from stream
        (errorEvent) => {
          setError(errorEvent.message || "An unknown error occurred during translation.");
        },
        aborter.signal,
        // onProgress: update progress state from server-sent progress events
        (prog) => {
          setProgress((prev) => ({
            ...prev,
            done: prog.done,
            total: prog.total,
            echoed: prog.echoed,
          }));
          setStatus(`Translated ${prog.done} of ${prog.total} strings...`);
        }
      );
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setError(e.message || String(e));
      } else {
        setStatus("Translation cancelled.");
      }
    } finally {
      setIsTranslating(false);
      controllerRef.current = null;
    }
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
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
  };

  return (
    <div className="space-y-6">
      {/* File input and action buttons */}
      <div className="space-y-2">
        <label className="text-sm font-medium">PO file</label>
        <input
          type="file"
          accept=".po"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Source language</label>
          <input
            className="w-full rounded-md border px-3 py-2 text-sm"
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            placeholder="en"
          />
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
                    className={cn(
                      "text-left text-xs rounded px-2 py-1 border",
                      active ? "bg-black text-white border-black" : "bg-white"
                    )}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
            {selectedLocales.length > 0 && (
              <div className="mt-2 text-xs text-gray-600">
                Selected: {selectedLocales.map((l) => l.code).join(", ")}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <Button
          disabled={isTranslating || !file || selectedLocales.length === 0}
          onClick={handleTranslateNDJSON}
        >
          {isTranslating ? "Translating…" : "Translate"}
        </Button>
        {isTranslating && (
          <Button onClick={cancelTranslation} className="bg-gray-200 text-gray-800 hover:bg-gray-300">
            Cancel
          </Button>
        )}
      </div>

      {/* Progress bar and status */}
      {isTranslating && (
        <div className="mt-4 space-y-2">
          <div className="h-2 bg-gray-200 rounded overflow-hidden">
            <div
              className="h-2 bg-black rounded transition-all duration-300 ease-in-out"
              style={{
                width:
                  progress.total > 0
                    ? `${Math.min(100, (progress.done / progress.total) * 100)}%`
                    : "100%", // indeterminate if total not known
              }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-600">
            <span>{status || "Waiting..."}</span>
            <span>
              {progress.total > 1 ? `${progress.done} / ${progress.total}` : ""}
              {progress.echoed > 0 && ` (retried: ${progress.echoed})`}
            </span>
          </div>
        </div>
      )}

      {/* Error message display */}
      {error && (
        <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-red-700 text-sm mt-4">
          {error}
        </div>
      )}
    </div>
  );
}
