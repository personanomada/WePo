import { useEffect, useRef, useState } from "react";
import { LOCALES, LocaleOption } from "../lib/locales";
import { Button } from "../components/ui/Button"; 
import { cn } from "../lib/cn"; 

type Props = {};

/**
 * Parses a stream of newline-delimited JSON, specifically handling the nested
 * structure of AI chat completion responses. It extracts the actual translation
 * content and calls the appropriate handler.
 */
async function streamAndParseAIResponse(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onMeta: (meta: any) => void,
  onBatch: (batch: any) => void,
  onDone: (done: any) => void,
  onError: (error: any) => void,
  signal: AbortSignal
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

        // Handle specific server events for progress and metadata
        if (serverEvent.type === "meta") {
          onMeta(serverEvent);
          continue;
        }
        if (serverEvent.type === "progress") { // Keep this for potential future backend changes
           onBatch(serverEvent.items || []);
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

        // --- Start of logic to handle raw AI responses ---
        // This is the key part for parsing the nested structure from your logs.
        let contentStr = null;
        if (serverEvent.choices && serverEvent.choices[0]?.message?.content) {
            contentStr = serverEvent.choices[0].message.content;
        }

        if (contentStr) {
          try {
            // The actual translations are a stringified JSON inside the 'content' field
            const innerData = JSON.parse(contentStr); 
            if (innerData.items && Array.isArray(innerData.items)) {
              onBatch(innerData.items);
            }
          } catch (e) {
            console.error("Failed to parse inner JSON from AI content:", contentStr, e);
          }
        }
        // --- End of logic to handle raw AI responses ---

      } catch (e) {
        console.error("Failed to parse outer NDJSON line:", line, e);
      }
    }
    
    await read();
  };
  
  await read();
}


export default function TranslatePanel({}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState("en");
  const [selectedLocales, setSelectedLocales] = useState<LocaleOption[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0, echoed: 0 });
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);

  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setError(null);
  }, [file, sourceLang, selectedLocales]);

  function onLocaleToggle(opt: LocaleOption) {
    setSelectedLocales((prev) => {
      const exists = prev.find((p) => p.code === opt.code);
      if (exists) {
        return prev.filter((p) => p.code !== opt.code);
      }
      return [...prev, opt];
    });
  }

  async function handleTranslateNDJSON() {
    if (!file) {
      setError("Please choose a PO file first.");
      return;
    }
    if (selectedLocales.length === 0) {
      setError("Choose at least one locale.");
      return;
    }

    setIsTranslating(true);
    setError(null);
    setStatus("Initializing translation...");
    setProgress({ done: 0, total: 1, echoed: 0 }); // Set total to 1 to show indeterminate progress initially

    const form = new FormData();
    form.append("po", file);
    form.append(
      "locales",
      selectedLocales.map((l) => l.code).join(",")
    );
    form.append("sourceLang", sourceLang);

    const aborter = new AbortController();
    controllerRef.current = aborter;

    try {
      const res = await fetch("http://127.0.0.1:8000/translate/ndjson", {
        method: "POST",
        body: form,
        signal: aborter.signal,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`${res.status} ${txt}`);
      }
      
      if (!res.body) {
        throw new Error("Response body is missing");
      }
      const reader = res.body.getReader();

      let currentDone = 0;

      await streamAndParseAIResponse(
        reader, 
        (meta) => { // onMeta
          setStatus(`Found ${meta.total} strings. Translating...`);
          setProgress(p => ({ ...p, total: meta.total || 0 }));
        },
        (batchItems) => { // onBatch
            currentDone += batchItems.length;
            setProgress(p => ({ ...p, done: currentDone }));
            setStatus(`Translated ${currentDone} of ${p.total} strings...`);
        },
        (doneEvent) => { // onDone
            setStatus("Translation complete! Preparing download...");
            const b = doneEvent.zipBase64 as string;
            const blob = b64ToBlob(b, "application/zip");
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "translations.zip";
            a.click();
            URL.revokeObjectURL(url);
        },
        (errorEvent) => { // onError
            setError(errorEvent.message || "An unknown error occurred");
        },
        aborter.signal
      );

    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || String(e));
      } else {
        setStatus("Translation cancelled.");
      }
    } finally {
      setIsTranslating(false);
      controllerRef.current = null;
    }
  }

  function b64ToBlob(b64: string, contentType: string) {
    const byteCharacters = atob(b64);
    const byteNumbers = new Array(byteCharacters.length)
      .fill(0)
      .map((_, i) => byteCharacters.charCodeAt(i));
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: contentType });
  }

  function cancelTranslation() {
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
  }

  return (
    <div className="space-y-6">
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
        <Button disabled={isTranslating || !file || selectedLocales.length === 0} onClick={handleTranslateNDJSON}>
          {isTranslating ? "Translating…" : "Translate"}
        </Button>
        {isTranslating && (
          <Button onClick={cancelTranslation} className="bg-gray-200 text-gray-800 hover:bg-gray-300">
            Cancel
          </Button>
        )}
      </div>

      {isTranslating && (
        <div className="mt-4 space-y-2">
          <div className="h-2 bg-gray-200 rounded overflow-hidden">
            <div
              className="h-2 bg-black rounded transition-all duration-300 ease-in-out"
              style={{
                width:
                  progress.total > 0
                    ? `${Math.min(100, (progress.done / progress.total) * 100)}%`
                    : "100%", // Indeterminate
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

      {error && (
        <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-red-700 text-sm mt-4">
          {error}
        </div>
      )}
    </div>
  );
}