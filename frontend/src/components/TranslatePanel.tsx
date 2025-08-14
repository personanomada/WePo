import { useEffect, useRef, useState } from "react";
import { LOCALES, LocaleOption } from "../lib/locales";
import { Button } from "../components/ui/Button"; // adjust if your Button path differs
import { cn } from "../lib/cn"; // optional helper; remove if you don't have it

type Props = {};

export default function TranslatePanel({}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState("en");
  const [selectedLocales, setSelectedLocales] = useState<LocaleOption[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0, echoed: 0 });
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
    setProgress({ done: 0, total: 0, echoed: 0 });

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

      const reader = res.body!.getReader();
      const dec = new TextDecoder();

      let leftover = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        leftover += chunk;
        const lines = leftover.split("\n");
        leftover = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line);
          if (msg.type === "meta") {
            setProgress((p) => ({ ...p, total: msg.total || 0 }));
          } else if (msg.type === "progress") {
            setProgress({
              done: msg.done ?? 0,
              total: msg.total ?? 0,
              echoed: msg.echoed ?? 0,
            });
          } else if (msg.type === "done") {
            const b = msg.zipBase64 as string;
            const blob = b64ToBlob(b, "application/zip");
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "translations.zip";
            a.click();
            URL.revokeObjectURL(url);
          } else if (msg.type === "error") {
            setError(msg.message || "Unknown error");
          }
        }
      }
    } catch (e: any) {
      setError(e.message || String(e));
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

      <Button disabled={isTranslating} onClick={handleTranslateNDJSON}>
        {isTranslating ? "Translating…" : "Translate with progress"}
      </Button>

      <div className="mt-4 h-2 bg-gray-200 rounded">
        <div
          className="h-2 bg-black rounded"
          style={{
            width:
              progress.total > 0
                ? `${Math.min(100, Math.round((progress.done / progress.total) * 100))}%`
                : "0%",
          }}
        />
      </div>
      <div className="text-xs text-gray-600">
        {progress.done} / {progress.total}
        {progress.echoed ? ` • echoed (retried): ${progress.echoed}` : ""}
      </div>

      {error && (
        <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-red-700 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
