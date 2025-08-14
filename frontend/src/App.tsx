import React, { useState } from "react";
import AnalyzePanel from "./components/AnalyzePanel";
import SettingsPanel from "./components/SettingsPanel";
import TranslatePanel from "./components/TranslatePanel";

export default function App() {
  const [active, setActive] = useState<"analyze" | "settings" | "translate">("analyze");

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">WP Plugin PO File AI Translator</h1>
          <nav className="flex gap-2">
            <button className={`px-3 py-2 rounded ${active==="analyze"?"bg-gray-900 text-white":"bg-gray-100"}`} onClick={()=>setActive("analyze")}>Analyze</button>
            <button className={`px-3 py-2 rounded ${active==="settings"?"bg-gray-900 text-white":"bg-gray-100"}`} onClick={()=>setActive("settings")}>Settings</button>
            <button className={`px-3 py-2 rounded ${active==="translate"?"bg-gray-900 text-white":"bg-gray-100"}`} onClick={()=>setActive("translate")}>Translate</button>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {active === "analyze" && <AnalyzePanel />}
        {active === "settings" && <SettingsPanel />}
        {active === "translate" && <TranslatePanel />}
      </main>

      <footer className="mx-auto max-w-6xl px-4 py-6 text-sm text-gray-500">
        Built for WordPress plugin PO and MO workflows. Preserve placeholders and plural forms.
      </footer>
    </div>
  );
}
