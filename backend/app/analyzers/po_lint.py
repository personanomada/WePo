# app/analyzers/po_lint.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import polib

# -------------------------
# Local helpers (self-contained)
# -------------------------

# printf-style placeholders used by WP: %s, %d, %1$s, %2$d, etc.
PRINTF_RE = re.compile(
    r"%(?:(?P<pos>\d+)\$)?[-+ #0]*(?:\d+)?(?:\.\d+)?(?P<type>[bcdeEfFgGiosuxX%])"
)

# Simple named placeholder: {name}
BRACE_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")

def _extract_placeholders(text: str) -> Counter:
    """
    Extract placeholders from a string and return a Counter with normalized tokens.
    - printf tokens normalized to e.g. "%1$s" if numbered, otherwise "%s"
    - named placeholders keep their literal form, e.g. "{name}"
    """
    if not text:
        return Counter()

    tokens: List[str] = []

    for m in PRINTF_RE.finditer(text):
        typ = m.group("type")
        # treat "%%" as a literal and ignore it
        if typ == "%":
            continue
        pos = m.group("pos")
        if pos:
            tokens.append(f"%{pos}${typ}")
        else:
            tokens.append(f"%{typ}")

    for m in BRACE_RE.finditer(text):
        tokens.append("{" + m.group(1) + "}")

    return Counter(tokens)


def extract_all_placeholders_from_entry(e: polib.POEntry) -> Counter:
    """
    For validation, we require that translations preserve all placeholders.
    We take the union across msgid and (if present) msgid_plural.
    """
    src = e.msgid or ""
    if e.msgid_plural:
        src += " " + e.msgid_plural
    return _extract_placeholders(src)


def validate_translation_placeholders(expected: Counter, translation: str) -> None:
    """
    Raise ValueError if the translation does not contain the same placeholder multiset.
    """
    got = _extract_placeholders(translation or "")
    # Missing tokens
    missing = []
    for tok, cnt in expected.items():
        if got[tok] < cnt:
            missing.append(f"{tok}×{cnt - got[tok]}")
    # Extra tokens
    extra = []
    for tok, cnt in got.items():
        if cnt > expected.get(tok, 0):
            extra.append(f"{tok}×{cnt - expected.get(tok, 0)}")
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("extra " + ", ".join(extra))
        raise ValueError("; ".join(parts))


def detect_locale_from_po(po: polib.POFile, filename: Optional[str] = None) -> str:
    """
    Best-effort locale detection:
      1) header 'Language'
      2) filename like 'es_ES.po' or 'plugin-es.po'
      3) default 'en'
    Returns a lowercase IETF-ish tag (e.g., 'es' or 'es-es').
    """
    lang = (po.metadata.get("Language") or "").strip()
    if lang:
        # normalize separators and case
        lang = lang.replace("_", "-").lower()
        return lang

    if filename:
        name = os.path.basename(filename)
        m = re.search(r"([a-zA-Z]{2,3}(?:[_-][a-zA-Z]{2,3})?)\.po$", name)
        if not m:
            m = re.search(r"-([a-zA-Z]{2,3}(?:[_-][a-zA-Z]{2,3})?)", name)
        if m:
            return m.group(1).replace("_", "-").lower()

    return "en"


def nplurals_for_locale(locale: str) -> int:
    """
    Minimal nplurals map (covers common cases; default=2).
    """
    loc = (locale or "").lower()
    loc_base = loc.split("-")[0]
    # 3-plural languages
    if loc_base in {"ru", "uk", "sr", "hr", "bs", "cs", "sk", "pl", "ro"}:
        return 3
    # 4-plural examples
    if loc_base in {"sl"}:
        return 4
    # 1-plural (no plural) examples
    if loc_base in {"ja", "ko", "zh", "th"}:
        return 1
    # Default and most Western European languages
    return 2


def nplurals_from_po(po: polib.POFile) -> Optional[int]:
    pf = po.metadata.get("Plural-Forms") or ""
    m = re.search(r"nplurals\s*=\s*(\d+)", pf)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

# -------------------------
# Linter
# -------------------------

TAG_RE = re.compile(r"</?([a-zA-Z0-9]+)(\s[^>]*)?>")
END_PUNCT = ".!?:…？！。"

@dataclass
class LintItem:
    index: int
    key: str
    ctx: Optional[str]
    rule: str
    severity: str  # "error" | "warn" | "info"
    message: str
    suggestion: Optional[str] = None          # short human suggestion
    fix: Optional[Tuple[str, str]] = None     # ("replace_msgstr" | "replace_plural_i", new_text)
    sample: Optional[str] = None
    source: Optional[str] = None              # NEW: show source string
    translation: Optional[str] = None         # NEW: show current translation

def _key_for(idx: int, plural_index: Optional[int]) -> str:
    return f"{idx}|p{plural_index}" if plural_index is not None else f"{idx}"

def _same_end_punct(a: str, b: str) -> bool:
    sa = (a or "").strip()
    sb = (b or "").strip()
    if not sa or not sb:
        return True
    ca = sa[-1] if sa[-1] in END_PUNCT else None
    cb = sb[-1] if sb[-1] in END_PUNCT else None
    return ca == cb

def _first_cap_policy(s: str) -> Optional[bool]:
    s = (s or "").strip()
    if not s:
        return None
    if s[0].isalpha():
        return s[0].isupper()
    return None

def _html_tags(s: str) -> List[str]:
    return [m.group(1).lower() for m in TAG_RE.finditer(s or "")]

def analyze_po(src: polib.POFile, locale: Optional[str] = None, filename: Optional[str] = None) -> Dict:
    """
    Produce a structured lint report. Each lint has an optional 'fix' that the client can apply.
    This file is self-contained to avoid circular imports with app.main.
    """
    detected_locale = locale or detect_locale_from_po(src, filename)
    npl = nplurals_from_po(src) or nplurals_for_locale(detected_locale)

    items: List[LintItem] = []

    for idx, e in enumerate(src):
        if e.obsolete:
            continue

        expected_placeholders = extract_all_placeholders_from_entry(e)
        src_tags = sorted(_html_tags(e.msgid + (e.msgid_plural or "")))
        key_base = _key_for(idx, None)
        ctx = e.msgctxt

        if e.msgid_plural:
            # plural forms
            for p in range(npl):
                key = _key_for(idx, p)
                t = e.msgstr_plural.get(p, "") or ""
                src_text = e.msgid_plural or e.msgid

                # 1) missing plural text
                if not t.strip():
                    items.append(LintItem(
                        idx, key, ctx, "missing_plural", "error",
                        "Plural form is empty", suggestion="Provide a translation for this plural form",
                        source=src_text, translation=t
                    ))
                else:
                    # 2) placeholder check
                    try:
                        validate_translation_placeholders(expected_placeholders, t)
                    except ValueError as ve:
                        items.append(LintItem(
                            idx, key, ctx, "placeholder_mismatch", "error",
                            f"Placeholders differ: {ve}", suggestion="Insert all placeholders from source",
                            source=src_text, translation=t
                        ))
                    # 3) HTML tags
                    if sorted(_html_tags(t)) != src_tags:
                        items.append(LintItem(
                            idx, key, ctx, "html_mismatch", "error",
                            "HTML tags differ from source", suggestion="Preserve same tag set and order",
                            source=src_text, translation=t
                        ))

                    # 4) punctuation
                    if not _same_end_punct(src_text, t):
                        sug = None
                        src_trim = (src_text or "").strip()
                        if src_trim and src_trim[-1] in END_PUNCT:
                            sug = t.rstrip() + src_trim[-1]
                        items.append(LintItem(
                            idx, key, ctx, "punctuation", "warn",
                            "Ending punctuation differs",
                            suggestion="Match ending punctuation",
                            fix=("replace_plural_i", sug) if sug else None,
                            sample=t, source=src_text, translation=t
                        ))

                    # 5) capitalization
                    src_cap = _first_cap_policy(src_text)
                    dst_cap = _first_cap_policy(t)
                    if src_cap is not None and dst_cap is not None and src_cap != dst_cap:
                        if dst_cap is False and t[:1].isalpha():
                            items.append(LintItem(
                                idx, key, ctx, "capitalization", "warn",
                                "Capitalization differs",
                                suggestion="Start sentence with a capital letter",
                                fix=("replace_plural_i", t[:1].upper() + t[1:]),
                                source=src_text, translation=t
                            ))

                    # 6) length ratio
                    if src_text and t:
                        ratio = len(t) / max(1, len(src_text))
                        if ratio < 0.5 or ratio > 2.5:
                            items.append(LintItem(
                                idx, key, ctx, "length_ratio", "info",
                                f"Length ratio {ratio:.2f} looks unusual",
                                source=src_text, translation=t
                            ))

        else:
            t = e.msgstr or ""
            src_text = e.msgid

            # 1) identical to source
            if t and t == src_text:
                items.append(LintItem(
                    idx, key_base, ctx, "identical", "warn",
                    "Translation identical to source",
                    suggestion="Accept echo, edit, or re-translate",
                    source=src_text, translation=t
                ))

            # 2) placeholder
            if t:
                try:
                    validate_translation_placeholders(expected_placeholders, t)
                except ValueError as ve:
                    items.append(LintItem(
                        idx, key_base, ctx, "placeholder_mismatch", "error",
                        f"Placeholders differ: {ve}", suggestion="Insert all placeholders from source",
                        source=src_text, translation=t
                    ))

            # 3) HTML tag set
            if t and sorted(_html_tags(t)) != src_tags:
                items.append(LintItem(
                    idx, key_base, ctx, "html_mismatch", "error",
                    "HTML tags differ from source", suggestion="Preserve same tag set and order",
                    source=src_text, translation=t
                ))

            # 4) punctuation
            if t and not _same_end_punct(src_text, t):
                sug = None
                src_trim = (src_text or "").strip()
                if src_trim and src_trim[-1] in END_PUNCT:
                    sug = t.rstrip() + src_trim[-1]
                items.append(LintItem(
                    idx, key_base, ctx, "punctuation", "warn",
                    "Ending punctuation differs",
                    suggestion="Match ending punctuation",
                    fix=("replace_msgstr", sug) if sug else None,
                    sample=t, source=src_text, translation=t
                ))

            # 5) capitalization
            src_cap = _first_cap_policy(src_text)
            dst_cap = _first_cap_policy(t) if t else None
            if t and src_cap is not None and dst_cap is not None and src_cap != dst_cap:
                if dst_cap is False and t[:1].isalpha():
                    items.append(LintItem(
                        idx, key_base, ctx, "capitalization", "warn",
                        "Capitalization differs",
                        suggestion="Start sentence with a capital letter",
                        fix=("replace_msgstr", t[:1].upper() + t[1:]),
                        source=src_text, translation=t
                    ))

            # 6) length ratio
            if t:
                ratio = len(t) / max(1, len(src_text))
                if ratio < 0.5 or ratio > 2.5:
                    items.append(LintItem(
                        idx, key_base, ctx, "length_ratio", "info",
                        f"Length ratio {ratio:.2f} looks unusual",
                        source=src_text, translation=t
                    ))

            # 7) whitespace
            if t and (t != t.strip() or "  " in t):
                items.append(LintItem(
                    idx, key_base, ctx, "whitespace", "info",
                    "Leading/trailing or double spaces",
                    suggestion="Trim and collapse spaces",
                    fix=("replace_msgstr", re.sub(r"\s{2,}", " ", t.strip())),
                    source=src_text, translation=t
                ))

    # Build summary
    by_rule: Dict[str, int] = {}
    by_sev = {"error": 0, "warn": 0, "info": 0}
    for it in items:
        by_rule[it.rule] = by_rule.get(it.rule, 0) + 1
        by_sev[it.severity] += 1

    return {
        "detected_locale": detected_locale,
        "nplurals": npl,
        "summary": {
            "total": len(items),
            "by_severity": by_sev,
            "by_rule": by_rule,
        },
        "items": [it.__dict__ for it in items],
    }
