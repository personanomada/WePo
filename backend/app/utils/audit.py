from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
import polib

from .placeholders import extract_all_placeholders_from_entry, validate_translation_placeholders
from .plurals import nplurals_for_locale, get_plural_forms_header

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

# Map common language codes to pyspellchecker dictionaries.
# Fallback to 'en' if unknown or unavailable.
SPELL_LANG_MAP = {
    "en": "en",
    "en_US": "en",
    "en_GB": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pt_BR": "pt",
    "nl": "nl",
    "pl": "pl",
}


def _language_from_headers(meta: Dict[str, str]) -> Optional[str]:
    lang = (meta or {}).get("Language") or (meta or {}).get("Language-Team")
    if not lang:
        return None
    lang = lang.strip()
    # Normalize like "es_ES" -> "es"
    if "_" in lang:
        lang = lang.split("_", 1)[0]
    if "-" in lang:
        lang = lang.split("-", 1)[0]
    return lang or None


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _spell_checker(lang_code: str):
    try:
        from spellchecker import SpellChecker  # type: ignore
    except Exception:
        return None
    dic = SPELL_LANG_MAP.get(lang_code, "en")
    try:
        return SpellChecker(language=dic, case_sensitive=False)
    except Exception:
        try:
            return SpellChecker(language="en", case_sensitive=False)
        except Exception:
            return None


def _looks_english(text: str) -> bool:
    # Very light heuristic: if >80% of letters are ASCII letters, assume English.
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    ascii_count = sum(1 for c in letters if "A" <= c <= "Z" or "a" <= c <= "z")
    return (ascii_count / max(1, len(letters))) >= 0.8


def audit_po(
    catalog: polib.POFile,
    assume_source_lang: Optional[str] = None,
) -> Dict:
    """
    Inspect a PO catalog and return an audit report with a summary and issue list.
    This does NOT modify the catalog. All checks are conservative and safe.
    """
    meta = dict(catalog.metadata or {})
    lang_from_header = _language_from_headers(meta)
    language = lang_from_header or assume_source_lang or "en"

    issues: List[Dict] = []
    counts = {
        "entries": len(catalog),
        "plural_entries": 0,
        "untranslated": 0,
        "identical": 0,
        "placeholder_mismatches": 0,
        "html_mismatches": 0,
        "missing_plural_forms": 0,
        "duplicates": 0,
        "spelling_warnings": 0,
        "suspicious_source_strings": 0,
    }

    # Structural: duplicate msgid/msgctxt, empty msgid, header sanity
    seen = set()
    for idx, e in enumerate(catalog):
        key = (e.msgctxt or "", e.msgid)
        if key in seen:
            issues.append({
                "type": "duplicate",
                "index": idx,
                "msgid": e.msgid,
                "detail": "Duplicate msgid/msgctxt appears multiple times.",
            })
            counts["duplicates"] += 1
        else:
            seen.add(key)
        if not e.msgid:
            issues.append({
                "type": "structural",
                "index": idx,
                "msgid": "",
                "detail": "Empty msgid found.",
            })

    # Plural headers sanity (if target language appears set)
    pf = meta.get("Plural-Forms", "")
    if language:
        expected_header = get_plural_forms_header(language)
        if pf and pf.strip() != expected_header:
            issues.append({
                "type": "header",
                "index": -1,
                "msgid": "",
                "detail": f"Plural-Forms header mismatch. Found: `{pf}` Expected: `{expected_header}`",
            })

    # Entry checks
    npl = nplurals_for_locale(language)
    sc = _spell_checker(language)
    do_spell = sc is not None

    for idx, e in enumerate(catalog):
        is_plural = bool(e.msgid_plural)
        if is_plural:
            counts["plural_entries"] += 1

        expected = extract_all_placeholders_from_entry(e)

        # Untranslated and identical checks
        if e.msgid_plural:
            # Any of the forms missing or empty
            missing_any = False
            for form_index in range(npl):
                t = e.msgstr_plural.get(form_index) if isinstance(e.msgstr_plural, dict) else None
                if not t:
                    missing_any = True
            if missing_any:
                issues.append({
                    "type": "plural_missing",
                    "index": idx,
                    "msgid": e.msgid,
                    "detail": f"Missing one or more plural forms for nplurals={npl}.",
                })
                counts["missing_plural_forms"] += 1
        else:
            if not e.msgstr:
                counts["untranslated"] += 1
            elif e.msgid.strip() == e.msgstr.strip():
                counts["identical"] += 1
                issues.append({
                    "type": "identical",
                    "index": idx,
                    "msgid": e.msgid,
                    "detail": "msgid and msgstr are identical; likely untranslated.",
                })

        # Placeholder/HTML validation against existing translations (if any)
        def _validate_text(t: Optional[str], form: Optional[int] = None):
            if not t:
                return
            try:
                validate_translation_placeholders(expected, t)
            except ValueError as ve:
                # classify HTML vs other placeholder
                msg = str(ve)
                kind = "placeholder_mismatch"
                if "<" in msg or ">" in msg:
                    kind = "html_mismatch"
                counts["placeholder_mismatches"] += 1 if kind == "placeholder_mismatch" else 0
                counts["html_mismatches"] += 1 if kind == "html_mismatch" else 0
                issues.append({
                    "type": kind,
                    "index": idx,
                    "msgid": e.msgid if form is None else f"{e.msgid} [plural {form}]",
                    "detail": msg,
                })

        if e.msgid_plural:
            if isinstance(e.msgstr_plural, dict):
                for i_form, t in e.msgstr_plural.items():
                    _validate_text(t, i_form)
        else:
            _validate_text(e.msgstr)

        # Spelling warnings (optional)
        if do_spell:
            texts = []
            if e.msgid_plural:
                texts.extend((e.msgstr_plural or {}).values())
            else:
                texts.append(e.msgstr)

            for t in texts:
                if not t:
                    continue
                words = _tokenize_words(t)
                if not words:
                    continue
                # Ignore placeholders and HTML-ish tokens
                filtered = [w for w in words if not (w.startswith("%") or w.startswith("{") or w.lower() in {"http", "https"})]
                unknown = sc.unknown(filtered)
                # Heuristic: 3+ unknown words may indicate spelling issues
                if len(unknown) >= 3:
                    counts["spelling_warnings"] += 1
                    issues.append({
                        "type": "spelling",
                        "index": idx,
                        "msgid": e.msgid,
                        "detail": f"Possible misspellings: {', '.join(list(unknown)[:8])}",
                    })
                    break

    # Suspicious source language (only if header claims English)
    if (language in {"en", "en_US", "en_GB"} or (assume_source_lang == "en" and not lang_from_header)):
        non_english = 0
        checked = 0
        for e in catalog[: min(200, len(catalog))]:  # sample for speed
            if not e.msgid:
                continue
            checked += 1
            if not _looks_english(e.msgid):
                non_english += 1
        if checked >= 10 and non_english / checked > 0.1:
            counts["suspicious_source_strings"] = non_english
            issues.append({
                "type": "source_language",
                "index": -1,
                "msgid": "",
                "detail": f"{non_english}/{checked} source strings do not look like English though Language is set to {language}.",
            })

    summary = {
        "counts": counts,
        "language_detected": language,
        "spellcheck_enabled": bool(sc),
    }

    return {"summary": summary, "issues": issues}
