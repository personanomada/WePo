import re
from collections import Counter
from typing import Dict, List

import polib

# 1) printf style
PRINTF_RE = re.compile(r"%(?:\d+\$)?[+-]?(?:\d+)?(?:\.\d+)?[sdifouxXc]", re.UNICODE)

# 2) {name} or {{var}}
BRACE_RE = re.compile(r"(\{\{\s*[\w.\-]+\s*\}\}|\{\s*[\w.\-]+\s*\})", re.UNICODE)

# 3) HTML tags like <a> and </a>
HTML_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9\-]*)\b[^>]*>", re.UNICODE)


def extract_printf(text: str) -> List[str]:
    return PRINTF_RE.findall(text or "")


def extract_braces(text: str) -> List[str]:
    return BRACE_RE.findall(text or "")


def extract_html_tags(text: str) -> List[str]:
    tags = []
    for m in HTML_TAG_RE.finditer(text or ""):
        full = m.group(0)
        name = m.group(1)
        if full.startswith("</"):
            tags.append(f"</{name}>")
        else:
            tags.append(f"<{name}>")
    return tags


def extract_all_placeholders(text: str) -> Dict[str, Counter]:
    return {
        "printf": Counter(extract_printf(text)),
        "brace": Counter(extract_braces(text)),
        "html": Counter(extract_html_tags(text)),
    }


def merge_counters(a: Dict[str, Counter], b: Dict[str, Counter]) -> Dict[str, Counter]:
    out = {}
    for k in {"printf", "brace", "html"}:
        out[k] = a.get(k, Counter()).copy()
        out[k].update(b.get(k, Counter()))
    return out


def extract_all_placeholders_from_entry(entry: polib.POEntry) -> Dict[str, Counter]:
    base = extract_all_placeholders(entry.msgid or "")
    if entry.msgid_plural:
        base = merge_counters(base, extract_all_placeholders(entry.msgid_plural))
    return base


def validate_translation_placeholders(expected: Dict[str, Counter], translated: str) -> None:
    found = {
        "printf": Counter(extract_printf(translated)),
        "brace": Counter(extract_braces(translated)),
        "html": Counter(extract_html_tags(translated)),
    }
    for cls in ("printf", "brace", "html"):
        exp_c = expected.get(cls, {})
        for token, cnt in exp_c.items():
            if found[cls].get(token, 0) < cnt:
                raise ValueError(f"Missing {cls} token '{token}'")


def extract_samples_from_catalog(catalog: polib.POFile) -> List[str]:
    samples = []
    for e in catalog:
        if e.obsolete:
            continue
        for t in set(extract_printf(e.msgid) + extract_braces(e.msgid) + extract_html_tags(e.msgid)):
            if t not in samples:
                samples.append(t)
        if e.msgid_plural:
            for t in set(extract_printf(e.msgid_plural) + extract_braces(e.msgid_plural) + extract_html_tags(e.msgid_plural)):
                if t not in samples:
                    samples.append(t)
        if len(samples) >= 50:
            break
    return samples
