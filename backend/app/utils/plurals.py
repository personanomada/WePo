from typing import Dict

PLURAL_MAP: Dict[str, str] = {
    "en": "nplurals=2; plural=(n != 1);",
    "en_US": "nplurals=2; plural=(n != 1);",
    "en_GB": "nplurals=2; plural=(n != 1);",
    "fr": "nplurals=2; plural=(n > 1);",
    "de": "nplurals=2; plural=(n != 1);",
    "es": "nplurals=2; plural=(n != 1);",
    "it": "nplurals=2; plural=(n != 1);",
    "pt": "nplurals=2; plural=(n != 1);",
    "pt_BR": "nplurals=2; plural=(n > 1);",
    "nl": "nplurals=2; plural=(n != 1);",
    "sv": "nplurals=2; plural=(n != 1);",
    "no": "nplurals=2; plural=(n != 1);",
    "da": "nplurals=2; plural=(n != 1);",
    "ro": "nplurals=3; plural=(n==1?0:(n==0|| (n%100>0 && n%100<20))?1:2);",
    "ru": "nplurals=3; plural=(n%10==1 && n%100!=11?0:(n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20)?1:2));",
    "uk": "nplurals=3; plural=(n%10==1 && n%100!=11?0:(n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20)?1:2));",
    "pl": "nplurals=3; plural=(n==1?0:(n%10>=2 && n%10<=4 && (n%100<12 || n%100>14)?1:2));",
    "cs": "nplurals=3; plural=(n==1)?0:(n>=2 && n<=4)?1:2;",
    "sk": "nplurals=3; plural=(n==1)?0:(n>=2 && n<=4)?1:2;",
    "sl": "nplurals=4; plural=(n%100==1?1:n%100==2?2:n%100==3 or n%100==4?3:0);",
    "lt": "nplurals=3; plural=(n%10==1 && n%100!=11?0:(n%10>=2 && (n%100<10 || n%100>=20)?1:2));",
    "lv": "nplurals=3; plural=(n%10==0 || (n%100>=11 && n%100<=19)?0:(n%10==1 && n%100!=11)?1:2);",
    "ga": "nplurals=5; plural=(n==1?0:n==2?1:n<7?2:n<11?3:4);",
    "ar": "nplurals=6; plural=(n==0?0:n==1?1:n==2?2:n%100>=3 && n%100<=10?3:n%100>=11 && n%100<=99?4:5);",
    "he": "nplurals=2; plural=(n != 1);",
    "tr": "nplurals=2; plural=(n > 1);",
    "el": "nplurals=2; plural=(n != 1);",
    "bg": "nplurals=2; plural=(n != 1);",
    "sr": "nplurals=3; plural=(n%10==1 && n%100!=11?0:(n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20)?1:2));",
    "hu": "nplurals=2; plural=(n != 1);",
    "ja": "nplurals=1; plural=0;",
    "ko": "nplurals=1; plural=0;",
    "zh": "nplurals=1; plural=0;",
    "zh_CN": "nplurals=1; plural=0;",
    "zh_TW": "nplurals=1; plural=0;",
}

def normalize_locale(loc: str) -> str:
    loc = (loc or "").replace("-", "_")
    parts = loc.split("_")
    if len(parts) == 1:
        return parts[0]
    return parts[0] + "_" + parts[1].upper()

def get_plural_forms_header(locale: str) -> str:
    norm = normalize_locale(locale)
    return PLURAL_MAP.get(norm) or PLURAL_MAP.get(norm.split("_")[0], "nplurals=2; plural=(n != 1);")

def nplurals_for_locale(locale: str) -> int:
    header = get_plural_forms_header(locale)
    try:
        start = header.index("nplurals=") + len("nplurals=")
        end = header.index(";", start)
        return int(header[start:end])
    except Exception:
        return 2
