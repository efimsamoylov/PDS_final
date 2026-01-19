import re
import unicodedata
from typing import Any

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).lower()

    # unify separators
    s = s.replace("&", " ").replace("/", " ").replace("|", " ")

    # strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # remove punctuation, collapse whitespace
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
