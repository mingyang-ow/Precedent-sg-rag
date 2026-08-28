from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)?", re.UNICODE)
TOKENIZATION_VERSION = "nfkc-casefold-unicode-word-v1"


def tokenize(text: str) -> list[str]:
    """Tokenize legal text deterministically without language-specific resources."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_RE.findall(normalized)
