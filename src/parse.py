"""Extract SQL statements from model responses."""
from __future__ import annotations

import re


_SQL_FENCE = re.compile(r"\x60{3}sql\s*(.*?)\s*(?:\x60{3}|$)", re.IGNORECASE | re.DOTALL)
_STOP_MARKERS = (";", "\n\n", "Вопрос:", "<|endoftext|>")


def extract_sql(model_output: str) -> str:
    """Extract fenced SQL when present, then stop at the first end marker."""
    text = model_output or ""
    fenced = _SQL_FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    end = min((position for marker in _STOP_MARKERS
               if (position := text.find(marker)) >= 0), default=len(text))
    return text[:end].strip()
