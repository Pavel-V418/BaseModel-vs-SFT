"""Extract SQL statements from model responses."""
from __future__ import annotations

import re


_SQL_FENCE = re.compile(r"\x60{3}sql\s*(.*?)\s*(?:\x60{3}|$)", re.IGNORECASE | re.DOTALL)
_SQL_BOUNDARIES = re.compile(
    r"--[^\r\n]*|/\*.*?(?:\*/|$)"
    r"|'(?:''|[^'])*(?:'|$)"
    r'|"(?:""|[^"])*(?:"|$)'
    r"|`(?:``|[^`])*(?:`|$)|\[[^\]]*(?:\]|$)"
    r"|(?P<open>\()|(?P<close>\))|(?P<blank>\r?\n\r?\n)"
    r"|(?P<stop>;|Вопрос:|<\|endoftext\|>)",
    re.DOTALL,
)


def extract_sql(model_output: str) -> str:
    """Extract fenced SQL when present, then stop at the first end marker."""
    text = model_output or ""
    fenced = _SQL_FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    depth = 0
    for token in _SQL_BOUNDARIES.finditer(text):
        kind = token.lastgroup
        if kind == "stop" or (kind == "blank" and depth == 0):
            return text[:token.start()].strip()
        if kind == "open":
            depth += 1
        elif kind == "close":
            depth = max(0, depth - 1)
    return text.strip()
