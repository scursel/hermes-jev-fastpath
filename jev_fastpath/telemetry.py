"""Secret redaction primitives shared by the Jev client and the audit writer.

Redaction is defense in depth: candidates and messages are already filtered upstream, but
anything that ever reaches a log, a telemetry row, or the TypeSafe request is scrubbed of
credential-shaped substrings first.
"""

from __future__ import annotations

import re

# (pattern, replacement) pairs; every match is credential-shaped, never prose.
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(?:authorization|api[_-]?key|api[_-]?secret|access[_-]?token|"
                r"refresh[_-]?token|password|passwd|secret|senha)\b(\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"), "<redacted>"),
    (re.compile(r"(?i)\bts_[A-Za-z0-9_-]{8,}\b"), "<redacted>"),
    (re.compile(r"(?i)\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b"), "<redacted-jwt>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<redacted-email>"),
)


def redact(value: str) -> str:
    """Scrub credential-shaped substrings from one string."""
    if not isinstance(value, str):
        return ""
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact_and_bound(text: str, max_chars: int) -> str:
    """Redact and hard-bound a user message before it leaves the process."""
    if not isinstance(text, str):
        return ""
    return redact(text)[: max(0, int(max_chars))]
