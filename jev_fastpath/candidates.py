"""Conservative candidate detection: an eligibility filter, never the route decision.

Every local parser must independently recognize the request before its handler ID can be
proposed to Jev; ``normal_llm`` is never a local candidate.
"""

from __future__ import annotations

import re

from .arithmetic import ArithmeticRejected, evaluate_expression, extract_expression
from .config import KNOWN_HANDLERS
from .handlers import (
    classify_acknowledgement,
    classify_clock_request,
    classify_identity_fields,
    classify_status_request,
)

KNOWN_ORDER = tuple(KNOWN_HANDLERS)

_REJECT_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"(?i)\b(?:token|password|passwd|secret|api[_-]?key)\s*[:=]"),
    re.compile(r"(?:&&|\|\||;|`|\$\(|>|<)"),
    re.compile(r"(?i)\b(?:delete|remove|apague|exclua|envie|publique|compre|pague|reinicie|execute)\b"),
)


def _is_calculator_request(text: str) -> bool:
    try:
        evaluate_expression(extract_expression(text))
    except ArithmeticRejected:
        return False
    return True


def detect_candidates(text: str, settings) -> tuple[str, ...]:
    """Return the ordered tuple of locally supported handler IDs for this text."""
    if not isinstance(text, str):
        return ()
    stripped = text.strip()
    if not stripped or len(text) > settings.max_input_chars:
        return ()
    # Hermes owns slash-command routing; the fast path never competes with it.
    if stripped.startswith("/"):
        return ()
    if any(pattern.search(text) for pattern in _REJECT_PATTERNS):
        return ()
    candidates: list[str] = []
    if "calculator" in settings.enabled_handlers and _is_calculator_request(text):
        candidates.append("calculator")
    if "clock" in settings.enabled_handlers and classify_clock_request(text):
        candidates.append("clock")
    if "runtime_identity" in settings.enabled_handlers and classify_identity_fields(text):
        candidates.append("runtime_identity")
    if "acknowledgement" in settings.enabled_handlers and classify_acknowledgement(text):
        candidates.append("acknowledgement")
    if "fastpath_status" in settings.enabled_handlers and classify_status_request(text):
        candidates.append("fastpath_status")
    return tuple(candidates)
