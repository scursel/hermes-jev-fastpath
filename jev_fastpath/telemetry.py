"""Redacted, bounded JSONL audit telemetry plus the shared redaction primitives.

The writer serializes only a fixed field set: hashed session/turn identities, bounded
redacted previews, and flat numeric Jev usage. Nested provider payloads are discarded and
every I/O failure is swallowed after one debug signal — telemetry can never affect a turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .types import TelemetryEvent

logger = logging.getLogger("jev_fastpath.telemetry")

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


class TelemetryWriter:
    """Append-only JSONL decision log under ``<HERMES_HOME>/artifacts/jev_fastpath``."""

    def __init__(self, home: Path, max_preview: int = 160):
        self.path = Path(home) / "artifacts" / "jev_fastpath" / "decisions.jsonl"
        self.max_preview = int(max_preview)
        self._lock = threading.Lock()

    def write(self, event: TelemetryEvent, context: Mapping[str, Any]) -> None:
        """Serialize one fixed-shape record; never raises, never affects the turn."""
        try:
            self._write(event, context)
        except Exception:
            logger.debug("jev-fastpath telemetry write failed", exc_info=True)

    def _write(self, event: TelemetryEvent, context: Mapping[str, Any]) -> None:
        def digest(value: object) -> str:
            return hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()[:16]

        usage = {
            key: value
            for key in ("cost", "input_tokens", "output_tokens")
            if isinstance((value := event.usage.get(key)), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": str(context.get("mode") or ""),
            "session_hash": digest(context.get("session_id")),
            "turn_hash": digest(context.get("turn_id")),
            "platform": str(context.get("platform") or "")[:64],
            "provider": str(context.get("provider") or "")[:128],
            "model": str(context.get("model") or "")[:256],
            "api_mode": str(context.get("api_mode") or "")[:64],
            "text_hash": digest(event.text),
            "text_preview": redact(event.text)[: self.max_preview],
            "candidates": list(event.candidates),
            "selected_handler": event.selected_handler,
            "confidence": event.confidence,
            "short_circuit_probability": event.short_circuit_probability,
            "latency_ms": event.latency_ms,
            "outcome": event.outcome,
            "reason": redact(event.reason)[:160],
            "provider_call_avoided": event.outcome == "short_circuit",
            "usage": usage,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as exc:
            raise RuntimeError("telemetry sink unavailable") from exc
