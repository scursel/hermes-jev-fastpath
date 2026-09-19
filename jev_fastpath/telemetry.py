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
                r"refresh[_-]?token|secret[_-]?access[_-]?key|password|passwd|secret|senha|"
                r"aws[_-]?secret[_-]?access[_-]?key)\b(\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"(?i)\b(?:password|passwd|senha)\b\s+(?:is|é)\s+\S+"), "<redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(?:sk|pk)[-_](?:live|test)[-_][A-Za-z0-9]{8,}\b"), "<redacted>"),
    (re.compile(r"(?i)\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"), "<redacted>"),
    (re.compile(r"(?i)\bts_[A-Za-z0-9_-]{8,}\b"), "<redacted>"),
    (re.compile(r"(?i)\bAIza[0-9A-Za-z_-]{30,}\b"), "<redacted>"),
    (re.compile(r"-----BEGIN (?:[A-Z ]* )?PRIVATE KEY-----"), "<redacted-pem>"),
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
    """Hard-bound FIRST, then redact (audit M3): nothing past the limit is ever scanned."""
    if not isinstance(text, str):
        return ""
    return redact(text[: max(0, int(max_chars))])


class TelemetryWriter:
    """Append-only JSONL decision log under ``<HERMES_HOME>/artifacts/jev_fastpath``.

    The file is bounded: when it exceeds ``max_bytes`` it rotates to a single
    ``decisions.jsonl.1`` backup, and restrictive permissions (0600) are applied where
    the platform supports them.
    """

    def __init__(self, home: Path, max_preview: int = 160, max_bytes: int = 5 * 1024 * 1024):
        self.path = Path(home) / "artifacts" / "jev_fastpath" / "decisions.jsonl"
        self.max_preview = int(max_preview)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()

    def write(self, event: TelemetryEvent, context: Mapping[str, Any]) -> None:
        """Serialize one fixed-shape record; never raises, never affects the turn."""
        try:
            self._write(event, context)
        except Exception:
            logger.debug("jev-fastpath telemetry write failed", exc_info=True)

    def _rotate_locked(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                backup = self.path.with_suffix(self.path.suffix + ".1")
                backup.unlink(missing_ok=True)
                self.path.rename(backup)
        except OSError:
            pass

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
        # Bound before any regex runs (audit M3): previews never scan unbounded text.
        preview = "" if event.outcome == "no_candidate" else redact(event.text[: self.max_preview])
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
            "text_length": len(event.text),
            "text_preview": preview,
            "candidates": list(event.candidates),
            "selected_handler": event.selected_handler,
            "confidence": event.confidence,
            "short_circuit_probability": event.short_circuit_probability,
            "latency_ms": event.latency_ms,
            "outcome": event.outcome,
            "reason": redact(event.reason[:160]),
            "provider_call_avoided": event.outcome == "short_circuit",
            "usage": usage,
            "answer_hash": digest(event.answer) if event.answer else "",
            "answer_preview": redact(event.answer[: self.max_preview]) if event.answer else "",
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            try:
                self.path.chmod(0o600)
            except OSError:
                pass  # best effort; not all filesystems support restrictive modes
            self._rotate_locked()
        except OSError as exc:
            raise RuntimeError("telemetry sink unavailable") from exc
