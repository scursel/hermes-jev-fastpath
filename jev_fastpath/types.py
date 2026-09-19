"""Typed domain objects shared by every layer of the fast path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

HandlerId = Literal[
    "calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status"
]
Outcome = Literal[
    "no_candidate", "normal_llm", "would_short_circuit", "short_circuit", "fallback"
]


@dataclass(frozen=True)
class Decision:
    """Jev's typed route decision for one candidate set."""

    handler_id: str
    confidence: float
    short_circuit_probability: float
    latency_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


@dataclass(frozen=True)
class HandlerResult:
    """Locally rendered deterministic answer."""

    handler_id: str
    text: str


@dataclass(frozen=True)
class TelemetryEvent:
    """One bounded, redacted audit record."""

    outcome: Outcome
    reason: str
    text: str
    candidates: tuple[str, ...]
    selected_handler: str = ""
    confidence: float | None = None
    short_circuit_probability: float | None = None
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
