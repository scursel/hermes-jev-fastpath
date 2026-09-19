"""llm_execution middleware orchestration and the Hermes ``register(ctx)`` entrypoint.

Every uncertainty or failure calls the wrapped provider exactly once; an accepted active
fast path returns the raw synthetic response and never calls ``next_call``. Prior
messages, system prompt, tools, and history are read-only here.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .cache import DecisionCache
from .candidates import detect_candidates
from .config import Settings, SettingsError, load_settings
from .handlers import render_handler
from .input import extract_latest_user_text
from .jev import classify
from .responses import build_synthetic_response
from .telemetry import TelemetryWriter
from .types import TelemetryEvent

logger = logging.getLogger("jev_fastpath.plugin")


class FastPathRuntime:
    """One profile-scoped fast-path runtime bound to a validated :class:`Settings`.

    Classifier, renderer, response factory, cache, and telemetry are injectable so tests
    can replace them without network or filesystem access.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        classifier: Callable = classify,
        renderer: Callable = render_handler,
        response_factory: Callable = build_synthetic_response,
        telemetry: Any = None,
        cache: DecisionCache | None = None,
    ):
        self.settings = settings
        self.classifier = classifier
        self.renderer = renderer
        self.response_factory = response_factory
        self.telemetry = telemetry
        self.cache = cache or DecisionCache(settings.cache_ttl_seconds)
        self._status_lock = threading.Lock()
        self._last_decision_at: str | None = None

    def status(self) -> dict[str, Any]:
        """Bounded status surface used by the ``fastpath_status`` handler."""
        with self._status_lock:
            return {
                "mode": self.settings.mode,
                "enabled_handlers": self.settings.enabled_handlers,
                "confidence_threshold": self.settings.confidence_threshold,
                "short_circuit_threshold": self.settings.short_circuit_threshold,
                "last_decision_at": self._last_decision_at,
            }

    def _emit(self, event: TelemetryEvent, meta: Mapping[str, Any]) -> None:
        if self.telemetry is not None:
            self.telemetry.write(event, meta)

    def middleware(self, *, request, next_call: Callable, api_call_count: int = 0,
                   session_id: str = "", turn_id: str = "", api_mode: str = "", **context: Any):
        """``llm_execution`` middleware callback; accepts ``**kwargs`` for forward compatibility."""
        downstream_called = False

        def downstream():
            nonlocal downstream_called
            if downstream_called:
                raise RuntimeError("downstream provider call attempted more than once")
            downstream_called = True
            return next_call(request)

        meta = {
            **context,
            "session_id": str(session_id),
            "turn_id": str(turn_id),
            "api_mode": str(api_mode),
            "mode": self.settings.mode,
        }
        try:
            return self._evaluate_or_fallthrough(
                request=request,
                downstream=downstream,
                api_call_count=api_call_count,
                session_id=str(session_id),
                turn_id=str(turn_id),
                api_mode=str(api_mode),
                context=meta,
            )
        except Exception as exc:
            if downstream_called:
                # The real provider already ran and raised; never turn that into a second call.
                raise
            self._emit(TelemetryEvent(
                outcome="fallback",
                reason=type(exc).__name__,
                text="",
                candidates=(),
            ), meta)
            return downstream()

    def _evaluate_or_fallthrough(self, *, request, downstream: Callable, api_call_count,
                                 session_id: str, turn_id: str, api_mode: str,
                                 context: Mapping[str, Any]):
        if self.settings.mode == "off" or int(api_call_count or 0) != 0:
            return downstream()
        text = extract_latest_user_text(request, api_mode)
        if text is None:
            return downstream()
        candidates = detect_candidates(text, self.settings)
        if not candidates:
            self._emit(TelemetryEvent(
                outcome="no_candidate", reason="local_filter", text=text, candidates=(),
            ), context)
            return downstream()

        key = self.cache.key(session_id, turn_id, text)
        decision = self.cache.get(key) if session_id and turn_id else None
        if decision is None:
            decision = self.classifier(text, candidates, context, self.settings)
            if session_id and turn_id:
                self.cache.put(key, decision)
        with self._status_lock:
            self._last_decision_at = datetime.now(timezone.utc).isoformat()

        rejected = (
            decision.handler_id == "normal_llm"
            or decision.handler_id not in candidates
            or decision.confidence < self.settings.confidence_threshold
            or decision.short_circuit_probability < self.settings.short_circuit_threshold
        )
        if rejected:
            self._emit(TelemetryEvent(
                outcome="normal_llm", reason="jev_rejected", text=text,
                candidates=candidates, selected_handler=decision.handler_id,
                confidence=decision.confidence,
                short_circuit_probability=decision.short_circuit_probability,
                latency_ms=decision.latency_ms, usage=decision.usage,
            ), context)
            return downstream()

        rendered = self.renderer(decision.handler_id, text, context, self.settings, self.status())
        if self.settings.mode == "shadow":
            self._emit(TelemetryEvent(
                outcome="would_short_circuit", reason="shadow", text=text,
                candidates=candidates, selected_handler=decision.handler_id,
                confidence=decision.confidence,
                short_circuit_probability=decision.short_circuit_probability,
                latency_ms=decision.latency_ms, usage=decision.usage,
            ), context)
            return downstream()

        response = self.response_factory(api_mode, rendered.text)
        self._emit(TelemetryEvent(
            outcome="short_circuit", reason="accepted", text=text,
            candidates=candidates, selected_handler=decision.handler_id,
            confidence=decision.confidence,
            short_circuit_probability=decision.short_circuit_probability,
            latency_ms=decision.latency_ms, usage=decision.usage,
        ), context)
        return response


def register(ctx) -> None:
    """Hermes directory-plugin entrypoint: validate settings, wire one middleware callback.

    Invalid settings emit exactly one bounded warning and disable the plugin; discovery
    never crashes on them. ``get_hermes_home`` is imported from Hermes only here so the
    package stays importable (and testable) without the Hermes source tree.
    """
    try:
        settings = load_settings(ctx)
    except SettingsError as exc:
        logger.warning("jev-fastpath disabled: %s", exc)
        return
    try:
        from hermes_constants import get_hermes_home
    except Exception as exc:  # pragma: no cover - Hermes always provides this at runtime
        logger.warning("jev-fastpath disabled: Hermes home unavailable (%s)", type(exc).__name__)
        return
    runtime = FastPathRuntime(settings, telemetry=TelemetryWriter(get_hermes_home()))
    ctx.register_middleware("llm_execution", runtime.middleware)
    logger.info(
        "jev-fastpath registered llm_execution middleware (mode=%s, handlers=%s)",
        settings.mode, ",".join(settings.enabled_handlers),
    )
