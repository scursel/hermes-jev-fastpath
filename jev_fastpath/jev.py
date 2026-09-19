"""TypeSafe Jev typed-routing client and strict response parser.

One request, no retries, bounded timeout. Jev is a typed router only: its answer is
either an allowlisted handler ID plus two probabilities, or an error — never prose,
shell, URLs, tool arguments, or code.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Mapping

from .telemetry import redact_and_bound
from .types import Decision

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

HANDLER_CRITERIA: dict[str, str] = {
    "calculator": "The request is fully answered by the validated arithmetic parser.",
    "clock": "The request asks for the current date, the current time, or both in the configured timezone.",
    "runtime_identity": "The request asks which provider, model, API mode, or platform is serving this turn.",
    "acknowledgement": "The message is an exact allowlisted social acknowledgement with no other content.",
    "fastpath_status": "The request asks for this plugin's own mode, enabled handlers, thresholds, or last decision.",
}

_NO_LLM_CRITERIA = (
    "Probability that the chosen deterministic result fully satisfies this turn without tools or an LLM."
)


class JevError(RuntimeError):
    """Any TypeSafe transport, protocol, credential, or parsing failure; fails open upstream.

    Messages are fixed strings plus status codes/type names — never a response body, the
    authorization header, or a full exception chain. ``code`` is a bounded, non-secret
    reason code used by telemetry.
    """

    def __init__(self, message: str, code: str = "invalid_response") -> None:
        super().__init__(message)
        self.code = code


# Bounded reason codes surfaced in telemetry (audit L2): missing key, credential scope
# errors, timeout, network, HTTP/auth, redirect, oversized response, invalid payload,
# open circuit.
def _resolve_api_key() -> str:
    """Resolve ``TYPESAFE_API_KEY`` through the Hermes secret scope (multiplex-safe).

    When the Hermes ``agent.secret_scope`` module is importable it is authoritative: an
    empty or missing scoped value raises (fail open) instead of silently reading the
    ambient launch-profile environment, and ``UnscopedSecretError``/runtime failures are
    converted to :class:`JevError`. ``os.environ`` is only consulted when the Hermes
    module truly is unavailable (standalone test/package compatibility).
    """
    try:
        from agent import secret_scope
    except ImportError:
        value = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not value:
            raise JevError(
                "TYPESAFE_API_KEY is unavailable in the environment",
                code="missing_credential",
            )
        return value
    try:
        value = secret_scope.get_secret("TYPESAFE_API_KEY")
    except Exception as exc:
        raise JevError(
            "Hermes secret scope rejected the credential read", code="credential_error",
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise JevError(
            "TYPESAFE_API_KEY is unavailable in the Hermes secret scope",
            code="missing_credential",
        )
    return value.strip()


def build_questions(candidates: tuple[str, ...]) -> dict[str, Any]:
    """Typed question block: one allowlisted choice plus the short-circuit probability."""
    criteria = {handler_id: HANDLER_CRITERIA[handler_id] for handler_id in candidates}
    criteria["normal_llm"] = (
        "Any context, reasoning, tools, writing, interpretation, or unsupported data is needed."
    )
    return {
        "handler": {
            "type": "choice",
            "instructions": "Choose exactly one deterministic handler, otherwise normal_llm.",
            "criteria": criteria,
        },
        "safe_to_short_circuit": {
            "type": "noul",
            "instructions": _NO_LLM_CRITERIA,
            "criteria": {
                "true": "The chosen handler completely answers the literal request.",
                "false": "Any context, tool, judgment, ambiguity, side effect, or generative work is needed.",
            },
        },
    }


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError(f"{field} is not a numeric probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JevError(f"{field} is outside [0, 1]")
    return result


def parse_response(payload: object, candidates: tuple[str, ...], latency_ms: int = 0) -> Decision:
    """Strictly parse one typed Jev answer; anything malformed raises :class:`JevError`."""
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise JevError("response has no typed answers")
    answers = payload["answers"]
    handler = answers.get("handler")
    short = answers.get("safe_to_short_circuit")
    if not isinstance(handler, dict) or not isinstance(short, dict):
        raise JevError("response is missing handler or short-circuit answer")
    choice = handler.get("choice")
    allowed = set(candidates) | {"normal_llm"}
    if not isinstance(choice, str) or choice not in allowed:
        raise JevError("handler choice is not allowed for this request")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    return Decision(
        handler_id=choice,
        confidence=_probability(handler.get("confidence"), "handler.confidence"),
        short_circuit_probability=_probability(short.get("noul"), "safe_to_short_circuit.noul"),
        latency_ms=int(latency_ms),
        usage=usage,
        model=model,
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject every redirect: the typed POST is never replayed elsewhere (audit L1)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(newurl, code, msg, headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)
# ``OpenerDirector`` is not callable: every call site passes ``_OPENER.open`` (audit C1).

# TypeSafe answers a bounded typed decision; 64 KiB is generous and caps read memory.
MAX_RESPONSE_BYTES = 64 * 1024


class CircuitBreaker:
    """Thread-safe failure breaker: open after N consecutive failures, probe after cooldown.

    Bounds the blast radius of a TypeSafe outage: an open circuit answers immediately
    (fail open to the real LLM) instead of paying the full deadline on every candidate
    turn. State is per-process and deliberately tiny.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0,
                 monotonic=time.monotonic):
        self._threshold = int(failure_threshold)
        self._cooldown = float(cooldown_seconds)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        now = self._monotonic()
        with self._lock:
            if self._opened_at is None:
                return False
            if now - self._opened_at >= self._cooldown:
                # Half-open: allow one probe attempt again.
                self._opened_at = None
                self._consecutive_failures = self._threshold - 1
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        now = self._monotonic()
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._opened_at = now


BREAKER = CircuitBreaker()

# Bounded shared pool: the whole HTTP exchange runs on one of two workers with a socket
# timeout, so a stalled call can never accumulate threads (audit M1).
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jev-fastpath")


def _exchange(request: urllib.request.Request, opener, timeout_seconds: float) -> bytes:
    """One capped HTTP exchange; converts every transport failure into a coded JevError.

    ``opener`` is a callable ``(request, timeout=...)`` returning a context-managed
    response; :func:`classify` passes the hardened ``_OPENER.open`` by default.
    """
    try:
        with opener(request, timeout=timeout_seconds) as response:
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise JevError(
                        "TypeSafe response exceeds the 64 KiB response cap",
                        code="response_too_large",
                    )
                chunks.append(chunk)
            raw = b"".join(chunks)
        return raw.decode("utf-8")
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise JevError(f"TypeSafe redirect {exc.code} rejected", code="redirect") from exc
        code = "auth_error" if exc.code in (401, 403) else "http_error"
        raise JevError(f"TypeSafe HTTP {exc.code}", code=code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JevError(f"TypeSafe network failure: {type(exc).__name__}", code="network_error") from exc
    except JevError:
        raise
    except OSError as exc:
        # Socket/http.client read failures; ValueError (unicode) handled by the caller.
        raise JevError(f"TypeSafe transport failure: {type(exc).__name__}", code="network_error") from exc


def classify(
    text: str,
    candidates: tuple[str, ...],
    context: Mapping[str, Any],
    settings,
    *,
    opener=None,
    breaker: CircuitBreaker | None = None,
) -> Decision:
    """Ask Jev for one typed routing decision; exactly one network attempt, no retries.

    The exchange runs under a hard wall-clock deadline (:attr:`Settings.timeout_seconds`)
    with a capped response read and a failure circuit breaker, so a TypeSafe outage costs
    every candidate turn at most one bounded deadline before failing open.
    """
    breaker = breaker or BREAKER
    if breaker.is_open():
        raise JevError("TypeSafe circuit breaker is open", code="circuit_open")
    api_key = _resolve_api_key()
    payload_body = build_payload(text, candidates, context, settings)
    request = urllib.request.Request(
        TYPESAFE_URL,
        data=payload_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-jev-fastpath/0.1",
        },
    )
    started = time.monotonic()
    try:
        raw = _EXECUTOR.submit(
            _exchange, request, opener or _OPENER.open, settings.timeout_seconds,
        ).result(timeout=settings.timeout_seconds)
    except FuturesTimeoutError as exc:
        breaker.record_failure()
        raise JevError("TypeSafe deadline exceeded", code="timeout") from exc
    except JevError as exc:
        breaker.record_failure()
        raise
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        breaker.record_failure()
        raise JevError("TypeSafe returned invalid JSON", code="invalid_response") from exc
    try:
        decision = parse_response(payload, candidates, latency_ms)
    except JevError:
        breaker.record_failure()
        raise
    breaker.record_success()
    return decision


def build_payload(
    text: str,
    candidates: tuple[str, ...],
    context: Mapping[str, Any],
    settings,
) -> bytes:
    """Serialize the exact TypeSafe request: redacted bounded state plus typed questions."""
    body = {
        "model": JEV_MODEL,
        "state": {
            "message": redact_and_bound(text, settings.max_input_chars),
            "platform": str(context.get("platform") or ""),
            "candidate_handlers": list(candidates),
        },
        "questions": build_questions(candidates),
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")
