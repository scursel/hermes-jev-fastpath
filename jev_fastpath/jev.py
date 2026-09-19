"""TypeSafe Jev typed-routing client and strict response parser.

One request, no retries, bounded timeout. Jev is a typed router only: its answer is
either an allowlisted handler ID plus two probabilities, or an error — never prose,
shell, URLs, tool arguments, or code.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
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
    """Any TypeSafe transport, protocol, or parsing failure; always fails open upstream.

    Messages are fixed strings plus status codes/type names — never a response body, the
    authorization header, or a full exception chain.
    """


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


def classify(
    text: str,
    candidates: tuple[str, ...],
    context: Mapping[str, Any],
    settings,
    *,
    opener=urllib.request.urlopen,
) -> Decision:
    """Ask Jev for one typed routing decision; exactly one network attempt, no retries."""
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise JevError("TYPESAFE_API_KEY is unavailable")
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
        with opener(request, timeout=settings.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise JevError(f"TypeSafe HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JevError(f"TypeSafe network failure: {type(exc).__name__}") from exc
    except (OSError, ValueError) as exc:
        # OSError covers socket/http.client read failures; ValueError covers unicode decoding.
        raise JevError(f"TypeSafe transport failure: {type(exc).__name__}") from exc
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JevError("TypeSafe returned invalid JSON") from exc
    return parse_response(payload, candidates, latency_ms)


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
