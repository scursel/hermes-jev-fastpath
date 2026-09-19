"""Allowlisted deterministic handler registry.

No dynamic imports: ``HANDLERS`` is a fixed mapping of module-local functions. Request
shape classifiers live here so candidate detection and rendering can never drift apart.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .arithmetic import render_calculation
from .config import Settings
from .types import HandlerResult

Handler = Callable[[str, Mapping[str, Any], Settings, Mapping[str, Any]], str]

MAX_RENDER_CHARS = 2000


class HandlerRejected(ValueError):
    """The handler is disabled, unknown, or produced no bounded answer."""


# ---- classification primitives (shared with candidate detection) ----

_CLOCK_PATTERNS = (
    re.compile(r"(?i)^\s*que\s+horas?\s+(?:s[aã]o|e)\s*(?:agora)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*(?:qual|que)\s+(?:e|é)\s+(?:a\s+)?hora(?:\s+atual|\s+agora|\s+de\s+agora)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*que\s+dia\s+e\s+horas?\s+(?:s[aã]o|e)\s*\??\s*$"),
    re.compile(r"(?i)^\s*(?:qual|que)\s+(?:e|é)\s+(?:o\s+dia\s+de\s+hoje|a\s+data(?:\s+de\s+hoje|\s+atual)?)\s*\??\s*$"),
    re.compile(r"(?i)^\s*data\s+de\s+hoje\s*\??\s*$"),
    re.compile(r"(?i)^\s*que\s+dia\s+[eé]\s+hoje\s*\??\s*$"),
    re.compile(r"(?i)^\s*hora\s+e\s+data(?:\s+atual)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*data\s+e\s+hora(?:\s+atual(?:s)?)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*what\s+time\s+is\s+it\s*(?:now|right\s+now)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*what(?:'s|is)?\s+the\s+(?:time|date)(?:\s+now|\s+today)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*(?:current|local|today's)\s+(?:time|date)\s*\??\s*$"),
    re.compile(r"(?i)^\s*(?:date|time)\s+and\s+(?:time|date)\s*(?:now|today)?\s*\??\s*$"),
    re.compile(r"(?i)^\s*what\s+is\s+today(?:'s\s+date)?\s*\??\s*$"),
)
_TIME_ONLY = re.compile(r"(?i)hora|time")
_DATE_ONLY = re.compile(r"(?i)dia|date|data|today")

_IDENTITY_FIELDS = ("provider", "model", "api_mode", "platform")
_IDENTITY_PATTERNS = {
    "provider": (
        re.compile(r"(?i)^\s*(?:qual|que|which|what)(?:\s+(?:e|é|is|'s))?\s+(?:o\s+|a\s+|seu\s+|teu\s+|your\s+|the\s+)?"
                   r"(?:provider|provedor|fornecedor)\b(?:\s+voc[êe]?\s+(?:usa|usando|est[áa]\s+usando))?\s*\??\s*$"),
    ),
    "model": (
        re.compile(r"(?i)^\s*(?:qual|que|which|what)\s+(?:e|é|is|'s)?\s*(?:o\s+|a\s+)?(?:seu\s+|teu\s+|your\s+|the\s+)?"
                   r"modelo\s*(?:e|é|de\s+voc[êe]|voc[êe]\s+[ée])?\s*\??\s*$"),
        re.compile(r"(?i)^\s*what\s+model\s+are\s+you\s*\??\s*$"),
        re.compile(r"(?i)^\s*which\s+model\s+are\s+you\s*(?:using|running)?\s*\??\s*$"),
    ),
    "api_mode": (
        re.compile(r"(?i)^\s*(?:qual|que|which|what)(?:\s+(?:e|é|is|'s))?\s+(?:o\s+|a\s+|seu\s+|your\s+|the\s+)?"
                   r"(?:modo\s+de\s+api|modo\s+api|api\s*mode|api\s+protocol)\s*\??\s*$"),
    ),
    "platform": (
        re.compile(r"(?i)^\s*(?:qual|que|which|what)(?:\s+(?:e|é|is|'s))?\s+(?:a\s+|o\s+|seu\s+|your\s+|this\s+|the\s+)?"
                   r"plataforma\b\s*\??\s*$"),
        re.compile(r"(?i)^\s*what\s+platform\s+is\s+(?:this|it)\s*\??\s*$"),
    ),
}

# Gratitude only (audit decision 1): go-ahead/approval words ("ok", "certo", "beleza",
# "combinado", "perfeito", "sounds good", "roger that", …) are context-dependent replies
# to an assistant question or proposal and must fall through to the real LLM.
_ACK_GRATITUDE = frozenset({
    "obrigado", "obrigada", "obrigadao", "brigado", "brigada", "valeu", "vlw",
    "thanks", "thank", "thank you", "thankyou", "thx", "ty", "tks",
})
_TRAILING_PUNCTUATION = "!!.,;:...)\"'"


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    flattened = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    flattened = flattened.rstrip(_TRAILING_PUNCTUATION)
    return re.sub(r"\s+", " ", flattened).strip()


def classify_clock_request(text: str) -> str | None:
    """``time``, ``date``, ``both`` for a direct current date/time request, else ``None``."""
    if not any(pattern.match(text) for pattern in _CLOCK_PATTERNS):
        return None
    wants_time = bool(_TIME_ONLY.search(text))
    wants_date = bool(_DATE_ONLY.search(text))
    if wants_time and wants_date:
        return "both"
    if wants_time:
        return "time"
    if wants_date:
        return "date"
    return "both"


def classify_identity_fields(text: str) -> tuple[str, ...]:
    """Context fields directly asked about, in fixed order."""
    fields = tuple(
        field for field in _IDENTITY_FIELDS
        if any(pattern.match(text) for pattern in _IDENTITY_PATTERNS[field])
    )
    return fields


def classify_acknowledgement(text: str) -> str | None:
    """``gratitude`` for an exact allowlisted gratitude, else ``None``.

    Confirmation/go-ahead words are deliberately excluded: as replies to an assistant
    question or proposal they are context-dependent and must reach the real LLM.
    """
    normalized = _normalize(text)
    if not normalized:
        return None
    if normalized in _ACK_GRATITUDE:
        return "gratitude"
    return None


_STATUS_PATTERNS = (
    re.compile(r"(?i)^\s*(?:qual|que)\s+(?:e|é)\s+o\s+status\s+do\s+(?:jev\s+)?fast\s?-?\s?path\s*\??\s*$"),
    re.compile(r"(?i)^\s*(?:qual|que)\s+(?:e|é)\s+o?\s*status\s+do\s+plugin\s*\??\s*$"),
    re.compile(r"(?i)^\s*what(?:'s| is)?\s+the\s+(?:jev\s+)?fast\s?-?\s?path\s+status\s*\??\s*$"),
    re.compile(r"(?i)^\s*(?:jev\s+)?fast\s?-?\s?path\s+status\s*\??\s*$"),
    re.compile(r"(?i)^\s*status\s+do\s+(?:jev\s+)?fast\s?-?\s?path\s*\??\s*$"),
    re.compile(r"(?i)^\s*status\s+of\s+the\s+(?:jev\s+)?fast\s?-?\s?path\s*\??\s*$"),
)


def classify_status_request(text: str) -> bool:
    """True for a direct request about this plugin's own status."""
    return any(pattern.match(text) for pattern in _STATUS_PATTERNS)


# ---- deterministic renderers ----

_ACK_TEXT = {
    ("gratitude", "pt-BR"): "Por nada! Se precisar de outra coisa, é só falar.",
    ("gratitude", "en"): "You're welcome! Let me know if you need anything else.",
}


def _calculator(text: str, context: Mapping[str, Any], settings: Settings, status: Mapping[str, Any],
           *, now_fn=None) -> str:
    return render_calculation(text)


def _clock(text: str, context: Mapping[str, Any], settings: Settings, status: Mapping[str, Any],
           *, now_fn: Callable | None = None) -> str:
    shape = classify_clock_request(text)
    if shape is None:
        raise HandlerRejected("not a direct clock request")
    try:
        tz = ZoneInfo(settings.timezone)
        now = now_fn(tz) if callable(now_fn) else datetime.now(tz)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise HandlerRejected("configured timezone is unavailable") from exc
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise HandlerRejected("clock did not produce an aware datetime")
    tz_name = str(tz)
    if settings.locale == "en":
        if shape == "time":
            return f"It is {now:%H:%M} ({tz_name})."
        if shape == "date":
            return f"Today is {now:%Y-%m-%d} ({tz_name})."
        return f"It is {now:%H:%M} on {now:%Y-%m-%d} ({tz_name})."
    if shape == "time":
        return f"Agora são {now:%H:%M} ({tz_name})."
    if shape == "date":
        return f"Hoje é {now:%d/%m/%Y} ({tz_name})."
    return f"Agora são {now:%H:%M} de {now:%d/%m/%Y} ({tz_name})."


_IDENTITY_LABELS = {
    ("provider", "pt-BR"): "Provider",
    ("model", "pt-BR"): "Modelo",
    ("api_mode", "pt-BR"): "API mode",
    ("platform", "pt-BR"): "Plataforma",
    ("provider", "en"): "Provider",
    ("model", "en"): "Model",
    ("api_mode", "en"): "API mode",
    ("platform", "en"): "Platform",
}


def _runtime_identity(text: str, context: Mapping[str, Any], settings: Settings, status: Mapping[str, Any],
           *, now_fn=None) -> str:
    fields = classify_identity_fields(text)
    if not fields:
        raise HandlerRejected("not a direct identity question")
    parts = []
    for field in fields:
        value = context.get(field)
        if not isinstance(value, str) or not value.strip():
            # The middleware context does not prove this field, so it must not be claimed.
            raise HandlerRejected(f"context does not prove field: {field}")
        label = _IDENTITY_LABELS[(field, settings.locale)]
        parts.append(f"{label}: {value.strip()}")
    return ", ".join(parts)


def _acknowledgement(text: str, context: Mapping[str, Any], settings: Settings, status: Mapping[str, Any],
           *, now_fn=None) -> str:
    kind = classify_acknowledgement(text)
    if kind is None:
        raise HandlerRejected("not an exact allowlisted acknowledgement")
    return _ACK_TEXT[(kind, settings.locale)]


def _fastpath_status(text: str, context: Mapping[str, Any], settings: Settings, status: Mapping[str, Any],
           *, now_fn=None) -> str:
    if not classify_status_request(text):
        raise HandlerRejected("not a direct status request")
    handlers_list = ", ".join(status.get("enabled_handlers", ()))
    last = status.get("last_decision_at")
    if settings.locale == "en":
        last_text = str(last) if last else "never"
        return (
            f"Jev fast path: mode {status.get('mode', '?')}, handlers [{handlers_list}], "
            f"thresholds confidence {status.get('confidence_threshold', '?')} / short-circuit "
            f"{status.get('short_circuit_threshold', '?')}, last decision {last_text}."
        )
    last_text = str(last) if last else "nunca"
    return (
        f"Jev fast path: modo {status.get('mode', '?')}, handlers [{handlers_list}], "
        f"limiares confiança {status.get('confidence_threshold', '?')} / short-circuit "
        f"{status.get('short_circuit_threshold', '?')}, última decisão {last_text}."
    )


HANDLERS: dict[str, Handler] = {
    "calculator": _calculator,
    "clock": _clock,
    "runtime_identity": _runtime_identity,
    "acknowledgement": _acknowledgement,
    "fastpath_status": _fastpath_status,
}


def render_handler(
    handler_id: str,
    text: str,
    context: Mapping[str, Any],
    settings: Settings,
    status: Mapping[str, Any],
    *,
    now_fn: Callable | None = None,
) -> HandlerResult:
    """Validate the handler choice against the allowlist and render a bounded local answer.

    ``now_fn`` is the explicit internal clock seam for deterministic tests; Hermes runtime
    context can never inject it.
    """
    if handler_id not in settings.enabled_handlers or handler_id not in HANDLERS:
        raise HandlerRejected(f"handler not enabled: {handler_id}")
    try:
        rendered = HANDLERS[handler_id](text, context, settings, status, now_fn=now_fn).strip()
    except HandlerRejected:
        raise
    except Exception as exc:
        raise HandlerRejected(f"handler failed: {type(exc).__name__}") from exc
    if not rendered or len(rendered) > MAX_RENDER_CHARS:
        raise HandlerRejected("handler returned invalid text")
    return HandlerResult(handler_id=handler_id, text=rendered)
