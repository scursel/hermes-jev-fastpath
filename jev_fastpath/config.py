"""Strict plugin settings: typed defaults, validation without coercion.

Invalid settings disable the fast path at registration time (``register`` logs one
bounded warning and returns); they never crash plugin discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

KNOWN_HANDLERS = (
    "calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status",
)

LOCALES = ("pt-BR", "en")
MODES = ("off", "shadow", "active")


class SettingsError(ValueError):
    """Raised when profile settings cannot be trusted; the plugin stays disabled."""


@dataclass(frozen=True)
class Settings:
    mode: str = "shadow"
    confidence_threshold: float = 0.92
    short_circuit_threshold: float = 0.90
    timeout_seconds: float = 3.0
    timezone: str = "America/Sao_Paulo"
    locale: str = "pt-BR"
    max_input_chars: int = 4000
    enabled_handlers: tuple[str, ...] = KNOWN_HANDLERS
    cache_ttl_seconds: float = 900.0


def _float_setting(name: str, value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result) or not low <= result <= high:
        raise SettingsError(f"{name} must be between {low} and {high}")
    return result


def _int_setting(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise SettingsError(f"{name} must be an integer between {low} and {high}")
    return value


def load_settings(ctx) -> Settings:
    """Read every setting through ``ctx.get_config(key, default)`` and validate strictly.

    No string-to-number or bool-to-int coercion happens here: a wrongly typed profile
    value is a configuration error, not something to silently repair.
    """
    defaults = Settings()
    mode = ctx.get_config("mode", defaults.mode)
    locale = ctx.get_config("locale", defaults.locale)
    timezone = ctx.get_config("timezone", defaults.timezone)
    handlers = ctx.get_config("enabled_handlers", list(defaults.enabled_handlers))
    if mode not in MODES:
        raise SettingsError("mode must be off, shadow, or active")
    if locale not in LOCALES:
        raise SettingsError("locale must be pt-BR or en")
    if not isinstance(timezone, str) or not timezone.strip():
        raise SettingsError("timezone must be a non-empty IANA name")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise SettingsError("timezone is not available") from exc
    if not isinstance(handlers, (list, tuple)) or any(not isinstance(x, str) for x in handlers):
        raise SettingsError("enabled_handlers must be a list of handler IDs")
    if len(set(handlers)) != len(handlers) or any(x not in KNOWN_HANDLERS for x in handlers):
        raise SettingsError("enabled_handlers contains a duplicate or unknown ID")
    return Settings(
        mode=mode,
        confidence_threshold=_float_setting(
            "confidence_threshold",
            ctx.get_config("confidence_threshold", defaults.confidence_threshold), 0.0, 1.0,
        ),
        short_circuit_threshold=_float_setting(
            "short_circuit_threshold",
            ctx.get_config("short_circuit_threshold", defaults.short_circuit_threshold), 0.0, 1.0,
        ),
        timeout_seconds=_float_setting(
            "timeout_seconds", ctx.get_config("timeout_seconds", defaults.timeout_seconds), 0.2, 10.0,
        ),
        timezone=timezone,
        locale=locale,
        max_input_chars=_int_setting(
            "max_input_chars", ctx.get_config("max_input_chars", defaults.max_input_chars), 32, 12000,
        ),
        enabled_handlers=tuple(handlers),
        cache_ttl_seconds=_float_setting(
            "cache_ttl_seconds",
            ctx.get_config("cache_ttl_seconds", defaults.cache_ttl_seconds), 1.0, 3600.0,
        ),
    )
