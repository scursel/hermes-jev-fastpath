"""Typed settings parsing: defaults, strict validation, no coercion."""

import math

import pytest

from jev_fastpath.config import Settings, SettingsError, load_settings


class FakeContext:
    def __init__(self, values=None):
        self.values = values or {}

    def get_config(self, key, default=None):
        return self.values.get(key, default)


def test_defaults_are_shadow_and_profile_safe():
    settings = load_settings(FakeContext())
    assert settings.mode == "shadow"
    assert settings.timezone == "America/Sao_Paulo"
    assert settings.confidence_threshold == 0.92
    assert settings.short_circuit_threshold == 0.90
    assert settings.timeout_seconds == 3.0
    assert settings.locale == "pt-BR"
    assert settings.max_input_chars == 4000
    assert settings.cache_ttl_seconds == 900.0
    assert "calculator" in settings.enabled_handlers
    assert settings.enabled_handlers == (
        "calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status",
    )


@pytest.mark.parametrize("value", ["enabled", "yes", 1, None, "", "OFF", "Active"])
def test_invalid_mode_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"mode": value}))


@pytest.mark.parametrize("value", ["off", "shadow", "active"])
def test_valid_mode_accepted(value):
    assert load_settings(FakeContext({"mode": value})).mode == value


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, "0.9", True, False, None])
def test_invalid_confidence_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"confidence_threshold": value}))


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_valid_confidence_accepted(value):
    assert load_settings(FakeContext({"confidence_threshold": value})).confidence_threshold == value


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, -math.inf, "0.8", True, None])
def test_invalid_short_circuit_threshold_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"short_circuit_threshold": value}))


@pytest.mark.parametrize("value", [0.19, 10.01, math.nan, math.inf, "3", True, None])
def test_invalid_timeout_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"timeout_seconds": value}))


@pytest.mark.parametrize("value", [0.2, 3.0, 10.0])
def test_valid_timeout_accepted(value):
    assert load_settings(FakeContext({"timeout_seconds": value})).timeout_seconds == value


def test_unknown_timezone_rejected():
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"timezone": "Mars/Olympus_Mons"}))


def test_blank_and_non_string_timezone_rejected():
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"timezone": "   "}))
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"timezone": 3}))


def test_valid_timezone_accepted():
    assert load_settings(FakeContext({"timezone": "Europe/Lisbon"})).timezone == "Europe/Lisbon"


@pytest.mark.parametrize("value", ["pt", "en-US", "PT-BR", "", None, 1])
def test_invalid_locale_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"locale": value}))


@pytest.mark.parametrize("value", [31, 12001, "4000", 4000.0, True, None])
def test_invalid_max_input_chars_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"max_input_chars": value}))


@pytest.mark.parametrize("value", [32, 4000, 12000])
def test_valid_max_input_chars_accepted(value):
    assert load_settings(FakeContext({"max_input_chars": value})).max_input_chars == value


def test_unknown_and_duplicate_handlers_rejected():
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": ["clock", "clock"]}))
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": ["shell"]}))
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": ["calculator", "normal_llm"]}))


def test_non_list_handlers_rejected():
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": "calculator"}))
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": [1]}))


def test_empty_handlers_allowed():
    settings = load_settings(FakeContext({"enabled_handlers": []}))
    assert settings.enabled_handlers == ()


def test_handlers_returned_in_declared_order():
    settings = load_settings(FakeContext({"enabled_handlers": ["clock", "calculator"]}))
    assert settings.enabled_handlers == ("clock", "calculator")


def test_invalid_cache_ttl_rejected():
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"cache_ttl_seconds": 0.5}))
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"cache_ttl_seconds": 3601}))


def test_settings_are_frozen():
    settings = load_settings(FakeContext())
    with pytest.raises(Exception):
        settings.mode = "active"
