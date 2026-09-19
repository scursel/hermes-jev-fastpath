"""Allowlisted deterministic handlers: rendering, rejection, and status surfaces."""

from datetime import datetime

import pytest

from jev_fastpath.handlers import HandlerRejected, render_handler

CONTEXT = {
    "provider": "nous",
    "model": "hermes-4-405b",
    "api_mode": "chat_completions",
    "platform": "telegram",
    "session_id": "abc123",
    "mode": "active",
}


def _status(**overrides):
    status = {
        "mode": "active",
        "enabled_handlers": ("calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status"),
        "confidence_threshold": 0.92,
        "short_circuit_threshold": 0.9,
        "last_decision_at": "2026-09-19T12:00:00+00:00",
    }
    status.update(overrides)
    return status


class TestCalculator:
    def test_renders_expression(self, settings):
        result = render_handler("calculator", "quanto é 2 + 2?", CONTEXT, settings, _status())
        assert result.handler_id == "calculator"
        assert result.text == "2 + 2 = 4"

    def test_rejects_non_arithmetic_text(self, settings):
        with pytest.raises(HandlerRejected):
            render_handler("calculator", "calcule o desconto ideal", CONTEXT, settings, _status())


class TestClock:
    def _now_fn(self, expected_tz_name):
        def _now(tz):
            assert str(tz) == expected_tz_name
            return datetime(2026, 9, 19, 14, 30, tzinfo=tz)
        return _now

    def test_time_only(self, settings):
        result = render_handler(
            "clock", "que horas são?", CONTEXT, settings, _status(), now_fn=self._now_fn("America/Sao_Paulo")
        )
        assert result.text == "Agora são 14:30 (America/Sao_Paulo)."

    def test_date_only(self, settings):
        result = render_handler(
            "clock", "que dia é hoje", CONTEXT, settings, _status(), now_fn=self._now_fn("America/Sao_Paulo")
        )
        assert result.text == "Hoje é 19/09/2026 (America/Sao_Paulo)."

    def test_date_and_time(self):
        from jev_fastpath.config import Settings

        lisbon = Settings(timezone="Europe/Lisbon")
        result = render_handler(
            "clock", "que dia e horas são?", CONTEXT, lisbon, _status(), now_fn=self._now_fn("Europe/Lisbon")
        )
        assert result.text == "Agora são 14:30 de 19/09/2026 (Europe/Lisbon)."

    def test_english_locale(self):
        from jev_fastpath.config import Settings

        en_settings = Settings(locale="en")
        result = render_handler(
            "clock", "what time is it?", CONTEXT, en_settings, _status(), now_fn=self._now_fn("America/Sao_Paulo")
        )
        assert result.text == "It is 14:30 (America/Sao_Paulo)."

    def test_unknown_timezone_rejects(self):
        from jev_fastpath.config import Settings

        broken = Settings(timezone="Mars/Olympus_Mons")
        with pytest.raises(Exception):
            render_handler(
                "clock", "que horas são?", CONTEXT, broken, _status(), now_fn=self._now_fn("America/Sao_Paulo")
            )

    def test_context_cannot_inject_now_fn(self, settings):
        # Hardening: a hostile runtime-context key must not become a clock override.
        context = {**CONTEXT, "now_fn": lambda tz: datetime(1999, 1, 1, tzinfo=tz)}
        result = render_handler("clock", "que horas são?", context, settings, _status())
        assert "1999" not in result.text


class TestRuntimeIdentity:
    def test_answers_model_from_context_only(self, settings):
        result = render_handler("runtime_identity", "qual é o seu modelo?", CONTEXT, settings, _status())
        assert result.text == "Modelo: hermes-4-405b"

    def test_answers_provider_and_platform(self, settings):
        result = render_handler("runtime_identity", "qual provider você usa?", CONTEXT, settings, _status())
        assert result.text == "Provider: nous"
        result = render_handler("runtime_identity", "what platform is this?", CONTEXT, settings, _status())
        assert result.text == "Plataforma: telegram"

    def test_unproven_field_rejects(self, settings):
        context = {key: value for key, value in CONTEXT.items() if key != "platform"}
        with pytest.raises(HandlerRejected):
            render_handler("runtime_identity", "what platform is this?", context, settings, _status())

    def test_never_invents_context_fields(self, settings):
        with pytest.raises(HandlerRejected):
            render_handler("runtime_identity", "qual é o seu modelo?", {}, settings, _status())
        partial = {"model": "hermes-4-405b"}
        result = render_handler("runtime_identity", "qual é o seu modelo?", partial, settings, _status())
        assert result.text == "Modelo: hermes-4-405b"
        assert "nous" not in result.text and "telegram" not in result.text


class TestAcknowledgement:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("obrigado", "Por nada! Se precisar de outra coisa, é só falar."),
            ("Valeu!", "Por nada! Se precisar de outra coisa, é só falar."),
            ("thanks", "Por nada! Se precisar de outra coisa, é só falar."),
        ],
    )
    def test_gratitude_responses_are_fixed(self, settings, text, expected):
        result = render_handler("acknowledgement", text, CONTEXT, settings, _status())
        assert result.text == expected

    @pytest.mark.parametrize(
        "text",
        ["ok", "entendi", "got it", "certo", "beleza", "combinado", "perfeito",
         "sounds good", "roger that", "no problem", "legal", "show"],
    )
    def test_confirmation_words_reject(self, settings, text):
        # Gratitude-only fast path (audit decision 1): go-ahead/approval replies to an
        # assistant question or proposal must fall through to the real LLM.
        with pytest.raises(HandlerRejected):
            render_handler("acknowledgement", text, CONTEXT, settings, _status())

    def test_english_locale(self):
        from jev_fastpath.config import Settings

        en_settings = Settings(locale="en")
        result = render_handler("acknowledgement", "thanks", CONTEXT, en_settings, _status())
        assert result.text == "You're welcome! Let me know if you need anything else."


class TestFastpathStatus:
    def test_reports_mode_handlers_thresholds_and_last_decision(self, settings):
        result = render_handler("fastpath_status", "fastpath status", CONTEXT, settings, _status())
        assert "active" in result.text
        assert "calculator" in result.text and "fastpath_status" in result.text
        assert "0.92" in result.text and "0.9" in result.text
        assert "2026-09-19T12:00:00+00:00" in result.text

    def test_reports_never_when_no_decision_yet(self, settings):
        result = render_handler(
            "fastpath_status", "fastpath status", CONTEXT, settings, _status(last_decision_at=None)
        )
        assert "nunca" in result.text

    def test_english_locale(self):
        from jev_fastpath.config import Settings

        en_settings = Settings(locale="en")
        result = render_handler(
            "fastpath_status", "fastpath status", CONTEXT, en_settings, _status(last_decision_at=None)
        )
        assert "never" in result.text


class TestRegistryGates:
    def test_disabled_handler_rejected(self):
        from jev_fastpath.config import Settings

        settings = Settings(enabled_handlers=("calculator",))
        with pytest.raises(HandlerRejected):
            render_handler("clock", "que horas são?", CONTEXT, settings, _status())

    def test_unknown_handler_rejected(self, settings):
        with pytest.raises(HandlerRejected):
            render_handler("shell", "rm -rf /", CONTEXT, settings, _status())
        with pytest.raises(HandlerRejected):
            render_handler("normal_llm", "anything", CONTEXT, settings, _status())

    def test_empty_render_rejected(self, settings, monkeypatch):
        import jev_fastpath.handlers as handlers

        monkeypatch.setitem(handlers.HANDLERS, "calculator", lambda *a, **k: "   ")
        with pytest.raises(HandlerRejected):
            render_handler("calculator", "2 + 2", CONTEXT, settings, _status())

    def test_oversized_render_rejected(self, settings, monkeypatch):
        import jev_fastpath.handlers as handlers

        monkeypatch.setitem(handlers.HANDLERS, "calculator", lambda *a, **k: "x" * 2001)
        with pytest.raises(HandlerRejected):
            render_handler("calculator", "2 + 2", CONTEXT, settings, _status())

    def test_handler_error_propagates_as_rejection(self, settings, monkeypatch):
        import jev_fastpath.handlers as handlers

        def _boom(*_a, **_k):
            raise ZeroDivisionError("unexpected")

        monkeypatch.setitem(handlers.HANDLERS, "calculator", _boom)
        with pytest.raises(HandlerRejected):
            render_handler("calculator", "2 + 2", CONTEXT, settings, _status())
