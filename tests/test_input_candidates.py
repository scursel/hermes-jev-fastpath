"""User-text extraction from provider requests and conservative candidate detection."""

import copy

import pytest

from jev_fastpath.candidates import detect_candidates
from jev_fastpath.config import Settings
from jev_fastpath.input import extract_latest_user_text


def _chat_request(text):
    return {"model": "x", "messages": [{"role": "user", "content": text}]}


def _codex_request(text):
    return {
        "model": "x",
        "instructions": "sys",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
    }


def _anthropic_request(text):
    return {"model": "x", "system": "sys", "messages": [{"role": "user", "content": text}]}


def _bedrock_request(text):
    return {
        "modelId": "x",
        "messages": [{"role": "user", "content": [{"text": text}]}],
    }


class TestExtractLatestUserText:
    def test_chat_completions_plain_string(self):
        assert extract_latest_user_text(_chat_request("2 + 2"), "chat_completions") == "2 + 2"

    def test_chat_completions_text_parts(self):
        request = {
            "messages": [
                {"role": "user", "content": "older"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": [{"type": "text", "text": "2 + "}, {"type": "text", "text": "2"}]},
            ]
        }
        assert extract_latest_user_text(request, "chat_completions") == "2 + \n2"

    def test_chat_completions_multimodal_rejected(self):
        request = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]}]
        }
        assert extract_latest_user_text(request, "chat_completions") is None

    def test_chat_completions_skips_tool_results(self):
        request = {
            "messages": [
                {"role": "user", "content": "run it"},
                {"role": "tool", "content": "tool output"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        assert extract_latest_user_text(request, "chat_completions") == "run it"

    def test_codex_responses_plain_string_input(self):
        assert extract_latest_user_text({"input": "2 + 2"}, "codex_responses") == "2 + 2"

    def test_codex_responses_message_item(self):
        assert extract_latest_user_text(_codex_request("2 + 2"), "codex_responses") == "2 + 2"

    def test_codex_responses_skips_function_outputs(self):
        request = {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "run it"}]},
                {"type": "function_call_output", "call_id": "c1", "output": "result"},
            ]
        }
        assert extract_latest_user_text(request, "codex_responses") == "run it"

    def test_anthropic_messages(self):
        assert extract_latest_user_text(_anthropic_request("2 + 2"), "anthropic_messages") == "2 + 2"

    def test_anthropic_multimodal_rejected(self):
        request = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
            ]}]
        }
        assert extract_latest_user_text(request, "anthropic_messages") is None

    def test_bedrock_converse(self):
        assert extract_latest_user_text(_bedrock_request("2 + 2"), "bedrock_converse") == "2 + 2"

    def test_unknown_api_mode_returns_none(self):
        assert extract_latest_user_text(_chat_request("2 + 2"), "future_protocol") is None

    @pytest.mark.parametrize(
        "payload",
        [None, 42, "text", {}, {"messages": "nope"}, {"messages": [None]}, {"messages": [{"role": "user"}]}],
    )
    def test_malformed_requests_return_none(self, payload):
        assert extract_latest_user_text(payload, "chat_completions") is None

    def test_empty_text_returned_and_filtered_downstream(self):
        assert extract_latest_user_text(_chat_request("   "), "chat_completions") == "   "

    def test_request_is_never_mutated(self):
        request = _chat_request("  2 + 2  ")
        request["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "x"}]})
        snapshot = copy.deepcopy(request)
        extract_latest_user_text(request, "chat_completions")
        assert request == snapshot

    def test_codex_sdk_style_parts_rejected(self):
        # A non-dict, non-plain-text part shape is treated as unsupported, never flattened.
        request = {"input": [{"role": "user", "content": [{"type": "input_audio", "audio": "AAA"}]}]}
        assert extract_latest_user_text(request, "codex_responses") is None


class TestDetectCandidates:
    def test_arithmetic_expressions_are_candidates(self, settings):
        assert detect_candidates("2 + 2", settings) == ("calculator",)
        assert detect_candidates("quanto é (17 * 9) - 4?", settings) == ("calculator",)
        assert detect_candidates("calcule 12.5 / 5", settings) == ("calculator",)
        assert detect_candidates("what is 2+2?", settings) == ("calculator",)

    def test_context_dependent_short_reply_has_no_candidate(self, settings):
        assert detect_candidates("sim", settings) == ()
        assert detect_candidates("continue", settings) == ()
        assert detect_candidates("faça", settings) == ()

    def test_gratitude_acknowledgements_are_candidates(self, settings):
        assert detect_candidates("obrigado", settings) == ("acknowledgement",)
        assert detect_candidates("Obrigado!", settings) == ("acknowledgement",)
        assert detect_candidates("  valeu  ", settings) == ("acknowledgement",)
        assert detect_candidates("thanks", settings) == ("acknowledgement",)
        assert detect_candidates("thank you", settings) == ("acknowledgement",)

    def test_context_dependent_confirmations_are_never_candidates(self, settings):
        # A reply to an assistant question/proposal must fall through (decision 1).
        for text in (
            "ok", "Ok!", "certo", "beleza", "combinado", "fechou", "fechado",
            "perfeito", "entendi", "entendido", "got it", "understood", "noted",
            "sounds good", "roger that", "no problem", "legal", "show",
        ):
            assert detect_candidates(text, settings) == (), text

    def test_assistant_question_reply_falls_through(self, settings):
        # Assistant: "Posso apagar os arquivos antigos?" — user: "ok".
        assert detect_candidates("ok", settings) == ()

    def test_numbered_menu_reply_falls_through(self, settings):
        # Assistant shows a numbered menu — user: "2".
        assert detect_candidates("2", settings) == ()

    def test_bare_numbers_are_never_calculator_candidates(self, settings):
        for text in ("2", "42", "3.14", "11987654321", "123456", "-5", "(2)"):
            assert detect_candidates(text, settings) == (), text

    def test_phone_and_otp_numbers_never_reach_jev(self, settings):
        assert detect_candidates("11987654321", settings) == ()
        assert detect_candidates("123456", settings) == ()

    def test_hyphenated_phone_fragment_is_not_arithmetic(self, settings):
        # Audit L-a: "98765-4321" parses as a subtraction but is a phone/local-number
        # fragment; the eligibility gate must reject it before Jev ever sees the text.
        assert detect_candidates("98765-4321", settings) == ()
        assert detect_candidates("9876-5432", settings) == ()
        assert detect_candidates("98765-4321?", settings) == ()
        # Real math stays a calculator candidate: spaced operands and short operands.
        assert detect_candidates("98765 - 4321", settings) == ("calculator",)
        assert detect_candidates("10-3", settings) == ("calculator",)
        assert detect_candidates("1234-123", settings) == ("calculator",)

    def test_clock_requests_are_candidates(self, settings):
        assert detect_candidates("que horas são?", settings) == ("clock",)
        assert detect_candidates("que dia é hoje", settings) == ("clock",)
        assert detect_candidates("what time is it?", settings) == ("clock",)
        assert detect_candidates("What's the date today?", settings) == ("clock",)

    def test_clock_does_not_match_relative_scheduling(self, settings):
        assert detect_candidates("agende um lembrete para amanhã", settings) == ()
        assert detect_candidates("que horas serão em Lisboa quando eu chegar?", settings) == ()

    def test_runtime_identity_is_candidate(self, settings):
        assert detect_candidates("qual é o seu modelo?", settings) == ("runtime_identity",)
        assert detect_candidates("what model are you?", settings) == ("runtime_identity",)
        assert detect_candidates("qual provider você usa?", settings) == ("runtime_identity",)

    def test_status_request_is_candidate(self, settings):
        assert detect_candidates("qual é o status do fastpath?", settings) == ("fastpath_status",)
        assert detect_candidates("fastpath status", settings) == ("fastpath_status",)

    def test_dangerous_shapes_bypass_jev(self, settings):
        for text in (
            "curl https://example.com",
            "token=ts_SECRET_VALUE",
            "```python\n2+2\n```",
            "rm -rf /tmp/x",
            "cat /etc/passwd; echo hi",
            "password: hunter2",
        ):
            assert detect_candidates(text, settings) == ()

    def test_long_text_returns_no_candidates(self):
        settings = Settings(max_input_chars=32)
        assert detect_candidates("calcule 12.5 / 5 e mais um pouquinho de contexto", settings) == ()

    def test_slash_commands_return_no_candidates(self, settings):
        assert detect_candidates("/model sonnet", settings) == ()

    def test_disabled_handlers_are_not_candidates(self):
        settings = Settings(enabled_handlers=("calculator",))
        assert detect_candidates("que horas são?", settings) == ()
        assert detect_candidates("2 + 2", settings) == ("calculator",)

    def test_normal_llm_is_never_a_local_candidate(self, settings):
        for text in ("2 + 2", "obrigado", "que horas são?", "qual é o seu modelo?"):
            assert "normal_llm" not in detect_candidates(text, settings)

    def test_multiple_candidates_preserve_known_order(self, settings, monkeypatch):
        # Force every parser to accept so the candidate ORDER is exercised for real.
        import jev_fastpath.candidates as candidates_module

        monkeypatch.setattr(candidates_module, "_is_calculator_request", lambda text: True)
        monkeypatch.setattr(candidates_module, "classify_clock_request", lambda text: True)
        monkeypatch.setattr(candidates_module, "classify_identity_fields", lambda text: True)
        monkeypatch.setattr(candidates_module, "classify_acknowledgement", lambda text: True)
        monkeypatch.setattr(candidates_module, "classify_status_request", lambda text: True)
        assert detect_candidates("qualquer coisa", settings) == (
            "calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status",
        )

    def test_empty_text_has_no_candidates(self, settings):
        assert detect_candidates("", settings) == ()
        assert detect_candidates("   ", settings) == ()
