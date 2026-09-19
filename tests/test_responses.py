"""Synthetic provider responses: raw protocol shapes plus real Hermes transport verification.

The transport-backed tests require the Hermes source tree on ``PYTHONPATH`` and skip with a
clear reason when it is absent; the raw-shape tests always run.
"""

import pytest

from jev_fastpath.responses import SUPPORTED_API_MODES, UnsupportedApiMode, build_synthetic_response

TEXT = "resultado"


class TestRawShapes:
    def test_chat_completions_shape(self):
        raw = build_synthetic_response("chat_completions", TEXT)
        assert raw.id and raw.model
        assert raw.choices[0].message.role == "assistant"
        assert raw.choices[0].message.content == TEXT
        assert raw.choices[0].message.tool_calls is None
        assert raw.choices[0].finish_reason == "stop"
        assert raw.usage is None

    def test_codex_responses_shape(self):
        raw = build_synthetic_response("codex_responses", TEXT)
        assert raw.status == "completed"
        assert raw.usage is None
        item = raw.output[0]
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.status == "completed"
        assert item.content[0].type == "output_text"
        assert item.content[0].text == TEXT

    def test_anthropic_messages_shape(self):
        raw = build_synthetic_response("anthropic_messages", TEXT)
        assert raw.role == "assistant"
        assert raw.content[0].type == "text"
        assert raw.content[0].text == TEXT
        assert raw.stop_reason == "end_turn"
        assert raw.usage is None

    def test_bedrock_converse_shape(self):
        raw = build_synthetic_response("bedrock_converse", TEXT)
        assert raw["stopReason"] == "end_turn"
        block = raw["output"]["message"]["content"][0]
        assert block["text"] == TEXT
        assert raw["output"]["message"]["role"] == "assistant"
        assert "usage" not in raw

    def test_marker_present_and_private(self):
        for api_mode in SUPPORTED_API_MODES:
            raw = build_synthetic_response(api_mode, TEXT)
            if isinstance(raw, dict):
                assert raw.get("_jev_fastpath") is True
            else:
                assert getattr(raw, "_jev_fastpath") is True

    def test_supported_modes_are_exactly_the_spec_four(self):
        assert SUPPORTED_API_MODES == (
            "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
        )

    def test_unknown_api_mode_rejected(self):
        with pytest.raises(UnsupportedApiMode):
            build_synthetic_response("future_protocol", "x")

    @pytest.mark.parametrize("bad_text", ["", "   ", None, 42])
    def test_empty_or_non_string_text_rejected(self, bad_text):
        with pytest.raises(ValueError):
            build_synthetic_response("chat_completions", bad_text)


@pytest.fixture
def hermes_transport():
    pytest.importorskip(
        "agent.transports",
        reason="real Hermes transports required; run with PYTHONPATH pointing at the Hermes checkout",
    )
    from agent.transports import get_transport

    def _factory(api_mode):
        transport = get_transport(api_mode)
        assert transport is not None, f"no Hermes transport registered for {api_mode}"
        return transport

    return _factory


@pytest.mark.parametrize("api_mode", SUPPORTED_API_MODES)
class TestHermesTransportNormalization:
    def test_synthetic_response_validates(self, hermes_transport, api_mode):
        raw = build_synthetic_response(api_mode, TEXT)
        assert hermes_transport(api_mode).validate_response(raw) is True

    def test_synthetic_response_normalizes(self, hermes_transport, api_mode):
        raw = build_synthetic_response(api_mode, TEXT)
        normalized = hermes_transport(api_mode).normalize_response(raw)
        assert normalized.content == TEXT
        assert normalized.finish_reason == "stop"
        assert normalized.tool_calls in (None, [])
        if api_mode == "bedrock_converse":
            # The raw dict carries no provider usage; Bedrock's normalizer synthesizes a
            # zero-usage object, which is the strongest "no fake provider usage" guarantee
            # this transport can express.
            usage = normalized.usage
            assert usage is None or (
                getattr(usage, "prompt_tokens", 0) == 0 and getattr(usage, "completion_tokens", 0) == 0
            )
        else:
            assert normalized.usage is None
