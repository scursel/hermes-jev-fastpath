"""Strict TypeSafe Jev client: payload shape, parsing, and failure isolation."""

import json
import math

import pytest

from jev_fastpath.config import Settings
from jev_fastpath.jev import (
    JEV_MODEL,
    TYPESAFE_URL,
    JevError,
    build_questions,
    classify,
    parse_response,
    redact_and_bound,
)

API_KEY = "test-typesafe-key"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", API_KEY)


def valid_response(choice="calculator", confidence=0.97, noul=0.95):
    return {
        "answers": {
            "handler": {"choice": choice, "confidence": confidence},
            "safe_to_short_circuit": {"noul": noul},
        },
        "usage": {"cost": 0.001, "input_tokens": 10, "output_tokens": 5},
        "model": "jev-latest",
    }


class FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_opener(payload, calls):
    def opener(request, timeout=None):
        calls.append({"request": request, "timeout": timeout})
        return FakeResponse(payload)

    return opener


def classify_with(payload, candidates=("calculator",), text="2 + 2", **kwargs):
    calls = []
    decision = classify(
        text, candidates, {"platform": "telegram"}, Settings(),
        opener=make_opener(payload, calls), **kwargs,
    )
    return decision, calls


class TestBuildQuestions:
    def test_contains_choice_and_noul_questions(self):
        questions = build_questions(("calculator",))
        assert questions["handler"]["type"] == "choice"
        assert questions["safe_to_short_circuit"]["type"] == "noul"
        assert set(questions["handler"]["criteria"]) == {"calculator", "normal_llm"}

    def test_criteria_cover_every_candidate(self):
        questions = build_questions(("clock", "acknowledgement"))
        assert set(questions["handler"]["criteria"]) == {"clock", "acknowledgement", "normal_llm"}


class TestClassifyRequest:
    def test_exact_payload_shape(self):
        decision, calls = classify_with(valid_response())
        assert len(calls) == 1
        request, timeout = calls[0]["request"], calls[0]["timeout"]
        assert request.full_url == TYPESAFE_URL
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == f"Bearer {API_KEY}"
        assert request.get_header("Content-type") == "application/json"
        assert timeout == 3.0
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == JEV_MODEL
        assert body["state"] == {
            "message": "2 + 2",
            "platform": "telegram",
            "candidate_handlers": ["calculator"],
        }
        assert set(body["questions"]) == {"handler", "safe_to_short_circuit"}
        assert decision.handler_id == "calculator"

    def test_exactly_one_network_attempt(self):
        calls = []
        opener = make_opener(valid_response(), calls)
        classify("2 + 2", ("calculator",), {}, Settings(), opener=opener)
        assert len(calls) == 1

    def test_message_is_redacted_and_bounded(self):
        calls = []
        opener = make_opener(valid_response(), calls)
        classify(
            "password: hunter2 " + "9" * 5000, ("calculator",), {}, Settings(max_input_chars=128),
            opener=opener,
        )
        body = json.loads(calls[0]["request"].data.decode("utf-8"))
        message = body["state"]["message"]
        assert "hunter2" not in message
        assert len(message) <= 128

    def test_missing_api_key_rejects(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=make_opener(valid_response(), []))
        monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=make_opener(valid_response(), []))


class TestClassifyFailures:
    def test_http_error_has_no_body_or_auth_leak(self):
        import urllib.error

        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 401, "Unauthorized", hdrs=None, fp=None,
            )

        with pytest.raises(JevError) as excinfo:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener)
        assert API_KEY not in str(excinfo.value)
        assert "401" in str(excinfo.value)

    def test_timeout_maps_to_jev_error(self):
        def opener(request, timeout=None):
            raise TimeoutError("slow")

        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener)

    def test_url_error_maps_to_jev_error(self):
        import urllib.error

        def opener(request, timeout=None):
            raise urllib.error.URLError("no route")

        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener)

    def test_invalid_json_rejects(self):
        def opener(request, timeout=None):
            return FakeResponse.__new__(FakeResponse)  # bypass __init__

        def opener2(request, timeout=None):
            class Raw:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def read(self):
                    return b"{not json"

            return Raw()

        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener2)

    def test_io_error_during_read_rejects(self):
        class Raw:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                raise OSError("connection reset")

        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=lambda request, timeout=None: Raw())


class TestParseResponse:
    def test_happy_path_preserves_usage_and_model(self):
        decision = parse_response(valid_response(), ("calculator",), 42)
        assert decision.handler_id == "calculator"
        assert decision.confidence == 0.97
        assert decision.short_circuit_probability == 0.95
        assert decision.latency_ms == 42
        assert decision.usage == {"cost": 0.001, "input_tokens": 10, "output_tokens": 5}
        assert decision.model == "jev-latest"

    def test_normal_llm_choice_is_valid(self):
        decision = parse_response(valid_response(choice="normal_llm"), ("calculator",), 1)
        assert decision.handler_id == "normal_llm"

    def test_non_dict_payload_rejected(self):
        for payload in (None, [], "answers", 42):
            with pytest.raises(JevError):
                parse_response(payload, ("calculator",), 1)

    def test_missing_answers_rejected(self):
        with pytest.raises(JevError):
            parse_response({"model": "jev-latest"}, ("calculator",), 1)

    @pytest.mark.parametrize("payload", [
        {}, {"answers": {"handler": {"choice": "calculator", "confidence": 0.9}}},
        {"answers": {"handler": None, "safe_to_short_circuit": {"noul": 0.9}}},
        {"answers": {"handler": {"choice": "calculator", "confidence": 0.9}, "safe_to_short_circuit": None}},
    ])
    def test_missing_answer_sections_rejected(self, payload):
        with pytest.raises(JevError):
            parse_response(payload, ("calculator",), 1)

    def test_unknown_handler_rejected(self):
        payload = valid_response(choice="terminal")
        with pytest.raises(JevError):
            parse_response(payload, ("calculator",))

    def test_choice_outside_candidates_rejected(self):
        with pytest.raises(JevError):
            parse_response(valid_response(choice="clock"), ("calculator",))

    @pytest.mark.parametrize("bad", [None, True, "0.95", -0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_probability_rejected(self, bad):
        payload = valid_response()
        payload["answers"]["handler"]["confidence"] = bad
        with pytest.raises(JevError):
            parse_response(payload, ("calculator",))

    @pytest.mark.parametrize("bad", [None, False, "0.9", -0.01, 1.01, float("nan"), -float("inf")])
    def test_invalid_noul_rejected(self, bad):
        payload = valid_response()
        payload["answers"]["safe_to_short_circuit"]["noul"] = bad
        with pytest.raises(JevError):
            parse_response(payload, ("calculator",))

    def test_zero_and_one_probabilities_accepted(self):
        decision = parse_response(valid_response(confidence=0, noul=1), ("calculator",), 1)
        assert decision.confidence == 0.0
        assert decision.short_circuit_probability == 1.0

    def test_non_dict_usage_becomes_empty(self):
        payload = valid_response()
        payload["usage"] = "free"
        assert parse_response(payload, ("calculator",), 1).usage == {}

    def test_non_string_model_becomes_none(self):
        payload = valid_response()
        payload["model"] = 42
        assert parse_response(payload, ("calculator",), 1).model is None

    def test_integer_probability_accepted(self):
        decision = parse_response(valid_response(confidence=1, noul=0), ("calculator",), 1)
        assert decision.confidence == 1.0
        assert decision.short_circuit_probability == 0.0


class TestRedactAndBound:
    def test_bounds_length(self):
        assert len(redact_and_bound("x" * 1000, 32)) <= 32

    def test_redacts_secret_shapes(self):
        text = "my key: sk-live-abcdefghij and bearer Abcdef123456 token"
        out = redact_and_bound(text, 400)
        assert "sk-live-abcdefghij" not in out
        assert "Abcdef123456" not in out

    def test_plain_text_unchanged(self):
        assert redact_and_bound("quanto é 2 + 2?", 100) == "quanto é 2 + 2?"

    def test_non_string_becomes_empty(self):
        assert redact_and_bound(None, 100) == ""


def test_constants_match_spec():
    assert TYPESAFE_URL == "https://api.typesafe.ai/v1/systemone"
    assert JEV_MODEL == "jev-latest"
    assert not math.isnan(1.0)
