"""Strict TypeSafe Jev client: payload shape, parsing, and failure isolation."""

import contextlib
import http.server
import json
import threading
import time
import urllib.error
from typing import Any

import pytest

from jev_fastpath.config import Settings
from jev_fastpath.jev import (
    JEV_MODEL,
    TYPESAFE_URL,
    CircuitBreaker,
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


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch):
    """Isolate the module-level circuit breaker between tests."""
    import jev_fastpath.jev as jev_module

    monkeypatch.setattr(jev_module, "BREAKER", jev_module.CircuitBreaker())


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
    """HTTP-response fake with EOF semantics: one full read, then empty chunks."""

    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")
        self._done = False

    def read(self, size=-1):
        if self._done:
            return b""
        self._done = True
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
        class Raw:
            def __init__(self):
                self._done = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size=-1):
                if self._done:
                    return b""
                self._done = True
                return b"{not json"

        with pytest.raises(JevError) as excinfo:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=lambda request, timeout=None: Raw())
        assert excinfo.value.code == "invalid_response"

    def test_io_error_during_read_rejects(self):
        class Raw:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size=-1):
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


@pytest.fixture
def fake_secret_scope(monkeypatch):
    """Install a fake ``agent.secret_scope`` module emulating Hermes' multiplexed scope."""
    import sys
    import types

    agent_mod = types.ModuleType("agent")
    scope_mod = types.ModuleType("agent.secret_scope")
    state = {"value": "scoped-key-A", "raise": None, "calls": 0}

    def get_secret(name, default=None):
        state["calls"] += 1
        if state["raise"] is not None:
            raise state["raise"]
        return state["value"]

    scope_mod.get_secret = get_secret
    agent_mod.secret_scope = scope_mod
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", scope_mod)
    return state


class TestCredentialResolution:
    def test_scoped_secret_wins_over_ambient_env(self, fake_secret_scope):
        # Multiplexed gateway: scope key A + ambient env key B => Authorization uses A.
        decision, calls = classify_with(valid_response())
        request = calls[0]["request"]
        assert request.get_header("Authorization") == "Bearer scoped-key-A"

    def test_empty_scoped_secret_never_falls_back_to_env(self, fake_secret_scope):
        # The scoped value is empty and an ambient env key exists: fail open, never use B.
        fake_secret_scope["value"] = None
        with pytest.raises(JevError) as excinfo:
            classify_with(valid_response())
        assert excinfo.value.code == "missing_credential"

    def test_blank_scoped_secret_never_falls_back_to_env(self, fake_secret_scope):
        fake_secret_scope["value"] = "   "
        with pytest.raises(JevError) as excinfo:
            classify_with(valid_response())
        assert excinfo.value.code == "missing_credential"

    def test_unscoped_secret_error_fails_open(self, fake_secret_scope):
        fake_secret_scope["raise"] = RuntimeError("unscoped secret")
        with pytest.raises(JevError) as excinfo:
            classify_with(valid_response())
        assert excinfo.value.code == "credential_error"

    def test_scope_is_consulted_exactly_once_per_call(self, fake_secret_scope):
        classify_with(valid_response())
        classify_with(valid_response())
        assert fake_secret_scope["calls"] == 2

    def test_env_fallback_only_when_module_unavailable(self, monkeypatch):
        import sys

        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-key-B")
        monkeypatch.setitem(sys.modules, "agent", None)  # forces ImportError on import
        decision, calls = classify_with(valid_response())
        assert calls[0]["request"].get_header("Authorization") == "Bearer env-key-B"


class _SlowRaw:
    """Response whose read blocks past the deadline (EOF after one chunk)."""

    def __init__(self, delay, payload):
        self._delay = delay
        self._data = json.dumps(payload).encode("utf-8")
        self._done = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if self._done:
            return b""
        self._done = True
        time.sleep(self._delay)
        return self._data


class TestTransportBounds:
    def test_deadline_bounds_slow_read(self):
        def opener(request, timeout=None):
            return _SlowRaw(0.5, valid_response())

        started = time.monotonic()
        with pytest.raises(JevError) as excinfo:
            classify("2 + 2", ("calculator",), {}, Settings(timeout_seconds=0.2), opener=opener)
        elapsed = time.monotonic() - started
        assert excinfo.value.code == "timeout"
        assert elapsed < 5.0

    def test_oversized_response_rejected(self):
        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            return _SlowRaw(0, {"answers": {}, "pad": "x" * 70000})

        with pytest.raises(JevError) as excinfo:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener)
        assert excinfo.value.code == "response_too_large"
        assert len(calls) == 1

    def test_redirect_rejected_without_second_request(self):
        import urllib.error

        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            raise urllib.error.HTTPError(request.full_url, 302, "Found", hdrs=None, fp=None)

        with pytest.raises(JevError) as excinfo:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener)
        assert excinfo.value.code == "redirect"
        assert len(calls) == 1  # never followed, so the API key is never re-sent

    def test_default_opener_rejects_redirects(self):
        from jev_fastpath.jev import _NoRedirect

        handler = _NoRedirect()
        with pytest.raises(urllib.error.HTTPError):
            handler.redirect_request(None, None, 302, "Found", None, "https://evil.example/")

    def test_http_error_reason_codes(self):
        import urllib.error

        def opener_for(code):
            def opener(request, timeout=None):
                raise urllib.error.HTTPError(request.full_url, code, "x", hdrs=None, fp=None)

            return opener

        with pytest.raises(JevError) as auth:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener_for(401))
        assert auth.value.code == "auth_error"
        with pytest.raises(JevError) as http:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener_for(500))
        assert http.value.code == "http_error"


class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
    """Real HTTP handler: records each request, then answers from ``server.script``."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        server: Any = self.server
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        server.requests.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": body,
        })
        status, headers, payload = server.script(self.path)
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def log_message(self, format, *args):  # keep the pytest output clean
        pass


class _LoopbackServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, script):
        super().__init__(("127.0.0.1", 0), _LoopbackHandler)
        self.script = script
        self.requests = []


@contextlib.contextmanager
def loopback_server(script):
    """Serve real HTTP on 127.0.0.1 and yield ``(server, url)`` (audit C1 regression seam)."""
    server = _LoopbackServer(script)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


class TestDefaultOpenerTransport:
    """Audit C1: ``opener=None`` must complete over a real socket, not only via fakes."""

    def test_default_opener_completes_against_loopback_200(self, monkeypatch):
        import jev_fastpath.jev as jev_module

        payload = json.dumps(valid_response()).encode("utf-8")

        def script(path):
            return 200, {"Content-Type": "application/json"}, payload

        with loopback_server(script) as (server, url):
            monkeypatch.setattr(jev_module, "TYPESAFE_URL", url)
            decision = classify("2 + 2", ("calculator",), {"platform": "cli"}, Settings())

        assert decision.handler_id == "calculator"
        assert decision.confidence == 0.97
        assert decision.short_circuit_probability == 0.95
        assert len(server.requests) == 1
        sent = server.requests[0]
        assert sent["path"] == "/v1/systemone"
        assert sent["authorization"] == f"Bearer {API_KEY}"
        body = json.loads(sent["body"].decode("utf-8"))
        assert body["state"]["message"] == "2 + 2"
        assert body["state"]["candidate_handlers"] == ["calculator"]

    def test_default_opener_rejects_loopback_redirect_without_replay(self, monkeypatch):
        import jev_fastpath.jev as jev_module

        payload = json.dumps(valid_response()).encode("utf-8")

        def script(path):
            if path == "/v1/systemone":
                return 302, {"Location": "/second"}, b""
            return 200, {"Content-Type": "application/json"}, payload

        with loopback_server(script) as (server, url):
            monkeypatch.setattr(jev_module, "TYPESAFE_URL", url)
            with pytest.raises(JevError) as excinfo:
                classify("2 + 2", ("calculator",), {}, Settings())

        assert excinfo.value.code == "redirect"
        # Exactly one request, to the configured endpoint only: the Authorization header
        # is never replayed to the redirect target.
        assert [request["path"] for request in server.requests] == ["/v1/systemone"]
        assert server.requests[0]["authorization"] == f"Bearer {API_KEY}"


class TestCircuitBreaker:
    def _two_failures(self, breaker):
        def opener(request, timeout=None):
            raise TimeoutError("down")

        for _ in range(2):
            with pytest.raises(JevError):
                classify(
                    "2 + 2", ("calculator",), {}, Settings(), opener=opener, breaker=breaker,
                )

    def test_opens_after_threshold_and_blocks_without_network(self):
        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            raise TimeoutError("down")

        breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30.0)
        self._two_failures(breaker)
        assert breaker.is_open()
        with pytest.raises(JevError) as excinfo:
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener, breaker=breaker)
        assert excinfo.value.code == "circuit_open"
        assert calls == []  # blocked before any network attempt

    def test_cooldown_expiry_allows_probe_again(self):
        breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
        self._two_failures(breaker)
        assert breaker.is_open()
        time.sleep(0.1)
        assert not breaker.is_open()  # half-open allows a probe again

    def test_success_resets_failure_count(self):
        attempts = {"n": 0}

        def opener(request, timeout=None):
            attempts["n"] += 1
            if attempts["n"] % 2 == 1:
                raise TimeoutError("down")
            return FakeResponse(valid_response())

        breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30.0)
        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener, breaker=breaker)
        classify("2 + 2", ("calculator",), {}, Settings(), opener=opener, breaker=breaker)
        with pytest.raises(JevError):
            classify("2 + 2", ("calculator",), {}, Settings(), opener=opener, breaker=breaker)
        assert not breaker.is_open()
