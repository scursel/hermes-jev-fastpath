"""FastPathRuntime middleware matrix: exactly-once fallthrough, zero-call fast paths."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import jev_fastpath.plugin as plugin_module
from jev_fastpath.config import Settings
from jev_fastpath.cache import DecisionCache
from jev_fastpath.plugin import FastPathRuntime, register
from jev_fastpath.types import Decision, TelemetryEvent


class TelemetryRecorder:
    def __init__(self):
        self.events = []

    def write(self, event, context):
        self.events.append((event, dict(context)))


def _decision(handler="calculator", confidence=0.97, noul=0.95):
    return Decision(
        handler_id=handler, confidence=confidence, short_circuit_probability=noul, latency_ms=5,
    )


def _request(text="2 + 2"):
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


def _context(**overrides):
    values = dict(
        platform="telegram", provider="nous", model="m",
        session_id="s1", turn_id="t1", api_mode="chat_completions",
    )
    values.update(overrides)
    return values


def _accepted_classifier(calls=None):
    def _classify(text, candidates, context, settings):
        if calls is not None:
            calls.append((text, candidates))
        return _decision()

    return _classify


@pytest.fixture
def runtime():
    return FastPathRuntime(
        Settings(mode="active"),
        classifier=_accepted_classifier(),
        telemetry=TelemetryRecorder(),
    )


@pytest.fixture
def downstream():
    mock = Mock(return_value=object())
    return mock


class TestActiveAcceptance:
    def test_active_acceptance_skips_provider(self, runtime, downstream):
        response = runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_not_called()
        assert response.choices[0].message.content == "2 + 2 = 4"

    def test_next_call_receives_original_request(self, runtime, downstream):
        request = _request()
        runtime.middleware(request=request, next_call=downstream, api_call_count=1, **_context())
        downstream.assert_not_called()

    def test_acceptance_emits_short_circuit_telemetry(self, runtime, downstream):
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        event, meta = runtime.telemetry.events[-1]
        assert event.outcome == "short_circuit"
        assert event.selected_handler == "calculator"
        assert meta["mode"] == "active"
        assert meta["session_id"] == "s1"
        assert meta["api_mode"] == "chat_completions"


class TestFailOpenPaths:
    def test_failure_calls_provider_exactly_once(self, runtime, downstream):
        runtime.classifier = Mock(side_effect=TimeoutError("slow"))
        downstream_response = object()
        downstream = Mock(return_value=downstream_response)
        assert runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context()) is downstream_response
        downstream.assert_called_once_with(_request())

    def test_mode_off_bypasses_without_classifier(self, downstream):
        runtime = FastPathRuntime(Settings(mode="off"), classifier=Mock(side_effect=AssertionError))
        response = object()
        downstream = Mock(return_value=response)
        assert runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context()) is response
        downstream.assert_called_once()

    def test_retry_round_bypasses(self, runtime, downstream):
        # Hermes counts attempts 1-based; the second attempt of a turn (tool round,
        # retry, fallback, or continuation) is api_call_count >= 2 and must bypass.
        downstream = Mock(return_value=object())
        runtime.middleware(
            request=_request(), next_call=downstream, api_call_count=2, **_context()
        )
        downstream.assert_called_once()

    def test_first_attempt_is_eligible_one_based(self, runtime, downstream):
        # Regression: Hermes increments api_call_count BEFORE the call, so the first
        # attempt of a turn arrives as api_call_count == 1, not 0.
        downstream = Mock(side_effect=AssertionError("provider must not run"))
        response = runtime.middleware(
            request=_request(), next_call=downstream, api_call_count=1, **_context()
        )
        downstream.assert_not_called()
        assert response.choices[0].message.content == "2 + 2 = 4"

    def test_unsupported_api_mode_falls_through_without_jev(self, downstream):
        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(Settings(mode="active"), classifier=Mock(side_effect=AssertionError), telemetry=telemetry)
        response = object()
        downstream = Mock(return_value=response)
        context = _context(api_mode="future_protocol")
        assert runtime.middleware(request=_request(), next_call=downstream, **context) is response
        downstream.assert_called_once()
        assert telemetry.events == []

    def test_no_candidate_falls_through_without_jev(self, downstream):
        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=Mock(side_effect=AssertionError), telemetry=telemetry,
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request("fale sobre arte moderna"), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()
        assert telemetry.events[0][0].outcome == "no_candidate"

    def test_normal_llm_choice_falls_through(self, downstream):
        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"),
            classifier=lambda *a: _decision(handler="normal_llm"),
            telemetry=telemetry,
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()
        event = telemetry.events[0][0]
        assert event.outcome == "normal_llm"
        assert event.selected_handler == "normal_llm"

    def test_low_confidence_falls_through(self, downstream):
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=lambda *a: _decision(confidence=0.5),
            telemetry=TelemetryRecorder(),
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()

    def test_low_short_circuit_probability_falls_through(self, downstream):
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=lambda *a: _decision(noul=0.5),
            telemetry=TelemetryRecorder(),
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()

    def test_handler_rejection_falls_through(self, downstream):
        runtime = FastPathRuntime(
            Settings(mode="active"),
            classifier=_accepted_classifier(),
            renderer=Mock(side_effect=ValueError("handler not enabled: calculator")),
            telemetry=TelemetryRecorder(),
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()

    def test_response_factory_rejection_falls_through(self, downstream):
        runtime = FastPathRuntime(
            Settings(mode="active"),
            classifier=_accepted_classifier(),
            response_factory=Mock(side_effect=ValueError("boom")),
            telemetry=TelemetryRecorder(),
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()

    def test_internal_exception_falls_through_once(self, monkeypatch, downstream):
        def _explode(request, api_mode):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(plugin_module, "extract_latest_user_text", _explode)
        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(Settings(mode="active"), telemetry=telemetry)
        downstream_response = object()
        downstream = Mock(return_value=downstream_response)
        assert runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context()) is downstream_response
        downstream.assert_called_once_with(_request())
        assert telemetry.events[-1][0].outcome == "fallback"

    def test_fallback_reason_uses_jev_reason_code(self, downstream):
        from jev_fastpath.jev import JevError

        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"),
            classifier=Mock(side_effect=JevError("TypeSafe deadline exceeded", code="timeout")),
            telemetry=telemetry,
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()
        event = telemetry.events[-1][0]
        assert event.outcome == "fallback"
        assert event.reason == "timeout"  # bounded non-secret reason code (audit L2)

    def test_provider_exception_is_not_swallowed_or_duplicated(self, runtime):
        runtime.classifier = lambda *a: _decision(handler="normal_llm")
        downstream = Mock(side_effect=RuntimeError("provider down"))
        with pytest.raises(RuntimeError, match="provider down"):
            runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        downstream.assert_called_once()


class TestShadowMode:
    def test_shadow_returns_exact_downstream_response(self, downstream):
        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="shadow"), classifier=_accepted_classifier(), telemetry=telemetry,
        )
        downstream_response = object()
        downstream = Mock(return_value=downstream_response)
        result = runtime.middleware(
            request=_request(), next_call=downstream, api_call_count=1, **_context()
        )
        assert result is downstream_response
        downstream.assert_called_once()
        event = telemetry.events[-1][0]
        assert event.outcome == "would_short_circuit"
        assert event.selected_handler == "calculator"

    def test_shadow_classification_and_rendering_occur(self, downstream):
        renderer = Mock(return_value="2 + 2 = 4")
        runtime = FastPathRuntime(
            Settings(mode="shadow"), classifier=_accepted_classifier(), renderer=renderer,
            telemetry=TelemetryRecorder(),
        )
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        renderer.assert_called_once()
        assert renderer.call_args[0][0] == "calculator"
        assert renderer.call_args[0][1] == "2 + 2"
        status = renderer.call_args[0][4]
        assert status["mode"] == "shadow"

    def test_injected_now_fn_reaches_renderer_not_context(self, downstream):
        # The clock seam is an explicit runtime dependency, never a Hermes-context override.
        renderer = Mock(return_value="2 + 2 = 4")
        clock = Mock(return_value="now")
        runtime = FastPathRuntime(
            Settings(mode="shadow"), classifier=_accepted_classifier(), renderer=renderer,
            telemetry=TelemetryRecorder(), now_fn=clock,
        )
        context = _context(now_fn="hostile-injection-attempt")
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **context)
        renderer.assert_called_once()
        assert renderer.call_args.kwargs["now_fn"] is clock


class TestDecisionCache:
    def test_same_turn_reuses_classifier_decision(self, downstream):
        calls = []
        cache = DecisionCache()
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls), cache=cache,
            telemetry=TelemetryRecorder(),
        )
        first = runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        # Second invocation for the SAME turn (retry/restart shape): the per-turn claim
        # makes it fall through to the provider without a second Jev call.
        second = runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        assert len(calls) == 1
        assert downstream.call_count == 1
        assert first.choices[0].message.content == "2 + 2 = 4"
        assert second is not first.choices  # the provider response came from downstream

    def test_different_turns_and_texts_miss_cache(self, downstream):
        calls = []
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls),
            telemetry=TelemetryRecorder(),
        )
        runtime.middleware(request=_request("2 + 2"), next_call=downstream, api_call_count=1, **_context())
        runtime.middleware(
            request=_request("3 + 3"), next_call=downstream, api_call_count=1, **_context(turn_id="t2")
        )
        assert len(calls) == 2

    def test_missing_identity_is_ineligible_not_uncached(self, downstream):
        calls = []
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls),
            telemetry=TelemetryRecorder(),
        )
        context = _context(session_id="", turn_id="")
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **context)
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **context)
        # Without a turn ID the turn is never eligible, so Jev is never consulted.
        assert calls == []
        assert downstream.call_count == 2

    def test_concurrent_distinct_turns_each_call_provider_once(self, downstream):
        telemetry = TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(), telemetry=telemetry,
        )
        lock = threading.Lock()
        seen = []

        def _turn(turn):
            response = runtime.middleware(
                request=_request(), next_call=downstream, api_call_count=1, **_context(turn_id=turn)
            )
            with lock:
                seen.append(response.choices[0].message.content)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_turn, (f"t{index}" for index in range(16))))
        assert len(seen) == 16
        assert all(text == "2 + 2 = 4" for text in seen)
        assert downstream.call_count == 0


class TestStatus:
    def test_status_reflects_settings_and_updates_last_decision(self, runtime, downstream):
        status = runtime.status()
        assert status["mode"] == "active"
        assert status["confidence_threshold"] == 0.92
        assert status["last_decision_at"] is None
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        assert runtime.status()["last_decision_at"] is not None


class TestRegister:
    class FakeContext:
        def __init__(self, values=None):
            self.values = values or {}
            self.registered = []

        def get_config(self, key, default=None):
            return self.values.get(key, default)

        def register_middleware(self, kind, callback):
            self.registered.append((kind, callback))

    @pytest.fixture(autouse=True)
    def _fake_hermes_home(self, tmp_path, monkeypatch):
        import sys
        import types

        module = types.ModuleType("hermes_constants")

        def get_hermes_home():
            return tmp_path

        module.get_hermes_home = get_hermes_home
        monkeypatch.setitem(sys.modules, "hermes_constants", module)
        return tmp_path

    def test_register_wires_llm_execution_middleware(self):
        ctx = self.FakeContext({"mode": "active"})
        register(ctx)
        kinds = [kind for kind, _ in ctx.registered]
        assert kinds == ["llm_execution"]
        callback = ctx.registered[0][1]
        assert callback.__name__ == "middleware"

    def test_register_with_invalid_settings_registers_nothing(self):
        ctx = self.FakeContext({"mode": "bogus"})
        register(ctx)
        assert ctx.registered == []

    def test_registered_middleware_end_to_end_fast_path(self, downstream):
        ctx = self.FakeContext({"mode": "active"})
        register(ctx)
        callback = ctx.registered[0][1]
        # Replace the real (credential-needing) classifier on the registered runtime.
        callback.__self__.classifier = _accepted_classifier()
        request = _request()
        response = callback(request=request, next_call=downstream, api_call_count=1, **_context())
        downstream.assert_not_called()
        assert response.choices[0].message.content == "2 + 2 = 4"


class TestTurnEligibility:
    """H1: exactly one evaluation per (session_id, turn_id), regardless of api_call_count."""

    def test_second_invocation_same_turn_never_re_evaluates(self):
        calls, telemetry = [], TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls), telemetry=telemetry,
        )
        downstream = Mock(return_value=object())
        # Hermes can deliver api_call_count == [1, 1] across retries/restarts; only the
        # FIRST middleware invocation for the turn may evaluate.
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        second = runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        third = runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        assert len(calls) == 1             # Jev exactly once for the turn
        assert downstream.call_count == 2  # invocations 2 and 3 fall through
        assert second is not None and third is not None
        assert len(telemetry.events) == 1  # no duplicate telemetry rows

    def test_different_turn_evaluates_again(self):
        calls = []
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls),
            telemetry=TelemetryRecorder(),
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        runtime.middleware(
            request=_request(), next_call=downstream, api_call_count=1, **_context(turn_id="t2")
        )
        assert len(calls) == 2

    def test_missing_turn_id_is_ineligible(self):
        calls, telemetry = [], TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls), telemetry=telemetry,
        )
        downstream = Mock(return_value=object())
        runtime.middleware(
            request=_request(), next_call=downstream, api_call_count=1, **_context(turn_id="")
        )
        downstream.assert_called_once()
        assert calls == []
        assert telemetry.events == []

    def test_missing_api_call_count_is_ineligible(self):
        # L3: an absent count must not imply the first attempt.
        calls, telemetry = [], TelemetryRecorder()
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(calls), telemetry=telemetry,
        )
        downstream = Mock(return_value=object())
        runtime.middleware(request=_request(), next_call=downstream, **_context())
        downstream.assert_called_once()
        assert calls == []
        assert telemetry.events == []

    def test_claim_is_bounded_and_expires(self):
        from jev_fastpath.plugin import TurnClaim

        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        clock = Clock()
        claims = TurnClaim(ttl_seconds=60.0, max_entries=2, monotonic=clock)
        assert claims.claim("s1", "t1") is True
        assert claims.claim("s1", "t1") is False          # already claimed
        clock.now = 61.0                                   # TTL elapsed -> new turn generation
        assert claims.claim("s1", "t1") is True
        assert claims.claim("s1", "") is False             # empty turn never claimable
        assert claims.claim("s1", "t2") is True
        assert claims.claim("s2", "t3") is True            # bound: evicts the oldest entry
        assert claims.claim("s1", "t1") is True

    def test_claim_is_concurrency_safe(self):
        from jev_fastpath.plugin import TurnClaim
        from concurrent.futures import ThreadPoolExecutor

        claims = TurnClaim()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: claims.claim("s1", "t1"), range(32)))
        assert results.count(True) == 1

    def test_status_shows_prior_decision_not_current(self, downstream):
        runtime = FastPathRuntime(
            Settings(mode="active"), classifier=_accepted_classifier(),
            telemetry=TelemetryRecorder(),
        )
        seen = []

        def renderer(handler_id, text, context, settings, status, *, now_fn=None):
            seen.append(status["last_decision_at"])
            return "2 + 2 = 4"

        runtime.renderer = renderer
        runtime.middleware(request=_request(), next_call=downstream, api_call_count=1, **_context())
        runtime.middleware(
            request=_request(), next_call=downstream, api_call_count=1, **_context(turn_id="t2")
        )
        assert seen[0] is None                      # first render: no prior decision yet
        assert seen[1] is not None                  # second render: shows the PRIOR decision
        assert seen[1] < runtime.status()["last_decision_at"]
