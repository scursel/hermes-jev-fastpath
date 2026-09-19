"""Real Hermes host integration: plugin discovery, middleware chain, and agent loops.

These tests exercise the actual Hermes runtime (PluginManager, transports,
``AIAgent.run_conversation``) with fake providers and no credentials; they skip when the
Hermes source tree or package is not importable.
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip(
    "hermes_cli.plugins",
    reason="Hermes runtime required; run with PYTHONPATH pointing at the Hermes checkout",
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_PLUGIN_FILES = ("__init__.py", "plugin.yaml")


def _copy_plugin(destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in _PLUGIN_FILES:
        shutil.copy2(REPO_ROOT / name, destination / name)
    shutil.copytree(REPO_ROOT / "jev_fastpath", destination / "jev_fastpath")


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME with this plugin installed and enabled in shadow mode."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - jev-fastpath\n"
        "  entries:\n"
        "    jev-fastpath:\n"
        "      enabled: true\n"
        "      settings:\n"
        "        mode: shadow\n"
        "        confidence_threshold: 0.92\n"
        "        short_circuit_threshold: 0.90\n",
        encoding="utf-8",
    )
    bundled = tmp_path / "bundled-plugins"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    yield tmp_path
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    _reset_plugin_managers_for_tests()


def _discover(hermes_home):
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager.discover_and_load()
    return manager


class TestPluginDiscovery:
    def test_discovery_registers_llm_execution_middleware(self, hermes_home):
        _copy_plugin(hermes_home / "plugins" / "jev-fastpath")
        manager = _discover(hermes_home)
        loaded = manager._plugins["jev-fastpath"]
        assert loaded.enabled is True
        assert loaded.error is None
        assert loaded.middleware_registered == ["llm_execution"]
        assert manager.has_middleware("llm_execution") is True

    def test_disabled_plugin_is_not_loaded(self, hermes_home):
        _copy_plugin(hermes_home / "plugins" / "jev-fastpath")
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  enabled: []\n", encoding="utf-8"
        )
        manager = _discover(hermes_home)
        loaded = manager._plugins.get("jev-fastpath")
        assert loaded is None or loaded.enabled is False


class TestMiddlewareChain:
    def test_real_chain_fails_open_without_credentials(self, hermes_home):
        """Through ``run_llm_execution_middleware`` with no API key: one downstream call."""
        from hermes_cli.middleware import run_llm_execution_middleware

        _copy_plugin(hermes_home / "plugins" / "jev-fastpath")
        manager = _discover(hermes_home)
        callback = manager._middleware["llm_execution"][0]

        request = {"model": "m", "messages": [{"role": "user", "content": "2 + 2"}]}
        sentinel = object()
        calls = []

        def next_call(received):
            calls.append(received)
            return sentinel

        response = run_llm_execution_middleware(
            request, next_call,
            session_id="s1", turn_id="t1", api_mode="chat_completions",
            api_call_count=1, platform="cli", provider="nous", model="m",
        )
        assert response is sentinel
        assert calls == [request]

    def test_real_chain_fast_path_skips_downstream(self, hermes_home, monkeypatch):
        from hermes_cli.middleware import run_llm_execution_middleware
        from jev_fastpath.types import Decision

        _copy_plugin(hermes_home / "plugins" / "jev-fastpath")
        manager = _discover(hermes_home)
        callback = manager._middleware["llm_execution"][0]
        runtime = callback.__self__
        runtime.settings = type(runtime.settings)(
            **{**runtime.settings.__dict__, "mode": "active"}
        )
        runtime.classifier = lambda *a: Decision(
            handler_id="calculator", confidence=0.97, short_circuit_probability=0.95, latency_ms=1,
        )

        request = {"model": "m", "messages": [{"role": "user", "content": "2 + 2"}]}

        def next_call(received):
            raise AssertionError("provider must not run")

        response = run_llm_execution_middleware(
            request, next_call,
            session_id="s1", turn_id="t1", api_mode="chat_completions",
            api_call_count=1, platform="cli", provider="nous", model="m",
        )
        assert response.choices[0].message.content == "2 + 2 = 4"


# ---- AIAgent loop integration ----

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _build_agent(hermes_home, api_mode):
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="sk-dummy-not-used",
        base_url="https://example.invalid/v1",
        api_mode=api_mode,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent._disable_streaming = True
    return agent


def _attach_runtime(agent, *, settings_mode, route):
    """Append a FastPathRuntime middleware to the manager for the active Hermes home.

    ``route`` picks the injected classifier outcome: ``"fastpath"`` -> accepted calculator
    decision, ``"normal_llm"`` -> Jev says the real LLM must answer.
    """
    from hermes_cli.plugins import get_plugin_manager
    from jev_fastpath.config import Settings
    from jev_fastpath.plugin import FastPathRuntime
    from jev_fastpath.types import Decision

    class Recorder:
        def __init__(self):
            self.events = []

        def write(self, event, context):
            self.events.append(event)

    def classifier(text, candidates, context, settings):
        classifier.calls.append((text, candidates))
        if route == "fastpath":
            return Decision(
                handler_id="calculator", confidence=0.97,
                short_circuit_probability=0.95, latency_ms=1,
            )
        return Decision(
            handler_id="normal_llm", confidence=0.97,
            short_circuit_probability=0.95, latency_ms=1,
        )

    classifier.calls = []

    runtime = FastPathRuntime(
        Settings(mode=settings_mode), classifier=classifier, telemetry=Recorder(),
    )
    manager = get_plugin_manager()
    manager._middleware.setdefault("llm_execution", []).append(runtime.middleware)
    return runtime


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_agent_loop_fast_path_completes_without_provider(hermes_home, monkeypatch, api_mode):
    agent = _build_agent(hermes_home, api_mode)
    _attach_runtime(agent, settings_mode="active", route="fastpath")
    monkeypatch.setattr(
        agent, "_interruptible_api_call",
        lambda api_kwargs: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    result = agent.run_conversation("2 + 2")

    assert result["completed"] is True
    assert result["final_response"] == "2 + 2 = 4"
    last = result["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == "2 + 2 = 4"


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_agent_loop_normal_llm_fallthrough_calls_provider_once(hermes_home, monkeypatch, api_mode):
    agent = _build_agent(hermes_home, api_mode)
    _attach_runtime(agent, settings_mode="active", route="normal_llm")
    calls = []

    def fake_provider(api_kwargs):
        calls.append(api_kwargs)
        if api_mode == "chat_completions":
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="the llm answer", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
                model="test-model",
            )
        return types.SimpleNamespace(
            output=[types.SimpleNamespace(
                type="message", role="assistant", status="completed",
                content=[types.SimpleNamespace(type="output_text", text="the llm answer")],
            )],
            usage=None,
            status="completed",
            model="test-model",
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_provider)

    result = agent.run_conversation("fale sobre arte moderna")

    assert result["completed"] is True
    assert result["final_response"] == "the llm answer"
    assert len(calls) == 1


def _invalid_provider_response(api_mode):
    if api_mode == "chat_completions":
        return types.SimpleNamespace(choices=[], usage=None, model="test-model")
    return types.SimpleNamespace(output=[], status="completed", usage=None, model="test-model")


def _provider_text_response(api_mode):
    if api_mode == "chat_completions":
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="the llm answer", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
            model="test-model",
        )
    return types.SimpleNamespace(
        output=[types.SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[types.SimpleNamespace(type="output_text", text="the llm answer")],
        )],
        usage=None,
        status="completed",
        model="test-model",
    )


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_agent_loop_retry_after_invalid_response_never_short_circuits(
    hermes_home, monkeypatch, api_mode,
):
    """H1 regression: invalid provider response -> retry. Jev runs at most once, exactly
    one telemetry row exists, and the retry is never short-circuited."""
    import time as time_module

    agent = _build_agent(hermes_home, api_mode)
    runtime = _attach_runtime(agent, settings_mode="active", route="normal_llm")
    provider_calls = []

    def fake_provider(api_kwargs):
        provider_calls.append(api_kwargs)
        if len(provider_calls) == 1:
            return _invalid_provider_response(api_mode)
        return _provider_text_response(api_mode)

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_provider)
    # Kill retry backoff so the test does not sleep on wall clock.
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(time_module, "sleep", lambda *_a, **_k: None)

    result = agent.run_conversation("2 + 2")

    assert result["completed"] is True
    assert result["final_response"] == "the llm answer"
    assert len(provider_calls) == 2          # invalid attempt + retry reached the provider
    assert len(runtime.classifier.calls) == 1  # Jev evaluated at most once for the turn
    assert len(runtime.telemetry.events) == 1  # exactly one telemetry row, no duplicates


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_agent_loop_retry_after_invalid_synthetic_reaches_provider(
    hermes_home, monkeypatch, api_mode,
):
    """H1 regression, accepted decision: a synthetic response Hermes rejects must not be
    re-rendered for the retry — the per-turn claim sends the retry to the provider."""
    import time as time_module

    agent = _build_agent(hermes_home, api_mode)
    runtime = _attach_runtime(agent, settings_mode="active", route="fastpath")
    factory_calls = []

    def flaky_factory(api_mode_, text):
        factory_calls.append((api_mode_, text))
        if len(factory_calls) == 1:
            return _invalid_provider_response(api_mode)  # synthetic Hermes must reject
        return _provider_text_response(api_mode)

    runtime.response_factory = flaky_factory
    provider_calls = []

    def fake_provider(api_kwargs):
        provider_calls.append(api_kwargs)
        return _provider_text_response(api_mode)

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_provider)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(time_module, "sleep", lambda *_a, **_k: None)

    result = agent.run_conversation("2 + 2")

    assert result["completed"] is True
    assert result["final_response"] == "the llm answer"
    assert len(provider_calls) == 1           # the retry reached the real provider
    assert len(runtime.classifier.calls) == 1  # Jev evaluated at most once for the turn
    assert len(runtime.telemetry.events) == 1  # no duplicate telemetry rows
    assert len(factory_calls) == 1            # the synthetic was never re-rendered
