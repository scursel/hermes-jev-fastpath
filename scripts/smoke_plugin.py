#!/usr/bin/env python3
"""Credential-free plugin smoke: manifest parse, profile-scoped discovery, middleware.

Runs the full local pipeline against an isolated throwaway HERMES_HOME with no
``TYPESAFE_API_KEY`` and no network: the classifier is injected, so shadow mode must
produce exactly one downstream provider call and active mode exactly zero.

Usage:
    PYTHONPATH=/path/to/hermes-agent python scripts/smoke_plugin.py [--mode shadow|active]

Prints one JSON object: {"manifest_ok", "discovery_ok", "middleware_ok", "provider_calls"}
and exits nonzero unless every check passes for the selected mode.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_CALLS = {"shadow": 1, "active": 0}


def _parse_manifest() -> bool:
    from hermes_cli.plugins import parse_manifest_file

    manifest = parse_manifest_file(REPO_ROOT / "plugin.yaml", REPO_ROOT, "smoke", "")
    ok = (
        manifest is not None
        and manifest.name == "jev-fastpath"
        and manifest.kind == "standalone"
        and int(manifest.manifest_version) == 2
    )
    print(f"manifest: {manifest.name} v{manifest.version}" if manifest else "manifest: PARSE FAILED", file=sys.stderr)
    return bool(ok)


def _install_plugin(home: Path) -> None:
    destination = home / "plugins" / "jev-fastpath"
    destination.mkdir(parents=True)
    for name in ("__init__.py", "plugin.yaml"):
        shutil.copy2(REPO_ROOT / name, destination / name)
    shutil.copytree(REPO_ROOT / "jev_fastpath", destination / "jev_fastpath")


def _write_config(home: Path, mode: str) -> None:
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - jev-fastpath\n"
        "  entries:\n"
        "    jev-fastpath:\n"
        "      enabled: true\n"
        "      settings:\n"
        f"        mode: {mode}\n",
        encoding="utf-8",
    )


def _discover(home: Path):
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager.discover_and_load()
    loaded = manager._plugins.get("jev-fastpath")
    if loaded is None or not loaded.enabled or loaded.error:
        return None
    return manager


def _run_middleware(manager, mode: str) -> tuple[bool, int]:
    """Invoke the discovered middleware with an injected classifier; count provider calls."""
    from hermes_cli.middleware import run_llm_execution_middleware
    from jev_fastpath.types import Decision

    callback = manager._middleware["llm_execution"][0]
    runtime = callback.__self__
    runtime.classifier = lambda *args: Decision(
        handler_id="calculator", confidence=0.97, short_circuit_probability=0.95, latency_ms=1,
    )

    request = {"model": "smoke", "messages": [{"role": "user", "content": "2 + 2"}]}
    provider_calls = 0

    def next_call(received):
        nonlocal provider_calls
        provider_calls += 1
        return {"provider": "response"}

    response = run_llm_execution_middleware(
        request, next_call,
        session_id="smoke-session", turn_id="smoke-turn", api_mode="chat_completions",
        api_call_count=1, platform="cli", provider="smoke", model="smoke",
    )
    ok = provider_calls == EXPECTED_CALLS[mode]
    if mode == "active":
        ok = ok and getattr(response, "choices", None) is not None
    else:
        ok = ok and response == {"provider": "response"}
    return bool(ok), provider_calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(EXPECTED_CALLS), default="shadow")
    args = parser.parse_args()

    result = {"manifest_ok": False, "discovery_ok": False, "middleware_ok": False, "provider_calls": 0}
    hermes_cli_ready = False
    try:
        import hermes_cli.plugins  # noqa: F401

        hermes_cli_ready = True
    except ImportError as exc:
        print(f"Hermes runtime not importable: {exc}", file=sys.stderr)
        print("Run with PYTHONPATH pointing at a Hermes Agent checkout, or install hermes-agent.", file=sys.stderr)

    if hermes_cli_ready:
        result["manifest_ok"] = _parse_manifest()
        home = Path(tempfile.mkdtemp(prefix="jev-fastpath-smoke-"))
        bundled = home / "bundled-plugins"
        bundled.mkdir()
        previous = {
            name: os.environ.get(name)
            for name in ("HERMES_HOME", "HERMES_BUNDLED_PLUGINS", "HERMES_ENABLE_PROJECT_PLUGINS", "TYPESAFE_API_KEY")
        }
        try:
            os.environ["HERMES_HOME"] = str(home)
            os.environ["HERMES_BUNDLED_PLUGINS"] = str(bundled)
            os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] = "0"
            os.environ.pop("TYPESAFE_API_KEY", None)
            _install_plugin(home)
            _write_config(home, args.mode)
            manager = _discover(home)
            if manager is not None:
                result["discovery_ok"] = True
                result["middleware_ok"], result["provider_calls"] = _run_middleware(manager, args.mode)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            shutil.rmtree(home, ignore_errors=True)

    print(json.dumps(result))
    expected = EXPECTED_CALLS[args.mode]
    success = (
        result["manifest_ok"] and result["discovery_ok"] and result["middleware_ok"]
        and result["provider_calls"] == expected
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
