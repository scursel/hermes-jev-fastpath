# Hermes Jev Fast Path v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a profile-scoped Hermes plugin that uses TypeSafe Jev to identify allowlisted deterministic turns and complete them without calling the configured LLM provider.

**Architecture:** Register one `llm_execution` middleware callback and leave Hermes core untouched. A local candidate detector narrows the possible deterministic handlers, Jev returns a typed choice plus short-circuit probability, local code validates and renders the selected handler, and a protocol adapter returns a synthetic provider response; every unsupported or failed path calls the original provider exactly once.

**Tech Stack:** Python 3.11+, Hermes Agent 0.21.3 plugin API, Python standard library (`urllib`, `ast`, `zoneinfo`, `dataclasses`, `types`, `threading`), pytest, PyYAML for manifest tests, GitHub Actions.

**Spec:** `docs/specs/jev-fastpath-v1.md`

**Authoritative Hermes contracts:**
- Plugin authoring: `https://hermes-agent.nousresearch.com/docs/developer-guide/plugins`
- Middleware: `https://hermes-agent.nousresearch.com/docs/developer-guide/middleware`
- Plugin operation and enablement: `https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins`

The implementation relies on the documented `llm_execution` short-circuit contract: execution middleware may intentionally skip `next_call(...)`, but it must return the unwrapped raw response shape expected by the selected provider adapter. Middleware callbacks accept `**kwargs` for additive compatibility. Validation must use `hermes plugins doctor . --ci`, which exercises real discovery, namespaced import, and `register(ctx)` rather than only parsing the manifest.

## Global Constraints

- The deliverable is a standalone Hermes plugin; do not modify `/home/scursel/.hermes/hermes-agent` or any live profile.
- Register only `llm_execution` middleware; do not register approval, gateway-blocking, or tool-blocking hooks.
- Jev is a typed router, never a source of shell, tool arguments, URLs, code, or response prose.
- Every failure and every uncertainty falls through to `next_call(request)` exactly once.
- An accepted active fast path calls `next_call` zero times.
- Only the first provider attempt of a turn is eligible; tool rounds, retries, fallback calls, and continuations bypass the plugin.
- Preserve prompt caching: never mutate request messages, system prompt, history, or tools.
- Credentials come only from `TYPESAFE_API_KEY`; never read another file or profile for secrets.
- Runtime code remains dependency-free outside Hermes and Python's standard library.
- Default plugin mode is `shadow`; active mode is an explicit profile setting.
- Development is performed by CommandCode using model `z-ai/glm-5.3-flash`; Claude Code performs a read-only final audit after implementation and tests are complete.

## Review Focus

- A context-dependent short utterance such as `sim`, `continue`, or `faça` must not be mistaken for a self-contained acknowledgement.
- A malicious or pathological arithmetic expression must reject before evaluation and must never allocate unbounded integers.
- A Jev timeout or malformed probability must add only bounded latency and must call the real provider once, not zero or twice.
- Synthetic responses must normalize correctly for `chat_completions`, `codex_responses`, `anthropic_messages`, and `bedrock_converse` without fake provider token usage.
- Concurrent turns sharing a session must never reuse a decision unless session ID, turn ID, and text hash all match.

---

## File Structure

Create these files and keep responsibilities separated:

```text
hermes-jev-fastpath/
├── __init__.py                         # Hermes directory-plugin entrypoint; re-exports register
├── plugin.yaml                         # Installer-compatible manifest v1 and config schema
├── pyproject.toml                      # Test/build metadata; no runtime dependencies
├── README.md                           # Install, configure, shadow, activate, verify, rollback
├── jev_fastpath/
│   ├── __init__.py                     # Package exports
│   ├── config.py                       # Typed settings parsing and validation
│   ├── types.py                        # Decision, candidate, render, and telemetry dataclasses
│   ├── input.py                        # User-text extraction, normalization, and rejection gates
│   ├── candidates.py                   # Conservative candidate detection
│   ├── handlers.py                     # Allowlisted deterministic handler registry
│   ├── arithmetic.py                   # Bounded safe AST calculator
│   ├── jev.py                          # TypeSafe HTTP client and strict response parser
│   ├── cache.py                        # Turn-identity TTL cache
│   ├── responses.py                    # Raw synthetic response adapters by Hermes api_mode
│   ├── telemetry.py                    # Redacted bounded JSONL audit writer
│   └── plugin.py                       # Middleware orchestration and register(ctx)
├── scripts/
│   └── smoke_plugin.py                 # Credential-free plugin discovery and middleware smoke
├── tests/
│   ├── conftest.py                     # Isolated HERMES_HOME and import helpers
│   ├── test_config.py
│   ├── test_input_candidates.py
│   ├── test_arithmetic.py
│   ├── test_handlers.py
│   ├── test_jev.py
│   ├── test_cache.py
│   ├── test_responses.py
│   ├── test_telemetry.py
│   ├── test_plugin_middleware.py
│   └── test_hermes_integration.py
└── .github/workflows/test.yml           # compile, unit/integration tests, SkillSpector when present
```

No file may import from a sibling by modifying `sys.path`; use package-relative imports.

### Task 1: Scaffold the plugin contract and typed configuration

**Files:**
- Create: `__init__.py`
- Create: `plugin.yaml`
- Create: `pyproject.toml`
- Create: `jev_fastpath/__init__.py`
- Create: `jev_fastpath/config.py`
- Create: `jev_fastpath/types.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: Hermes `PluginContext.get_config(key, default)` and `register_middleware(kind, callback)` contracts.
- Produces: `Settings`, `Decision`, `HandlerResult`, `TelemetryEvent`, and root `register(ctx)` entrypoint used by every later task.

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_config.py` with explicit tests for defaults, valid overrides, invalid enums, non-finite thresholds, timeout bounds, timezone validation, duplicate/unknown handlers, and wrong scalar types:

```python
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
    assert "calculator" in settings.enabled_handlers


@pytest.mark.parametrize("value", ["enabled", "yes", 1, None])
def test_invalid_mode_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"mode": value}))


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, "0.9"])
def test_invalid_confidence_rejected(value):
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"confidence_threshold": value}))


def test_unknown_and_duplicate_handlers_rejected():
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": ["clock", "clock"]}))
    with pytest.raises(SettingsError):
        load_settings(FakeContext({"enabled_handlers": ["shell"]}))
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: collection fails because `jev_fastpath.config` does not exist.

- [ ] **Step 3: Add typed domain objects**

Create `jev_fastpath/types.py` with immutable dataclasses and literal outcome vocabulary:

```python
from dataclasses import dataclass, field
from typing import Any, Literal

HandlerId = Literal[
    "calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status"
]
Outcome = Literal[
    "no_candidate", "normal_llm", "would_short_circuit", "short_circuit", "fallback"
]


@dataclass(frozen=True)
class Decision:
    handler_id: str
    confidence: float
    short_circuit_probability: float
    latency_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


@dataclass(frozen=True)
class HandlerResult:
    handler_id: str
    text: str


@dataclass(frozen=True)
class TelemetryEvent:
    outcome: Outcome
    reason: str
    text: str
    candidates: tuple[str, ...]
    selected_handler: str = ""
    confidence: float | None = None
    short_circuit_probability: float | None = None
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Implement strict settings parsing**

Create `jev_fastpath/config.py` with this public contract:

```python
from dataclasses import dataclass
from math import isfinite
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

KNOWN_HANDLERS = (
    "calculator", "clock", "runtime_identity", "acknowledgement", "fastpath_status"
)


class SettingsError(ValueError):
    pass


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
    defaults = Settings()
    mode = ctx.get_config("mode", defaults.mode)
    locale = ctx.get_config("locale", defaults.locale)
    timezone = ctx.get_config("timezone", defaults.timezone)
    handlers = ctx.get_config("enabled_handlers", list(defaults.enabled_handlers))
    if mode not in {"off", "shadow", "active"}:
        raise SettingsError("mode must be off, shadow, or active")
    if locale not in {"pt-BR", "en"}:
        raise SettingsError("locale must be pt-BR or en")
    if not isinstance(timezone, str) or not timezone.strip():
        raise SettingsError("timezone must be a non-empty IANA name")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
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
```

Implement this validation without coercing strings to numbers or booleans to integers.

- [ ] **Step 5: Add the manifest and package entrypoint**

Create root `__init__.py`:

```python
from .jev_fastpath.plugin import register

__all__ = ["register"]
```

Create `plugin.yaml` using installer-compatible manifest v1 (Hermes 0.21.3's runtime parser reads v2, but its public Git installer still caps at v1):

```yaml
name: jev-fastpath
version: "0.1.0"
description: "TypeSafe Jev deterministic fast paths before Hermes LLM execution"
author: "Gabriel Scursel"
license: MIT
homepage: "https://github.com/scursel/hermes-jev-fastpath"
manifest_version: 1
api_version: 1
kind: standalone
provides_tools: []
provides_hooks: []
provides_middleware:
  - llm_execution
requires_env:
  - TYPESAFE_API_KEY
python_dependencies: []
tags:
  - jev
  - fast-path
  - token-savings
config_schema:
  mode: {type: str, default: shadow, description: "off, shadow, or active"}
  confidence_threshold: {type: float, default: 0.92, description: "Minimum Jev choice confidence"}
  short_circuit_threshold: {type: float, default: 0.90, description: "Minimum Jev short-circuit probability"}
  timeout_seconds: {type: float, default: 3.0, description: "TypeSafe request timeout"}
  timezone: {type: str, default: America/Sao_Paulo, description: "IANA timezone for clock handler"}
  locale: {type: str, default: pt-BR, description: "pt-BR or en"}
  max_input_chars: {type: int, default: 4000, description: "Maximum candidate input length"}
  enabled_handlers: {type: list, default: [calculator, clock, runtime_identity, acknowledgement, fastpath_status], description: "Allowlisted deterministic handlers"}
```

Create `pyproject.toml` with Python `>=3.11`, package discovery for `jev_fastpath*`, and a `test` optional dependency containing `pytest>=8,<9` and `PyYAML>=6,<7`.

- [ ] **Step 6: Run the tests and manifest parse smoke**

Run:

```bash
python -m pytest tests/test_config.py -q
PYTHONPATH=/home/scursel/.hermes/hermes-agent \
  /home/scursel/.hermes/hermes-agent/venv/bin/python -c \
  "from pathlib import Path; from hermes_cli.plugins import parse_manifest_file; m=parse_manifest_file(Path('plugin.yaml')); assert m.name == 'jev-fastpath'; assert m.provides_middleware == ['llm_execution']"
HERMES_HOME="$(mktemp -d)" \
  /home/scursel/.hermes/hermes-agent/venv/bin/hermes plugins doctor . --ci
```

Expected: all config tests pass, manifest parsing exits 0, and Plugin Doctor reports successful discovery, namespaced import, and middleware registration.

- [ ] **Step 7: Commit the scaffold**

```bash
git add __init__.py plugin.yaml pyproject.toml jev_fastpath tests/conftest.py tests/test_config.py
git commit -m "feat: scaffold Jev fast-path plugin"
```

### Task 2: Build safe input extraction, candidate detection, and deterministic handlers

**Files:**
- Create: `jev_fastpath/input.py`
- Create: `jev_fastpath/candidates.py`
- Create: `jev_fastpath/arithmetic.py`
- Create: `jev_fastpath/handlers.py`
- Create: `tests/test_input_candidates.py`
- Create: `tests/test_arithmetic.py`
- Create: `tests/test_handlers.py`

**Interfaces:**
- Consumes: provider request dictionaries and middleware context from Task 1.
- Produces: `extract_latest_user_text(request, api_mode) -> str | None`, `detect_candidates(text, settings) -> tuple[str, ...]`, and `render_handler(handler_id, text, context, settings, status) -> HandlerResult`.

- [ ] **Step 1: Write failing input and candidate tests**

Cover Chat Completions `messages`, Responses `input` message items, Anthropic `messages`, malformed/multimodal content, slash commands, long text, code fences, URLs, credentials, shell syntax, exact acknowledgements, arithmetic, clock, runtime identity, and plugin status.

Required assertions include:

```python
def test_context_dependent_short_reply_has_no_candidate(settings):
    assert detect_candidates("sim", settings) == ()
    assert detect_candidates("continue", settings) == ()
    assert detect_candidates("faça", settings) == ()


def test_exact_acknowledgement_is_candidate(settings):
    assert detect_candidates("obrigado", settings) == ("acknowledgement",)
    assert detect_candidates("obrigado, agora apague o arquivo", settings) == ()


def test_dangerous_shapes_bypass_jev(settings):
    for text in ("curl https://example.com", "token=ts_SECRET_VALUE", "```python\n2+2\n```", "rm -rf /tmp/x"):
        assert detect_candidates(text, settings) == ()
```

- [ ] **Step 2: Run focused tests and verify failure**

```bash
python -m pytest tests/test_input_candidates.py tests/test_arithmetic.py tests/test_handlers.py -q
```

Expected: imports fail because the modules do not exist.

- [ ] **Step 3: Implement bounded arithmetic parsing**

Create `jev_fastpath/arithmetic.py` with these exact public interfaces:

```python
class ArithmeticRejected(ValueError):
    pass


def extract_expression(text: str) -> str:
    """Remove only an allowlisted Portuguese/English calculator prefix and one terminal question mark."""


def evaluate_expression(expression: str) -> int | float:
    """Parse with ast.parse(..., mode='eval'), validate every node, then evaluate recursively."""


def render_calculation(text: str) -> str:
    expression = extract_expression(text)
    value = evaluate_expression(expression)
    return f"{expression} = {format_number(value)}"
```

Use an explicit operator table:

```python
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
```

Count nodes before evaluation, reject `bool` constants, cap integer digits and exponent, and reject non-finite float results.

- [ ] **Step 4: Implement input extraction and conservative candidates**

Create `jev_fastpath/input.py` with protocol-specific extraction that returns plain text only and never flattens image/audio parts. Create `jev_fastpath/candidates.py` with anchored regexes and parser probes. The detector may include multiple candidates only when both local parsers validate; preserve the fixed `KNOWN_HANDLERS` order.

Before matching, reject text containing:

```python
_REJECT_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"https?://", re.I),
    re.compile(r"(?i)\b(?:token|password|passwd|secret|api[_-]?key)\s*[:=]"),
    re.compile(r"(?:&&|\|\||;|`|\$\(|>|<)"),
    re.compile(r"(?i)\b(?:delete|remove|apague|exclua|envie|publique|compre|pague|reinicie|execute)\b"),
)
```

Keep request text immutable; tests must compare a deep copy before and after extraction.

- [ ] **Step 5: Implement the handler registry**

Create `jev_fastpath/handlers.py` with a fixed mapping, not dynamic imports:

```python
Handler = Callable[[str, Mapping[str, Any], Settings, Mapping[str, Any]], str]
HANDLERS: dict[str, Handler] = {
    "calculator": _calculator,
    "clock": _clock,
    "runtime_identity": _runtime_identity,
    "acknowledgement": _acknowledgement,
    "fastpath_status": _fastpath_status,
}


def render_handler(handler_id, text, context, settings, status) -> HandlerResult:
    if handler_id not in settings.enabled_handlers or handler_id not in HANDLERS:
        raise HandlerRejected(f"handler not enabled: {handler_id}")
    rendered = HANDLERS[handler_id](text, context, settings, status).strip()
    if not rendered or len(rendered) > 2000:
        raise HandlerRejected("handler returned invalid text")
    return HandlerResult(handler_id=handler_id, text=rendered)
```

Use `ZoneInfo(settings.timezone)` and an injectable `now_fn` in status/context for deterministic clock tests. Runtime identity reads only `provider`, `model`, `api_mode`, and `platform` from middleware context. Plugin status reads only validated settings and in-memory status.

- [ ] **Step 6: Run focused tests**

```bash
python -m pytest tests/test_input_candidates.py tests/test_arithmetic.py tests/test_handlers.py -q
```

Expected: all tests pass, including malicious arithmetic and contextual short-reply cases.

- [ ] **Step 7: Commit deterministic handlers**

```bash
git add jev_fastpath/input.py jev_fastpath/candidates.py jev_fastpath/arithmetic.py jev_fastpath/handlers.py tests/test_input_candidates.py tests/test_arithmetic.py tests/test_handlers.py
git commit -m "feat: add allowlisted deterministic handlers"
```

### Task 3: Implement the strict Jev client and turn-scoped decision cache

**Files:**
- Create: `jev_fastpath/jev.py`
- Create: `jev_fastpath/cache.py`
- Create: `tests/test_jev.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Consumes: bounded text, candidate tuple, middleware context, `Settings`, and `TYPESAFE_API_KEY`.
- Produces: `classify(text, candidates, context, settings) -> Decision` and `DecisionCache.get/put` keyed by session, turn, and text hash.

- [ ] **Step 1: Write failing Jev parser/client tests**

Use an injected opener so default tests never use network. Cover exact payload, auth header redaction, timeout value, one attempt, HTTP error, invalid JSON, missing `answers`, unknown handler, `normal_llm`, missing confidence, bool-as-number, NaN/infinity, out-of-range values, and usage preservation.

Core parser tests:

```python
@pytest.mark.parametrize("bad", [None, True, "0.95", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_rejected(bad):
    payload = valid_response()
    payload["answers"]["handler"]["confidence"] = bad
    with pytest.raises(JevError):
        parse_response(payload, ("calculator",))


def test_unknown_handler_rejected():
    payload = valid_response(choice="terminal")
    with pytest.raises(JevError):
        parse_response(payload, ("calculator",))
```

- [ ] **Step 2: Write failing cache tests**

Use an injectable monotonic clock. Assert exact key isolation, text hash mismatch, expiry at 900 seconds, bounded entry count, and concurrent get/put under a thread pool.

- [ ] **Step 3: Run tests and verify failure**

```bash
python -m pytest tests/test_jev.py tests/test_cache.py -q
```

Expected: imports fail because the modules do not exist.

- [ ] **Step 4: Implement strict Jev request construction and parsing**

Create `jev_fastpath/jev.py` with constants and functions:

```python
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

class JevError(RuntimeError):
    pass


def build_questions(candidates: tuple[str, ...]) -> dict:
    criteria = {handler_id: HANDLER_CRITERIA[handler_id] for handler_id in candidates}
    criteria["normal_llm"] = "Any context, reasoning, tools, writing, interpretation, or unsupported data is needed."
    return {
        "handler": {"type": "choice", "instructions": "Choose exactly one deterministic handler, otherwise normal_llm.", "criteria": criteria},
        "safe_to_short_circuit": {"type": "noul", "instructions": "Probability that the chosen deterministic result fully satisfies this turn without tools or an LLM.", "criteria": {"true": "The handler completely answers the literal request.", "false": "Any context, tool, judgment, ambiguity, side effect, or generative work is needed."}},
    }


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError(f"{field} is not a numeric probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JevError(f"{field} is outside [0, 1]")
    return result


def parse_response(payload: object, candidates: tuple[str, ...], latency_ms: int) -> Decision:
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise JevError("response has no typed answers")
    answers = payload["answers"]
    handler = answers.get("handler")
    short = answers.get("safe_to_short_circuit")
    if not isinstance(handler, dict) or not isinstance(short, dict):
        raise JevError("response is missing handler or short-circuit answer")
    choice = handler.get("choice")
    allowed = set(candidates) | {"normal_llm"}
    if not isinstance(choice, str) or choice not in allowed:
        raise JevError("handler choice is not allowed for this request")
    return Decision(
        handler_id=choice,
        confidence=_probability(handler.get("confidence"), "handler.confidence"),
        short_circuit_probability=_probability(
            short.get("noul"), "safe_to_short_circuit.noul"
        ),
        latency_ms=latency_ms,
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
    )


def classify(text, candidates, context, settings, *, opener=urllib.request.urlopen) -> Decision:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise JevError("TYPESAFE_API_KEY is unavailable")
    state = {
        "message": redact_and_bound(text, settings.max_input_chars),
        "platform": str(context.get("platform") or ""),
        "candidate_handlers": list(candidates),
    }
    body = json.dumps({
        "model": JEV_MODEL,
        "state": state,
        "questions": build_questions(candidates),
    }).encode("utf-8")
    request = urllib.request.Request(
        TYPESAFE_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-jev-fastpath/0.1",
        },
    )
    started = time.monotonic()
    try:
        with opener(request, timeout=settings.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise JevError(f"TypeSafe HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JevError(f"TypeSafe network failure: {type(exc).__name__}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JevError("TypeSafe returned invalid JSON") from exc
    latency_ms = round((time.monotonic() - started) * 1000)
    return parse_response(payload, candidates, latency_ms)
```

The implementation must perform exactly one network attempt. Never include a raw response body, authorization value, or full exception string in `JevError`.

- [ ] **Step 5: Implement the cache**

Create `jev_fastpath/cache.py`:

```python
@dataclass(frozen=True)
class CacheKey:
    session_id: str
    turn_id: str
    text_hash: str


class DecisionCache:
    def __init__(self, ttl_seconds=900.0, max_entries=1024, monotonic=time.monotonic):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._entries: OrderedDict[CacheKey, tuple[float, Decision]] = OrderedDict()

    @staticmethod
    def key(session_id: str, turn_id: str, text: str) -> CacheKey:
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        return CacheKey(str(session_id), str(turn_id), digest)
```

`get` must prune expired entries and promote a hit. `put` must prune, replace exact keys, and evict oldest entries until bounded. Empty session or turn IDs must not be cacheable.

- [ ] **Step 6: Run focused tests**

```bash
python -m pytest tests/test_jev.py tests/test_cache.py -q
```

Expected: all tests pass with no network access.

- [ ] **Step 7: Commit Jev and cache**

```bash
git add jev_fastpath/jev.py jev_fastpath/cache.py tests/test_jev.py tests/test_cache.py
git commit -m "feat: add typed Jev routing and turn cache"
```

### Task 4: Build provider-compatible synthetic responses

**Files:**
- Create: `jev_fastpath/responses.py`
- Create: `tests/test_responses.py`

**Interfaces:**
- Consumes: deterministic response text and Hermes `api_mode`.
- Produces: a raw response object accepted by the corresponding Hermes transport's `validate_response` and `normalize_response` methods.

- [ ] **Step 1: Write failing response adapter tests**

For each supported API mode, assert the raw shape, absence of fake token usage, and normalized content/finish reason using the real Hermes transport classes when available on `PYTHONPATH`.

```python
@pytest.mark.parametrize("api_mode", [
    "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"
])
def test_synthetic_response_normalizes(api_mode, hermes_transport):
    raw = build_synthetic_response(api_mode, "resultado")
    transport = hermes_transport(api_mode)
    assert transport.validate_response(raw)
    normalized = transport.normalize_response(raw)
    assert normalized.content == "resultado"
    assert normalized.finish_reason == "stop"
    assert normalized.tool_calls in (None, [])
    assert normalized.usage is None


def test_unknown_api_mode_rejected():
    with pytest.raises(UnsupportedApiMode):
        build_synthetic_response("future_protocol", "x")
```

- [ ] **Step 2: Run the focused test and verify failure**

```bash
PYTHONPATH=/home/scursel/.hermes/hermes-agent python -m pytest tests/test_responses.py -q
```

Expected: import failure because `jev_fastpath.responses` does not exist.

- [ ] **Step 3: Implement explicit response factories**

Create `jev_fastpath/responses.py` using `types.SimpleNamespace` and exact raw protocol fields:

```python
class UnsupportedApiMode(ValueError):
    pass


def _chat(text: str):
    return SimpleNamespace(
        id="jev-fastpath",
        model="jev-fastpath",
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role="assistant", content=text, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )


def _codex(text: str):
    return SimpleNamespace(
        id="jev-fastpath",
        model="jev-fastpath",
        output=[SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=text)],
        )],
        status="completed",
        usage=None,
    )


def _anthropic(text: str):
    return SimpleNamespace(
        id="jev-fastpath",
        model="jev-fastpath",
        role="assistant",
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=None,
    )


def _bedrock(text: str):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }


_FACTORIES = {
    "chat_completions": _chat,
    "codex_responses": _codex,
    "anthropic_messages": _anthropic,
    "bedrock_converse": _bedrock,
}


def build_synthetic_response(api_mode: str, text: str):
    factory = _FACTORIES.get(api_mode)
    if factory is None:
        raise UnsupportedApiMode(api_mode)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("synthetic response text must be non-empty")
    return factory(text)
```

Run the response tests against the real Hermes transports and adjust only protocol field names proven by those transports; do not use reflection or dynamic imports.

- [ ] **Step 4: Run response tests across real Hermes transports**

```bash
PYTHONPATH=/home/scursel/.hermes/hermes-agent \
  /home/scursel/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_responses.py -q
```

Expected: every supported mode validates and normalizes to `resultado` with finish reason `stop`.

- [ ] **Step 5: Commit response adapters**

```bash
git add jev_fastpath/responses.py tests/test_responses.py
git commit -m "feat: add synthetic provider response adapters"
```

### Task 5: Add redacted telemetry and the execution middleware

**Files:**
- Create: `jev_fastpath/telemetry.py`
- Create: `jev_fastpath/plugin.py`
- Create: `tests/test_telemetry.py`
- Create: `tests/test_plugin_middleware.py`
- Modify: `jev_fastpath/__init__.py`

**Interfaces:**
- Consumes: Tasks 1–4 modules and Hermes middleware kwargs.
- Produces: `FastPathRuntime.middleware(...)`, `FastPathRuntime.status()`, and `register(ctx)`.

- [ ] **Step 1: Write failing telemetry tests**

Assert recursive redaction, bounded previews, HERMES_HOME-scoped path, one JSON object per line, concurrent appends, and swallowed I/O errors. Secret canaries must not appear anywhere in serialized records.

- [ ] **Step 2: Write failing middleware matrix tests**

Use a counting `next_call` and injected classifier/clock. Cover:

- mode `off` bypass;
- `api_call_count >= 2` bypass (attempts are counted 1-based: the first attempt arrives as
  `api_call_count == 1`; a re-delivered first attempt is caught by the per-turn claim);
- missing session or turn identity without cache corruption;
- unsupported API mode;
- no candidate without Jev call;
- `normal_llm` choice;
- low choice confidence;
- low short-circuit probability;
- Jev timeout/error;
- handler rejection;
- response factory rejection;
- shadow accepted decision returning the exact downstream response;
- active accepted decision not calling downstream;
- exact decision cache reuse;
- concurrent distinct turns;
- middleware internal exception still calling downstream once.

Required call-count assertions:

```python
def test_active_acceptance_skips_provider(runtime, request, context):
    downstream = Mock(side_effect=AssertionError("provider must not run"))
    response = runtime.middleware(request=request, next_call=downstream, **context)
    downstream.assert_not_called()
    assert response.choices[0].message.content == "2 + 2 = 4"


def test_failure_calls_provider_exactly_once(runtime, request, context):
    runtime.classifier = Mock(side_effect=TimeoutError("slow"))
    downstream_response = object()
    downstream = Mock(return_value=downstream_response)
    assert runtime.middleware(request=request, next_call=downstream, **context) is downstream_response
    downstream.assert_called_once_with(request)
```

- [ ] **Step 3: Run focused tests and verify failure**

```bash
python -m pytest tests/test_telemetry.py tests/test_plugin_middleware.py -q
```

Expected: imports fail because telemetry and plugin runtime do not exist.

- [ ] **Step 4: Implement telemetry**

Create `jev_fastpath/telemetry.py` with:

```python
def redact(value: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


class TelemetryWriter:
    def __init__(self, home: Path, max_preview=160):
        self.path = home / "artifacts" / "jev_fastpath" / "decisions.jsonl"
        self.max_preview = int(max_preview)
        self._lock = threading.Lock()

    def write(self, event: TelemetryEvent, context: Mapping[str, Any]) -> None:
        def digest(value: object) -> str:
            return hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()[:16]

        usage = {
            key: value
            for key in ("cost", "input_tokens", "output_tokens")
            if isinstance((value := event.usage.get(key)), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": str(context.get("mode") or ""),
            "session_hash": digest(context.get("session_id")),
            "turn_hash": digest(context.get("turn_id")),
            "platform": str(context.get("platform") or "")[:64],
            "provider": str(context.get("provider") or "")[:128],
            "model": str(context.get("model") or "")[:256],
            "api_mode": str(context.get("api_mode") or "")[:64],
            "text_hash": digest(event.text),
            "text_preview": redact(event.text)[:self.max_preview],
            "candidates": list(event.candidates),
            "selected_handler": event.selected_handler,
            "confidence": event.confidence,
            "short_circuit_probability": event.short_circuit_probability,
            "latency_ms": event.latency_ms,
            "outcome": event.outcome,
            "reason": redact(event.reason)[:160],
            "provider_call_avoided": event.outcome == "short_circuit",
            "usage": usage,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            logger.debug("jev-fastpath telemetry write failed", exc_info=True)
```

Serialize only the fixed fields above. Nested/raw provider payloads are discarded.

- [ ] **Step 5: Implement the runtime state machine**

Create `jev_fastpath/plugin.py` with dependency injection so tests can replace classifier, renderer, response factory, monotonic clock, and telemetry:

```python
class FastPathRuntime:
    def __init__(self, settings, *, classifier=classify, renderer=render_handler,
                 response_factory=build_synthetic_response, telemetry=None,
                 cache=None):
        self.settings = settings
        self.classifier = classifier
        self.renderer = renderer
        self.response_factory = response_factory
        self.telemetry = telemetry
        self.cache = cache or DecisionCache(settings.cache_ttl_seconds)
        self._status_lock = threading.Lock()
        self._last_decision_at = None

    def status(self):
        with self._status_lock:
            return {
                "mode": self.settings.mode,
                "enabled_handlers": self.settings.enabled_handlers,
                "confidence_threshold": self.settings.confidence_threshold,
                "short_circuit_threshold": self.settings.short_circuit_threshold,
                "last_decision_at": self._last_decision_at,
            }

    def _emit(self, event, meta):
        if self.telemetry is not None:
            self.telemetry.write(event, meta)

    def middleware(self, *, request, next_call, api_call_count=None, session_id="",
                   turn_id="", api_mode="", **context):
        downstream_called = False

        def downstream():
            nonlocal downstream_called
            if downstream_called:
                raise RuntimeError("downstream provider call attempted more than once")
            downstream_called = True
            return next_call(request)

        meta = {
            **context,
            "session_id": str(session_id),
            "turn_id": str(turn_id),
            "api_mode": str(api_mode),
            "mode": self.settings.mode,
        }
        try:
            return self._evaluate_or_fallthrough(
                request=request,
                downstream=downstream,
                api_call_count=api_call_count,
                session_id=str(session_id),
                turn_id=str(turn_id),
                api_mode=str(api_mode),
                context=meta,
            )
        except Exception as exc:
            if downstream_called:
                raise
            self._emit(TelemetryEvent(
                outcome="fallback",
                reason=type(exc).__name__,
                text="",
                candidates=(),
            ), meta)
            return downstream()

    def _evaluate_or_fallthrough(self, *, request, downstream, api_call_count,
                                 session_id, turn_id, api_mode, context):
        if self.settings.mode == "off" or int(api_call_count or 0) != 1:
            # Shipped contract: attempts are 1-based, and a per-turn claim (plugin.py)
            # makes a re-delivered first attempt fall through instead of re-evaluating.
            return downstream()
        text = extract_latest_user_text(request, api_mode)
        if text is None:
            return downstream()
        candidates = detect_candidates(text, self.settings)
        if not candidates:
            self._emit(TelemetryEvent(
                outcome="no_candidate", reason="local_filter", text=text, candidates=(),
            ), context)
            return downstream()

        key = self.cache.key(session_id, turn_id, text)
        decision = self.cache.get(key) if session_id and turn_id else None
        if decision is None:
            decision = self.classifier(text, candidates, context, self.settings)
            if session_id and turn_id:
                self.cache.put(key, decision)
        with self._status_lock:
            self._last_decision_at = datetime.now(timezone.utc).isoformat()

        rejected = (
            decision.handler_id == "normal_llm"
            or decision.handler_id not in candidates
            or decision.confidence < self.settings.confidence_threshold
            or decision.short_circuit_probability < self.settings.short_circuit_threshold
        )
        if rejected:
            self._emit(TelemetryEvent(
                outcome="normal_llm", reason="jev_rejected", text=text,
                candidates=candidates, selected_handler=decision.handler_id,
                confidence=decision.confidence,
                short_circuit_probability=decision.short_circuit_probability,
                latency_ms=decision.latency_ms, usage=decision.usage,
            ), context)
            return downstream()

        rendered = self.renderer(
            decision.handler_id, text, context, self.settings, self.status()
        )
        if self.settings.mode == "shadow":
            self._emit(TelemetryEvent(
                outcome="would_short_circuit", reason="shadow", text=text,
                candidates=candidates, selected_handler=decision.handler_id,
                confidence=decision.confidence,
                short_circuit_probability=decision.short_circuit_probability,
                latency_ms=decision.latency_ms, usage=decision.usage,
            ), context)
            return downstream()

        response = self.response_factory(api_mode, rendered.text)
        self._emit(TelemetryEvent(
            outcome="short_circuit", reason="accepted", text=text,
            candidates=candidates, selected_handler=decision.handler_id,
            confidence=decision.confidence,
            short_circuit_probability=decision.short_circuit_probability,
            latency_ms=decision.latency_ms, usage=decision.usage,
        ), context)
        return response
```

The outer exception path checks `downstream_called` before falling through, so an exception raised by the real provider is re-raised rather than triggering a duplicate provider request. Cache hits reuse only the typed route decision; handlers render again so clock/status output remains fresh.

- [ ] **Step 6: Register middleware with invalid-config fail-open behavior**

Implement:

```python
def register(ctx):
    try:
        settings = load_settings(ctx)
    except SettingsError as exc:
        logger.warning("jev-fastpath disabled: %s", exc)
        return
    runtime = FastPathRuntime(settings, telemetry=TelemetryWriter(get_hermes_home()))
    ctx.register_middleware("llm_execution", runtime.middleware)
```

Import `get_hermes_home` from Hermes only inside `register` so package unit tests remain runnable without the Hermes source tree. Keep a module-level weak/reference-safe runtime only if needed for tests; do not create global cross-profile state at import time.

- [ ] **Step 7: Run middleware and telemetry tests**

```bash
python -m pytest tests/test_telemetry.py tests/test_plugin_middleware.py -q
```

Expected: all matrix cases pass, and every path proves zero or one downstream call.

- [ ] **Step 8: Commit middleware**

```bash
git add jev_fastpath/telemetry.py jev_fastpath/plugin.py jev_fastpath/__init__.py tests/test_telemetry.py tests/test_plugin_middleware.py
git commit -m "feat: short-circuit deterministic turns before LLM execution"
```

### Task 6: Prove Hermes host integration and profile-scoped discovery

**Files:**
- Create: `tests/test_hermes_integration.py`
- Create: `scripts/smoke_plugin.py`

**Interfaces:**
- Consumes: an installed Hermes Agent 0.21.3 source/package and the plugin root.
- Produces: evidence that real plugin discovery registers `llm_execution`, real transports accept synthetic responses, and `AIAgent.run_conversation` persists fast-path text without a provider call.

- [ ] **Step 1: Write plugin discovery integration tests**

Create an isolated temporary `HERMES_HOME`, copy the plugin tree into `<temp>/plugins/jev-fastpath`, write a minimal config enabling `jev-fastpath` in shadow mode, set `HERMES_BUNDLED_PLUGINS` to an empty temp directory, discover with `PluginManager`, and assert:

```python
loaded = manager._plugins["jev-fastpath"]
assert loaded.enabled is True
assert loaded.error is None
assert loaded.middleware_registered == ["llm_execution"]
assert manager.has_middleware("llm_execution") is True
```

Do not read or modify the real profile config.

- [ ] **Step 2: Add real agent-loop fast-path tests**

Build minimal `AIAgent` instances following Hermes' own test helpers for `chat_completions` and `codex_responses`. Inject an active `FastPathRuntime` whose classifier returns an accepted `calculator` decision. Replace the provider call with a function that raises `AssertionError` if invoked. Assert:

```python
result = agent.run_conversation("2 + 2")
assert result["completed"] is True
assert result["final_response"] == "2 + 2 = 4"
assert result["messages"][-1] == {"role": "assistant", "content": "2 + 2 = 4"}
```

Then inject `normal_llm`, return a normal fake provider response, and assert one provider call and the provider's text.

- [ ] **Step 3: Run host integration tests**

```bash
PYTHONPATH=/home/scursel/.hermes/hermes-agent \
  /home/scursel/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_hermes_integration.py -q
```

Expected: discovery, Chat Completions, Codex Responses, fast-path, and fallthrough cases all pass.

- [ ] **Step 4: Add a credential-free smoke script**

Create `scripts/smoke_plugin.py` that:

1. resolves the repository root;
2. parses `plugin.yaml` with Hermes;
3. creates a temporary isolated HERMES_HOME;
4. copies the plugin into the temporary plugin root;
5. enables it in shadow mode;
6. discovers plugins;
7. invokes the middleware with an injected fake classifier and fake provider;
8. prints one JSON object containing `manifest_ok`, `discovery_ok`, `middleware_ok`, and `provider_calls`;
9. exits nonzero unless all booleans are true and the expected provider call count matches the selected smoke mode.

- [ ] **Step 5: Execute the smoke script**

```bash
PYTHONPATH=/home/scursel/.hermes/hermes-agent \
  /home/scursel/.hermes/hermes-agent/venv/bin/python scripts/smoke_plugin.py
```

Expected JSON:

```json
{"manifest_ok": true, "discovery_ok": true, "middleware_ok": true, "provider_calls": 1}
```

The smoke uses shadow mode, so one provider call is correct.

- [ ] **Step 6: Commit integration evidence**

```bash
git add tests/test_hermes_integration.py scripts/smoke_plugin.py
git commit -m "test: verify Hermes plugin fast-path integration"
```

### Task 7: Document operations, automate verification, and prepare the audited branch

**Files:**
- Modify: `README.md`
- Create: `.github/workflows/test.yml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the complete plugin and test suite.
- Produces: reproducible installation, shadow/active configuration, rollback instructions, CI, and a branch ready for Claude Code's read-only audit.

- [ ] **Step 1: Replace README with operational documentation**

Document:

- the exact distinction between Jev typed routing and an LLM;
- supported deterministic handlers;
- prerequisites: Hermes Agent `>=0.21.3`, Python `>=3.11`, `TYPESAFE_API_KEY` in the profile gateway environment;
- installation into one profile using an absolute destination path;
- exact `plugins.enabled` and `plugins.entries.jev-fastpath.settings` configuration;
- why `shadow` is default;
- telemetry path and privacy fields;
- verification commands for discovery, gateway PID/health, one fast path, one normal LLM fallthrough, and transcript persistence;
- coexistence warning: disable or reconfigure another Jev plugin before production activation to avoid duplicate TypeSafe calls;
- rollback: set mode `off`, unenroll plugin, restart only the affected profile gateway, verify a new PID and `/health`;
- limitations: no tools, no actions, no gateway health, no 9Router replacement.

Use `/home/scursel/.hermes/profiles/<profile>/plugins/jev-fastpath` only as an example placeholder path in documentation; do not write to it during development.

- [ ] **Step 2: Add GitHub Actions**

Create `.github/workflows/test.yml` with Ubuntu, Python 3.11 and 3.12, checkout, test-extra installation, Hermes Agent `0.21.3`, compileall, pytest, and manifest smoke. The workflow must not require `TYPESAFE_API_KEY`; network Jev tests remain excluded by default.

Core commands:

```yaml
- run: python -m pip install -e '.[test]' hermes-agent==0.21.3
- run: python -m compileall -q jev_fastpath __init__.py
- run: python -m pytest -q
- run: python scripts/smoke_plugin.py
```

Add a final optional SkillSpector step guarded by `command -v skillspector`, so local fleet verification remains authoritative even if GitHub runners do not have the scanner.

- [ ] **Step 3: Run all static and dynamic checks**

```bash
python -m compileall -q jev_fastpath __init__.py
PYTHONPATH=/home/scursel/.hermes/hermes-agent \
  /home/scursel/.hermes/hermes-agent/venv/bin/python -m pytest -q
PYTHONPATH=/home/scursel/.hermes/hermes-agent \
  /home/scursel/.hermes/hermes-agent/venv/bin/python scripts/smoke_plugin.py
skillspector scan . --no-llm
```

Expected:

- compileall exits 0;
- the complete test suite passes;
- smoke JSON reports all booleans true and one shadow provider call;
- SkillSpector returns an auditable report with no unresolved critical finding in runtime plugin files.

- [ ] **Step 4: Check forbidden implementation patterns**

Run a Python source scan that fails on these runtime patterns outside tests/docs:

```python
forbidden = ["eval(", "exec(", "subprocess", "os.system", "shell=True", "pre_tool_call", "pre_gateway_dispatch"]
```

Also inspect imports to confirm there is no HTTP client other than `urllib.request`, no dynamic import, and no direct profile path.

- [ ] **Step 5: Commit documentation and CI**

```bash
git add README.md .github/workflows/test.yml pyproject.toml
git commit -m "docs: add rollout and verification runbook"
```

- [ ] **Step 6: Push the implementation branch**

```bash
git status --short
git log --oneline --decorate -8
git push -u origin main
```

Expected: clean working tree and remote `main` at the same commit as local `HEAD`.

- [ ] **Step 7: Hand the complete branch to Claude Code for read-only audit**

Run Claude Code with `Claude Opus 5`, explicitly read-only, asking it to inspect the spec, plan, full diff/history, middleware call-count guarantees, Jev trust boundary, response protocol shapes, test quality, telemetry privacy, and installation docs. Require findings ordered by severity with file/line evidence; prohibit edits.

Audit prompt:

```text
Audit this repository read-only against docs/specs/jev-fastpath-v1.md and docs/superpowers/plans/2026-09-19-jev-fastpath-v1.md. Do not edit files. Focus on correctness, fail-open behavior, exactly-once downstream calls, zero provider calls on accepted active fast paths, TypeSafe/Jev trust boundaries, arithmetic safety, cache isolation, provider response compatibility, transcript persistence, profile isolation, telemetry redaction, and missing tests. Run the test suite and smoke script. Report findings first, ordered critical/high/medium/low, each with file and line evidence; then list executed verification commands and their real results. If no findings remain, say so explicitly and name residual risks.
```

- [ ] **Step 8: Resolve audit findings through CommandCode only**

For every actionable Claude finding, dispatch CommandCode again using `z-ai/glm-5.3-flash` with the exact finding, affected files, spec constraint, and required regression test. Re-run the full verification suite and ask Claude Code for one bounded re-audit. Kryten reviews diffs and test output but does not author implementation code.

- [ ] **Step 9: Create the release-ready commit only after a clean audit**

If audit fixes were required:

```bash
git add -A
git commit -m "fix: address final fast-path audit findings"
git push
git rev-parse HEAD
gh api repos/scursel/hermes-jev-fastpath/commits/main --jq .sha
```

Expected: local `HEAD` equals the GitHub `main` SHA.

## Self-Review Results

- **Spec coverage:** Every required behavior maps to Tasks 1–7; live profile activation remains intentionally outside repository development, as required by the spec.
- **Placeholder scan:** The plan contains no deferred implementation markers; every code-producing step names files, interfaces, concrete tests, commands, and expected outcomes.
- **Type consistency:** `Settings`, `Decision`, `HandlerResult`, `TelemetryEvent`, `DecisionCache`, `FastPathRuntime`, and response factory signatures are consistent across tasks.
- **Review Focus coverage:** Contextual replies are tested in Task 2; arithmetic attacks in Task 2; Jev failure/exactly-once behavior in Tasks 3 and 5; all response modes in Task 4; cache concurrency and identity in Tasks 3 and 5.
