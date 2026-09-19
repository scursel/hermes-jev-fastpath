# hermes-jev-fastpath

Profile-scoped Hermes plugin that uses TypeSafe Jev to short-circuit narrow, deterministic
turns **before** the configured LLM provider is called — with fail-open behavior on every
uncertainty, timeout, or error.

## What this is (and is not)

**Jev here is a typed router, not a writer.** For each eligible turn the plugin sends the
redacted user text plus a list of locally-detected candidate handlers to TypeSafe's
`jev-latest` model. Jev answers exactly two typed fields: one handler ID chosen from the
candidate allowlist (or `normal_llm`) and a short-circuit probability. It can never
produce shell commands, URLs, code, tool names, tool arguments, files, or response prose —
the answer text is always rendered locally by an allowlisted handler.

It is **not** a replacement for 9Router, provider routing, semantic caching, or token
compression. It does not execute tools, run actions, block dangerous work, make approval
decisions, or report gateway/systemd/quota health.

## Deterministic handlers (v1)

| Handler | Answers | Example |
|---|---|---|
| `calculator` | Pure arithmetic reducible to a bounded safe AST (`+ - * / // % **`, parentheses, unary signs) | `quanto é (17 * 9) - 4?` |
| `clock` | Current date, time, or both in the configured IANA timezone | `que horas são?` |
| `runtime_identity` | Which provider, model, API mode, or platform is serving this turn (middleware context only) | `qual é o seu modelo?` |
| `acknowledgement` | Exact, allowlisted social acknowledgements (`obrigado`, `valeu`, `ok`, `thanks`, `got it`, …) | `obrigado` |
| `fastpath_status` | This plugin's own mode, handlers, thresholds, last decision timestamp | `fastpath status` |

Everything else — reasoning, writing, tools, context, ambiguity — routes to `normal_llm`
and Hermes runs the real provider call exactly once.

## How a turn flows

1. Hermes fires the `llm_execution` middleware with the built provider request.
2. Only the **first provider attempt of a turn** is eligible (`api_call_count == 1`, the
   count Hermes increments before the call). Tool rounds, retries, fallbacks, and
   continuations bypass the plugin entirely.
3. A conservative local detector proposes candidate handlers. No candidates, slash
   commands, code blocks, URLs, credential-shaped strings, shell metacharacters, action
   requests, or oversized input → straight to `next_call(request)` with no Jev call.
4. Jev classifies once (3 s timeout default, no retries). The typed decision is validated
   locally: known handler from the candidate set, confidence ≥ 0.92, short-circuit
   probability ≥ 0.90 (both configurable).
5. `shadow` mode (the default): render locally, record `would_short_circuit` telemetry,
   then call `next_call(request)` — the provider answer is returned unchanged.
6. `active` mode: return a provider-shaped synthetic response for the active `api_mode`
   (`chat_completions`, `codex_responses`, `anthropic_messages`, `bedrock_converse`) and
   call the provider **zero** times.
7. Any timeout, HTTP error, malformed Jev answer, parser rejection, handler error,
   telemetry failure, or cache inconsistency fails open to `next_call(request)` exactly
   once. Prompt cache invariants are preserved: messages, system prompt, history, and
   tool definitions are never mutated.

## Prerequisites

- Hermes Agent `>= 0.21.3` with the native directory-plugin loader.
- Python `>= 3.11` (stdlib only at runtime; no pip dependencies).
- `TYPESAFE_API_KEY` present in the profile gateway environment. The plugin reads this
  one environment variable and nothing else — never `.env`, `auth.json`, or another
  profile.

## Installation (one profile)

Copy the repository into the target profile's plugin directory using an absolute
destination path, then enable it. Example placeholder path (use your real profile home;
this repository's development never writes there):

```bash
cp -r /home/scursel/hermes-jev-fastpath /home/scursel/.hermes/profiles/<profile>/plugins/jev-fastpath
```

Enable and configure in that profile's `config.yaml`:

```yaml
plugins:
  enabled:
    - jev-fastpath
  entries:
    jev-fastpath:
      enabled: true
      settings:
        mode: shadow                # off | shadow | active — shadow is the default
        confidence_threshold: 0.92  # minimum Jev choice confidence [0, 1]
        short_circuit_threshold: 0.90
        timeout_seconds: 3.0        # 0.2 – 10.0
        timezone: America/Sao_Paulo
        locale: pt-BR               # pt-BR | en
        max_input_chars: 4000       # 32 – 12000
        enabled_handlers:
          - calculator
          - clock
          - runtime_identity
          - acknowledgement
          - fastpath_status
```

Invalid settings emit one bounded warning and disable the fast path (the plugin stays
discovered and Hermes keeps booting normally).

## Why shadow is the default

Shadow mode performs the full classification and local rendering but always returns the
real provider response, recording what *would* have been short-circuited. Collect shadow
telemetry, compare suggested fast paths against normal Hermes answers, verify no other
Jev plugin duplicates the TypeSafe call, and only then set `mode: active` on **one**
profile.

## Coexistence warning

Another Jev-based plugin (for example an existing `jev-triage` style plugin) in the same
profile will call TypeSafe for the same turns. Disable or reconfigure it before enabling
`active` mode here to avoid duplicate TypeSafe calls and cost.

## Telemetry and privacy

Decisions append one JSON object per line to
`<HERMES_HOME>/artifacts/jev_fastpath/decisions.jsonl` with: UTC timestamp, mode, hashed
(not raw) session/turn IDs, platform/provider/model/api mode, a redacted and bounded text
preview plus text hash, local candidates, selected handler, Jev confidence and
short-circuit probability, Jev latency and reported cost/tokens when present, the outcome
(`no_candidate`, `normal_llm`, `would_short_circuit`, `short_circuit`, `fallback`), a
fallback reason code, and a `provider_call_avoided` boolean. Request bodies, full
conversation history, credentials, tool arguments, and the TypeSafe authorization header
are never logged; logging failures are swallowed and never affect the turn.

## Verification

After installing into a profile:

```bash
# Discovery and registration
HERMES_HOME=~/.hermes/profiles/<profile> hermes plugins list
HERMES_HOME=~/.hermes/profiles/<profile> hermes plugins doctor \
  ~/.hermes/profiles/<profile>/plugins/jev-fastpath --ci

# Gateway identity and health (record the PID before and after any restart)
systemctl --user status hermes-<profile>   # or your profile's gateway unit
curl -s localhost:8642/health              # adjust to the profile's gateway port
```

Then, on that one profile:

1. Send `2 + 2` (or `que horas são?`) with `mode: shadow` — expect the normal LLM answer
   and a `would_short_circuit` row in `decisions.jsonl`.
2. Set `mode: active` and send `2 + 2` — expect the deterministic answer with **no**
   provider usage and a `short_circuit` telemetry row.
3. Send an open question such as `fale sobre arte moderna` — expect a normal provider
   answer and a `no_candidate` or `normal_llm` row (exactly one provider call).
4. Check transcript/session persistence: the assistant message appears in the session
   exactly once, in the normal message shape (`hermes --resume` or your session viewer).

## Rollback

1. Set `mode: off` in the plugin settings (fast path disabled immediately).
2. Remove `jev-fastpath` from `plugins.enabled` (full unenroll).
3. Restart only the affected profile's gateway.
4. Verify the gateway came back with a **new PID** and `/health` is green, and that a
   normal turn answers via the provider.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest -q                                  # unit suite (no network, no credentials)
PYTHONPATH=/path/to/hermes-agent python -m pytest -q # full suite incl. Hermes-hosted integration
python scripts/smoke_plugin.py                       # credential-free end-to-end smoke
```

The Hermes-hosted tests (transports, plugin discovery, `AIAgent.run_conversation`) skip
automatically when the Hermes source tree is not importable.

Static verification also includes `python scripts/check_forbidden_patterns.py` (fails on
`eval`/`exec`/subprocess/shell/dynamic-import/forbidden-hook patterns in runtime code) and
SkillSpector (`skillspector scan <dir> --no-llm`). The runtime package scans clean of
critical findings; the two advisory signals are intentional, spec-mandated behavior: the
TypeSafe endpoint constant in `jev.py` (E1) and the required `TYPESAFE_API_KEY` read
(E2), which is never logged and only sent as the authorization header to that endpoint.
Scan the repo root with `.venv-test` (local test virtualenv) moved aside; the scanner
hangs on large third-party trees.

## Limitations

- No tools, no actions, no approval or blocking behavior, no gateway health reporting.
- No prose generation of any kind; the five allowlisted handlers produce fixed-shape
  answers only.
- The synthetic response marker (`_jev_fastpath`) is best-effort; telemetry never relies
  on it surviving Hermes normalization.
