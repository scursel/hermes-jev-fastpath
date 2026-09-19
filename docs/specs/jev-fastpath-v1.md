# Hermes Jev Fast Path — v1 Specification

## 1. Product

`hermes-jev-fastpath` is a profile-scoped Hermes plugin that can complete a narrow, deterministic user turn without calling the configured LLM provider.

The plugin runs at Hermes' existing `llm_execution` middleware boundary. It inspects only the first model attempt of a user turn, identifies locally supported candidates, asks TypeSafe Jev for a typed route decision, validates that decision against an allowlist, executes a deterministic in-process handler, and returns a provider-compatible synthetic response. Any uncertainty or failure falls through to Hermes' normal LLM call.

## 2. Primary goal

Reduce net model cost and latency for simple requests that do not require generative reasoning while preserving normal Hermes behavior, transcript persistence, prompt-cache invariants, profile isolation, provider compatibility, and the user's execution freedom.

## 3. Required behavior

1. The implementation is a standalone Hermes plugin. It must not patch the Hermes checkout.
2. The plugin registers exactly one behavior-changing surface: `llm_execution` middleware.
3. The plugin examines only the first LLM attempt of a turn (`api_call_count == 0`). Tool rounds, retries, fallback calls, and continuation calls always use `next_call(request)`.
4. A cheap deterministic candidate detector runs before Jev. If no installed handler can possibly handle the text, the plugin must not call Jev and must immediately call `next_call(request)`.
5. Jev may select only one handler ID from the candidate detector's allowlist or `normal_llm`.
6. Jev output is advisory input to a strict local validator. It can never produce shell, Python, URLs, tool names, files, commands, or free-form response text.
7. A fast path runs only when all of these are true:
   - plugin mode is `active`;
   - the turn is the first provider attempt;
   - the current user input is non-empty plain text;
   - at least one deterministic candidate exists;
   - Jev returns a known handler ID;
   - Jev choice confidence is at least `0.92` by default;
   - Jev's typed short-circuit `noul` is at least `0.90` by default;
   - the local handler validates and executes successfully;
   - the active Hermes `api_mode` has a tested synthetic-response adapter.
8. In `shadow` mode, the plugin performs classification and local rendering but always calls `next_call(request)`. It records what would have been returned without altering the turn.
9. Any timeout, HTTP error, malformed Jev answer, unsupported protocol, parser rejection, handler error, logging error, or cache inconsistency must fail open to `next_call(request)`.
10. The plugin must never block a turn, approve an action, alter tool permissions, mutate prior messages, rewrite the system prompt, register `pre_tool_call`, or suppress Hermes safety/approval behavior.
11. The synthetic response must pass through Hermes' normal response intake and finalization so the assistant message is persisted exactly once.
12. TypeSafe credentials come only from `TYPESAFE_API_KEY` in the process environment. The plugin must never read `.env`, `auth.json`, or another profile.

## 4. Built-in handlers

### 4.1 `calculator`

Handles a pure arithmetic request when the expression can be reduced to a safe AST containing only numeric constants, parentheses, unary `+`/`-`, and binary `+`, `-`, `*`, `/`, `//`, `%`, and `**`.

Limits:

- expression length: 160 characters;
- AST nodes: 64;
- integer digits: 100;
- absolute exponent: 12;
- no names, attributes, calls, indexing, strings, comprehensions, booleans, or containers;
- division by zero and non-finite results reject the fast path;
- result formatting is deterministic and locale-independent.

Examples accepted:

- `2 + 2`
- `quanto é (17 * 9) - 4?`
- `calcule 12.5 / 5`

Examples rejected:

- `calcule o desconto ideal para meu negócio`
- `__import__('os').system('id')`
- `2 ** 999999`

### 4.2 `clock`

Answers direct requests for the current date, current time, or both. It uses `zoneinfo.ZoneInfo` and the configured IANA timezone, defaulting to `America/Sao_Paulo`. Unknown timezones reject the fast path.

It does not interpret relative scheduling, travel-time conversion, calendar events, reminders, or historical/future date arithmetic.

### 4.3 `runtime_identity`

Answers direct questions about the current Hermes provider, model, API mode, or platform using only middleware context fields. It must not claim gateway, service, quota, account, profile, or host health that the middleware context does not prove.

### 4.4 `acknowledgement`

Handles only exact, bounded social acknowledgements such as `obrigado`, `valeu`, `ok`, `entendi`, `thanks`, and `got it`. Matching is normalized and allowlisted; substrings in a longer request do not qualify. Responses are fixed by locale.

### 4.5 `fastpath_status`

Answers a direct request for this plugin's own mode, enabled handler IDs, thresholds, and last in-process decision timestamp. It does not report provider quotas or Hermes gateway health.

## 5. Candidate detection

The detector receives the latest user text and returns an ordered tuple of possible handler IDs. It is conservative:

- slash commands return no candidates because Hermes owns command routing;
- empty, multimodal, binary, or synthetic input returns no candidates;
- text above `max_input_chars` returns no candidates;
- code blocks, URLs, file paths, credential-looking strings, shell metacharacters, or explicit requests to modify external state return no candidates;
- a handler ID appears only when its local parser recognizes the request shape;
- `normal_llm` is never a local candidate; it is always included only in Jev's choice criteria.

The local detector is an eligibility filter, not the final route decision. Jev resolves ambiguity among eligible handlers and `normal_llm`.

## 6. Jev contract

Endpoint: `https://api.typesafe.ai/v1/systemone`

Model: `jev-latest`

Request shape:

```json
{
  "model": "jev-latest",
  "state": {
    "message": "redacted and bounded user text",
    "platform": "telegram",
    "candidate_handlers": ["calculator"]
  },
  "questions": {
    "handler": {
      "type": "choice",
      "instructions": "Choose exactly one deterministic handler, otherwise normal_llm.",
      "criteria": {
        "calculator": "The request is fully answered by the validated arithmetic parser.",
        "normal_llm": "The request needs context, reasoning, tools, writing, interpretation, or unsupported data."
      }
    },
    "safe_to_short_circuit": {
      "type": "noul",
      "instructions": "Probability that returning the chosen deterministic result fully satisfies this turn without tools or an LLM.",
      "criteria": {
        "true": "The chosen handler completely answers the literal request.",
        "false": "Any context, tool, judgment, ambiguity, side effect, or generative work is needed."
      }
    }
  }
}
```

The parser requires:

- `answers` is a dictionary;
- `answers.handler.choice` is a string in `candidate_handlers ∪ {normal_llm}`;
- `answers.handler.confidence` is a finite number in `[0, 1]`;
- `answers.safe_to_short_circuit.noul` is a finite number in `[0, 1]`;
- missing or invalid probability is an error, not zero and not success.

The client uses one request, no retry, and a default timeout of 3 seconds. A timeout is more expensive than a normal fallthrough and must not stall the turn repeatedly.

## 7. Middleware lifecycle

For each `llm_execution` callback:

1. Reject non-first calls and unsupported contexts immediately.
2. Extract the latest user text from the provider request without modifying it.
3. Compute deterministic candidates.
4. Reuse a turn-scoped cached decision when `(session_id, turn_id, text_hash)` matches and the entry is younger than 15 minutes.
5. Otherwise call Jev once and cache the typed decision.
6. Render the selected deterministic handler locally.
7. In `shadow` mode, log `would_short_circuit` and call `next_call(request)`.
8. In `active` mode, build the raw synthetic response for the active `api_mode` and return it without calling `next_call`.
9. If any step fails, log a bounded fallback reason and call `next_call(request)` exactly once.

The middleware must never call `next_call` after it returns a synthetic response and must never call `next_call` more than once.

## 8. Provider protocol compatibility

The response factory supports these Hermes API modes:

- `chat_completions`: OpenAI-shaped object with one assistant choice, `finish_reason="stop"`, and no provider usage.
- `codex_responses`: Responses-shaped object with one completed `message` output item containing one `output_text` part, `status="completed"`, and no provider usage.
- `anthropic_messages`: Anthropic-shaped object with one text content block and `stop_reason="end_turn"`.
- `bedrock_converse`: Bedrock Converse dictionary with one assistant text content block and `stopReason="end_turn"`.

Unknown API modes always fall through.

A synthetic response carries private marker attributes/keys only where the host transport ignores them. Telemetry must not rely on that marker surviving Hermes normalization.

## 9. Configuration

Configuration lives under `plugins.entries.jev-fastpath.settings` in the active profile:

```yaml
plugins:
  entries:
    jev-fastpath:
      enabled: true
      settings:
        mode: shadow
        confidence_threshold: 0.92
        short_circuit_threshold: 0.90
        timeout_seconds: 3.0
        timezone: America/Sao_Paulo
        locale: pt-BR
        max_input_chars: 4000
        enabled_handlers:
          - calculator
          - clock
          - runtime_identity
          - acknowledgement
          - fastpath_status
```

Validation rules:

- `mode`: `off`, `shadow`, or `active`;
- thresholds: finite floats in `[0, 1]`;
- timeout: `0.2` through `10.0` seconds;
- `timezone`: valid `ZoneInfo` name;
- `locale`: `pt-BR` or `en` in v1;
- `max_input_chars`: integer from `32` through `12000`;
- enabled handlers: unique known IDs only.

Invalid settings disable the fast path and emit one bounded warning; they must not crash plugin discovery.

## 10. Telemetry and privacy

Append JSONL to `<HERMES_HOME>/artifacts/jev_fastpath/decisions.jsonl`.

Each record contains:

- UTC timestamp;
- mode;
- session/turn hash, not raw IDs;
- platform, provider, model, and API mode;
- bounded redacted text preview and text hash;
- local candidates;
- selected handler;
- Jev confidence and `noul`;
- Jev latency and reported cost/tokens when present;
- outcome: `no_candidate`, `normal_llm`, `would_short_circuit`, `short_circuit`, or `fallback`;
- fallback reason code;
- estimated provider call avoided as a boolean.

Never log request bodies, full conversation history, credentials, tool arguments, raw exceptions containing secrets, or the TypeSafe authorization header. Logging failures are swallowed after one debug/warning signal and never affect the turn.

## 11. Safety invariants

- No `eval`, `exec`, subprocess, shell, filesystem mutation, HTTP URL supplied by Jev, dynamic import, or arbitrary callable name.
- No free-form response from Jev.
- No action handlers in v1.
- No mutation of Hermes messages, tools, system prompt, provider request, or session state.
- No cross-profile reads or writes.
- Exactly one downstream provider call on every fallthrough path.
- Exactly zero downstream provider calls on every successful active fast path.
- Jev failures are always fail-open.
- Low confidence is always normal LLM, never a refusal or human gate.

## 12. Testing acceptance criteria

1. Unit tests cover configuration, extraction, candidate detection, all handlers, Jev parsing, cache identity/expiry, telemetry redaction, and each response adapter.
2. Malicious arithmetic payloads and oversized inputs reject without executing code.
3. Middleware tests prove `next_call` is invoked exactly once on every bypass/fallback path.
4. Middleware tests prove `next_call` is not invoked for an active accepted fast path.
5. Shadow-mode tests prove classification/rendering occurs while the provider response is returned unchanged.
6. Integration tests run a real Hermes `AIAgent.run_conversation` with fake provider calls for both `chat_completions` and `codex_responses`; the final assistant response is persisted in the normal message shape.
7. A live TypeSafe probe is optional and marked separately; the default test suite never requires credentials or network.
8. `python -m pytest -q`, `python -m compileall`, and `skillspector scan . --no-llm` complete successfully before release.

## 13. Non-goals for v1

- Replacing 9Router, provider routing, semantic caching, or token compression.
- Executing Hermes tools without an LLM.
- Generating prose, summaries, plans, code, emails, or business answers.
- Reading gateway/systemd health, cron state, files, databases, or external APIs other than TypeSafe.
- Blocking dangerous work or making approval decisions.
- Modifying the existing Zen `jev-triage-active` plugin.
- Automatically enabling the plugin in a live profile during repository development.

## 14. Rollout contract

The repository ships with `mode: shadow` as the documented default. Live installation and profile enrollment are separate operational steps. Before enabling `active`, collect shadow telemetry, compare suggested fast paths against normal Hermes answers, verify no duplicate TypeSafe call from another Jev plugin, then enable only on one profile and validate gateway PID, health, plugin discovery, a real fast path, a real fallthrough, and transcript persistence.
