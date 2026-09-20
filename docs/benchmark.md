# Benchmark and production-fit note

This document separates two questions that are easy to conflate:

1. **Does an accepted fast-path hit avoid the configured LLM call correctly?**
2. **Does the supported handler set match enough real traffic to matter?**

The first answer is yes. In the author's workload, the second answer was no.

## Controlled A/B

Run date: 2026-09-19.

Environment:

- Hermes Agent 0.21.3;
- `bai/deepseek-v4.1-flash` through an OpenAI-compatible router;
- fresh Hermes CLI session per prompt;
- identical profile, provider, model, reasoning level, and prompt set;
- plugin `active` for one arm and `off` for the other;
- five eligible prompts: four bounded arithmetic requests and one exact gratitude message.

The benchmark read Hermes session usage counters after each run and matched each active
turn to its `jev_fastpath` telemetry decision.

| Metric | Active | Off |
|---|---:|---:|
| Successful runs | 5/5 | 5/5 |
| Correct answers | 5/5 | 5/5 |
| Median wall time | 15.865 s | 25.438 s |
| Mean wall time | 16.277 s | 24.557 s |
| Main-provider input tokens | 0 | 96,731 |
| Main-provider output tokens | 0 | 376 |
| Main-provider reasoning tokens | 0 | 158 |
| Main-provider cache-read tokens | 0 | 160,768 |
| Provider calls avoided | 5 | 0 |
| Jev input tokens | 2,211 | 0 |
| Jev output tokens | 283 | 0 |

Derived from those counters:

- median wall-time reduction: **37.63%**;
- main-provider tokens avoided (input + output + reasoning): **97,265**;
- Jev tokens spent: **2,494**;
- net tokens avoided after subtracting Jev usage: **94,771 (97.44%)**;
- median Jev network latency inside the accepted decisions: **679 ms**.

The run does **not** establish monetary savings. Both Jev and provider cost fields reported
`0.0`/unknown in that environment, and cache-read accounting is provider-specific.

## Historical coverage replay

The local candidate detector was replayed against 7,513 non-empty human turns from the
author's Hermes session store. The replay:

- included Telegram, WhatsApp, Desktop, CLI, TUI, Web UI, API-server, and ACP sessions;
- stripped the trailing internal `<memory-context>` block before detection;
- excluded context-compaction wrappers, out-of-band control messages, blank turns, and
  synthetic `oneshot`/benchmark traffic;
- used the shipped handler set and active thresholds.

Result:

| Metric | Result |
|---|---:|
| Real turns replayed | 7,513 |
| Candidate turns | 1 |
| Coverage | **0.0133%** |
| Matching handler | `clock` |

This is why the author's installation was disabled even though the per-hit mechanism worked.
A large synthetic saving multiplied by almost-zero workload coverage produces almost-zero
aggregate value.

## Limitations

- Five prompts are a mechanism smoke benchmark, not a broad quality evaluation.
- The A/B arms were run sequentially rather than randomized; provider load and cache state
  can affect wall time.
- Fresh CLI startup overhead is included in both arms.
- Session counters are used as reported by Hermes/the provider and may not map directly to
  billed tokens.
- Historical coverage describes one workload. A deployment with frequent arithmetic,
  clock, runtime-identity, exact-gratitude, or plugin-status requests may see very different
  coverage.
- Open-ended coding, tool calls, approvals, follow-ups, and `no_agent` cron jobs are outside
  this plugin's fast-path contract.

## Reproduce safely

Use `shadow` mode first. Compare `would_short_circuit` telemetry against normal provider
answers, then run a matched active/off set with the same model and profile. Always report:

- correctness;
- accepted-hit rate;
- real workload coverage;
- provider calls and tokens avoided;
- Jev tokens and latency;
- fallthrough and failure counts.

Do not publish raw session text or telemetry previews. Aggregate locally and disclose only
bounded statistics.
