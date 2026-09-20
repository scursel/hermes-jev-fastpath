# Contributing

Contributions are welcome, but the plugin intentionally keeps a narrow safety boundary:
Jev may choose only from locally detected, allowlisted deterministic handlers. It must
never generate commands, URLs, tool arguments, code, or response prose.

## Development setup

Use Python 3.11 or newer and keep the virtual environment outside the plugin directory
when running Hermes' whole-tree plugin validator.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/smoke_plugin.py
python scripts/check_forbidden_patterns.py
bash scripts/skillspector_gate.sh
```

For Hermes-hosted integration tests:

```bash
PYTHONPATH=/path/to/hermes-agent python -m pytest -q
```

Before opening a pull request, also run:

```bash
HERMES_HOME="$(mktemp -d)" hermes plugins validate .
HERMES_HOME="$(mktemp -d)" hermes plugins doctor . --ci
git diff --check
```

## Pull-request expectations

- preserve fail-open behavior and exactly-once `next_call` semantics;
- add regression tests for every behavior change;
- test every supported API-mode response adapter through Hermes normalization;
- do not weaken input rejection, credential scoping, redirect rejection, response-size
  bounds, or telemetry redaction;
- keep public claims evidence-based and separate per-hit savings from workload coverage;
- do not commit credentials, real session transcripts, telemetry logs, or local profile
  configuration.

Use conventional commit subjects such as `fix:`, `feat:`, `test:`, `docs:`, and `chore:`.

## Security reports

Do not open a public issue for a suspected vulnerability. Follow [`SECURITY.md`](SECURITY.md).
