#!/usr/bin/env bash
# Mandatory SkillSpector release gate (audit L6).
#
# Scans every executable/plugin runtime scope with --no-llm and fails on any scanner
# error or any CRITICAL finding. The CRITICAL policy is enforced from the scanner's JSON
# report (audit L-b): the process exit code alone only signals execution errors, so the
# gate parses `--format json` output and fails when a CRITICAL finding is present.
# Docs and test fixtures are deliberately NOT scanned: they contain hostile example
# payloads and fake credential-shaped strings that are not runtime code.
#
# Why the repo-root scan is NOT authoritative for this repository: the scanner walks the
# whole tree it is given; on a checkout containing a virtualenv, packaged example data
# (e.g. botocore fixtures with fake AWS keys) or .git objects it either drowns the report
# in third-party noise or hangs outright. The relevant scopes below ARE the runtime.
#
# Usage: scripts/skillspector_gate.sh
set -uo pipefail

if ! command -v skillspector >/dev/null 2>&1; then
    echo "skillspector is required for the release gate but is not installed." >&2
    echo "Install v2.5.3 from its immutable source commit (not PyPI), e.g.:" >&2
    echo "uv tool install 'skillspector @ git+https://github.com/NVIDIA/skillspector.git@0562b964ec5ceac67ee15c163738e5404f14a908'" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "python3 is required to evaluate the skillspector JSON report" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT="$(mktemp)"
trap 'rm -f "$REPORT"' EXIT

status=0
for target in jev_fastpath scripts; do
    if [ ! -d "$target" ]; then
        echo "missing scope: $target" >&2
        status=1
        continue
    fi
    echo "== skillspector scan $target --no-llm =="
    rm -f "$REPORT"
    if ! skillspector scan "$target" --no-llm --format json -o "$REPORT" >/dev/null; then
        echo "skillspector scan failed for $target" >&2
        status=1
        continue
    fi
    if ! "$PYTHON_BIN" "$SCRIPT_DIR/skillspector_findings.py" "$REPORT"; then
        status=1
    fi
done

# The root entrypoint is executable runtime too; scan it as part of the package scope
# (skillspector takes directories, so __init__.py is covered by a source-level check):
if [ -f __init__.py ]; then
    echo "== root entrypoint __init__.py (source-checked) =="
    grep -nE "eval\(|exec\(|subprocess|os\.system|shell=True|__import__\(|importlib" __init__.py && {
        echo "forbidden pattern in root entrypoint" >&2
        status=1
    } || true
fi

exit "$status"
