"""Release-gate regression (audit L-b): a CRITICAL SkillSpector finding must fail the gate.

The gate is exercised end to end with a fake ``skillspector`` on PATH that writes a
fixture JSON report, so the test proves the failure logic without running the scanner or
any hostile payload.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "skillspector_gate.sh"
FINDINGS = REPO_ROOT / "scripts" / "skillspector_findings.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "skillspector"
CRITICAL_REPORT = FIXTURES / "report_critical.json"
CLEAN_REPORT = FIXTURES / "report_clean.json"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def run_findings(report):
    return subprocess.run(
        [sys.executable, str(FINDINGS), str(report)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


FAKE_SKILLSPECTOR = """#!/usr/bin/env bash
set -euo pipefail
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cp "{report}" "$out"
"""


def run_gate(tmp_path, report):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "skillspector"
    fake.write_text(FAKE_SKILLSPECTOR.format(report=report))
    fake.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(GATE)], capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )


class TestFindingsParser:
    def test_critical_report_fails_and_names_the_finding(self):
        result = run_findings(CRITICAL_REPORT)
        assert result.returncode == 1
        assert "CRITICAL E9" in result.stdout
        assert "danger.py:2" in result.stdout

    def test_non_critical_report_passes(self):
        result = run_findings(CLEAN_REPORT)
        assert result.returncode == 0
        assert "OK" in result.stdout
        assert "LOW=1" in result.stdout

    def test_unreadable_report_is_a_gate_error(self, tmp_path):
        result = run_findings(tmp_path / "missing.json")
        assert result.returncode == 2


class TestGateScript:
    def test_gate_fails_when_a_critical_finding_is_present(self, tmp_path):
        result = run_gate(tmp_path, CRITICAL_REPORT)
        assert result.returncode == 1
        assert "FAIL" in result.stdout
        assert "CRITICAL E9" in result.stdout

    def test_gate_passes_when_no_finding_is_critical(self, tmp_path):
        result = run_gate(tmp_path, CLEAN_REPORT)
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_fixtures_match_the_scanner_report_schema(self):
        # Guard the fixtures against silent schema drift: the gate parses these keys.
        report = json.loads(CRITICAL_REPORT.read_text(encoding="utf-8"))
        assert isinstance(report["skill"]["source"], str)
        severities = {issue["severity"] for issue in report["issues"]}
        assert "CRITICAL" in severities
        for issue in report["issues"]:
            assert set(issue) >= {"id", "severity", "location"}
            assert set(issue["location"]) >= {"file", "start_line"}
