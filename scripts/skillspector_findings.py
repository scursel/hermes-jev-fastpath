#!/usr/bin/env python3
"""Evaluate a SkillSpector JSON report for the release gate (audit L-b).

The scanner's exit code only reports execution errors, so the gate reads the JSON report
produced by ``skillspector scan --format json -o`` and fails when any finding is CRITICAL.
Read-only: parses the report and prints a bounded summary; it never scans or executes code.
"""

from __future__ import annotations

import argparse
import json
import sys

_SEVERITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")


def _bounded(text: object, limit: int = 160) -> str:
    value = str(text if text is not None else "").strip().replace("\n", " ")
    return value[:limit]


def summarize(report: object) -> tuple[list[str], dict[str, int], str]:
    """Return ``(critical lines, severity counts, scope)`` for one parsed report."""
    if not isinstance(report, dict):
        return [], {}, "unknown"
    scope = "unknown"
    skill = report.get("skill")
    if isinstance(skill, dict) and skill.get("source"):
        scope = _bounded(skill["source"], 200)
    counts: dict[str, int] = {}
    critical: list[str] = []
    issues = report.get("issues")
    for issue in issues if isinstance(issues, list) else []:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "").strip().upper()
        if not severity:
            continue
        counts[severity] = counts.get(severity, 0) + 1
        if severity != "CRITICAL":
            continue
        raw_location = issue.get("location")
        location = raw_location if isinstance(raw_location, dict) else {}
        critical.append(
            "CRITICAL {id} {category} {file}:{line} {finding}".format(
                id=_bounded(issue.get("id"), 24) or "?",
                category=_bounded(issue.get("category"), 60) or "?",
                file=_bounded(location.get("file"), 120) or "?",
                line=_bounded(location.get("start_line"), 12) or "?",
                finding=_bounded(issue.get("pattern") or issue.get("finding")),
            )
        )
    return critical, counts, scope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when a SkillSpector JSON report contains CRITICAL findings.",
    )
    parser.add_argument("report", help="path to the JSON report written by skillspector --format json")
    args = parser.parse_args(argv)
    try:
        with open(args.report, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"skillspector gate: unreadable report ({type(exc).__name__})", file=sys.stderr)
        return 2
    critical, counts, scope = summarize(report)
    summary = " ".join(f"{name}={counts[name]}" for name in _SEVERITY_ORDER if name in counts)
    suppressed = report.get("suppressed_count") if isinstance(report, dict) else None
    if isinstance(suppressed, int) and suppressed:
        summary = f"{summary} suppressed={suppressed}".strip()
    print(f"skillspector gate: {scope} - {summary or 'no findings'}")
    for line in critical:
        print(line)
    if critical:
        print(f"skillspector gate: FAIL - {len(critical)} CRITICAL finding(s)")
        return 1
    print("skillspector gate: OK - no CRITICAL findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
