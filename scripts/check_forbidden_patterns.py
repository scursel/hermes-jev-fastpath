#!/usr/bin/env python3
"""Fail the build on forbidden implementation patterns in the runtime plugin code.

Scans ``jev_fastpath/`` and the root ``__init__.py`` (never tests/docs) for:

- dynamic code execution: ``eval(``, ``exec(``, ``subprocess``, ``os.system``, ``shell=True``
- behavior-changing hooks the plugin must never register: ``pre_tool_call``,
  ``pre_gateway_dispatch``
- dynamic imports: ``__import__(``, ``importlib``
- HTTP clients other than ``urllib.request``
- absolute profile paths or cross-profile reads

Exit code 0 means clean; nonzero lists the offending file:line evidence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_TARGETS = (REPO_ROOT / "jev_fastpath", REPO_ROOT / "__init__.py")

FORBIDDEN = (
    ("dynamic eval", re.compile(r"\beval\(")),
    ("dynamic exec", re.compile(r"\bexec\(")),
    ("subprocess use", re.compile(r"\bsubprocess\b")),
    ("os.system", re.compile(r"\bos\.system\b")),
    ("shell=True", re.compile(r"shell\s*=\s*True")),
    ("pre_tool_call hook", re.compile(r"\bpre_tool_call\b")),
    ("pre_gateway_dispatch hook", re.compile(r"\bpre_gateway_dispatch\b")),
    ("dynamic import", re.compile(r"\b__import__\s*\(|\bimportlib\b")),
    ("non-urllib HTTP client", re.compile(r"^\s*import\s+(requests|httpx|aiohttp|urllib3)\b|^\s*from\s+(requests|httpx|aiohttp|urllib3)\b", re.M)),
    ("hardcoded profile path", re.compile(r"/home/|\.hermes[/\\]profiles|profiles[/\\]")),
)


def main() -> int:
    findings: list[str] = []
    files = sorted(list((REPO_ROOT / "jev_fastpath").rglob("*.py")) + [REPO_ROOT / "__init__.py"])
    for path in files:
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN:
            for index, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    findings.append(f"{path.relative_to(REPO_ROOT)}:{index}: {label}: {line.strip()[:120]}")
    if findings:
        print("FORBIDDEN PATTERNS FOUND:")
        print("\n".join(findings))
        return 1
    print(f"clean: {len(files)} runtime file(s) scanned, no forbidden patterns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
