"""Shared test fixtures: repo-root imports, default settings, isolated Hermes homes.

The default suite never requires the Hermes source tree, credentials, or network; the
Hermes-hosted integration modules opt in explicitly via ``pytest.importorskip``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jev_fastpath.config import Settings  # noqa: E402


@pytest.fixture
def settings():
    return Settings()
