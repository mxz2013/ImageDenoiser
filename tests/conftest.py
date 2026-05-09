"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root regardless of where pytest is invoked from."""
    return Path(__file__).resolve().parent.parent
