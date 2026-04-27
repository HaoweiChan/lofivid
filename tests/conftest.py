"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Throwaway cache directory for tests that touch the disk cache."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
