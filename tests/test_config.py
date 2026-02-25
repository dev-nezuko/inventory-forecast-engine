"""Tests for src.config: ROOT resolution, DATA_DIR default, optional PROJECT_ROOT override."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_root_is_repo_root() -> None:
    from src import config

    # ROOT should be the directory containing src/
    assert (config.ROOT / "src" / "config.py").exists()


def test_data_dir_default() -> None:
    from src import config

    assert config.DATA_DIR == config.ROOT / "examples" / "sample_data"


def test_project_root_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import src.config as mod

    monkeypatch.setenv("PROJECT_ROOT", "/tmp/custom_root")
    importlib.reload(mod)
    assert mod.ROOT == Path("/tmp/custom_root")
    # Restore so other tests are not affected
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    importlib.reload(mod)
