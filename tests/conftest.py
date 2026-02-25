"""Shared pytest fixtures. `pythonpath = ["."]` in pyproject.toml makes `src` importable."""

import pytest

TEST_HASH_KEY = "test-only-hmac-key-0123456789"


@pytest.fixture(autouse=True)
def hash_key_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("HASH_SALT", TEST_HASH_KEY)
    return TEST_HASH_KEY
