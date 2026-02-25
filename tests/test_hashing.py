"""Tests for src.utils.hashing: determinism, None input, key handling."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from src.utils.hashing import hash_id


def test_hash_id_deterministic() -> None:
    a = hash_id("customer_123")
    b = hash_id("customer_123")
    assert a == b and a is not None


def test_hash_id_is_hmac_sha256_of_env_key(hash_key_env: str) -> None:
    expected = hmac.new(hash_key_env.encode(), b"C001", hashlib.sha256).hexdigest()
    assert hash_id("C001") == expected


def test_hash_id_none_returns_none() -> None:
    assert hash_id(None) is None


def test_hash_id_different_key_different_output() -> None:
    h1 = hash_id("customer_123", key="first-key-0123456789")
    h2 = hash_id("customer_123", key="second-key-0123456789")
    assert h1 != h2


def test_hash_id_different_input_different_output() -> None:
    assert hash_id("customer_123") != hash_id("customer_456")


def test_hash_id_requires_key_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HASH_SALT")
    with pytest.raises(RuntimeError, match="HASH_SALT must be set"):
        hash_id("customer_123")


def test_hash_id_rejects_short_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HASH_SALT", "short")
    with pytest.raises(RuntimeError, match="at least 16"):
        hash_id("customer_123")


def test_hash_id_rejects_short_explicit_key() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        hash_id("customer_123", key="salt1")


def test_old_placeholder_default_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as config

    monkeypatch.delenv("HASH_SALT")
    assert not hasattr(config, "SALT")
    with pytest.raises(RuntimeError):
        config.hash_key()
