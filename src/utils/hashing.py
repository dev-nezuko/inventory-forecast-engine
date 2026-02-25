"""HMAC-SHA256 pseudonyms for customer-ID linkage."""

import hashlib
import hmac

from src.config import MIN_HASH_KEY_LENGTH, hash_key


def hash_id(raw_id: str | None, key: str | None = None) -> str | None:
    """Stable HMAC-SHA256 pseudonym for raw_id. The key defaults to $HASH_SALT."""
    if raw_id is None:
        return None
    if key is None:
        key = hash_key()
    elif len(key) < MIN_HASH_KEY_LENGTH:
        raise ValueError(f"HMAC key must be at least {MIN_HASH_KEY_LENGTH} characters")
    return hmac.new(key.encode(), str(raw_id).encode(), hashlib.sha256).hexdigest()
