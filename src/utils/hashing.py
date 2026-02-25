"""HMAC-SHA256 pseudonyms for customer-ID linkage; source identifiers are not removed."""

import hmac
import hashlib
from typing import Optional

from src.config import SALT


def hash_id(raw_id: Optional[str], salt: str = SALT) -> Optional[str]:
    """Stable HMAC-SHA256 hash for privacy-preserving IDs. Salt as str is encoded to bytes."""
    if raw_id is None:
        return None
    return hmac.new(salt.encode(), str(raw_id).encode(), hashlib.sha256).hexdigest()
