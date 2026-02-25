import os
from pathlib import Path

# ROOT is the repo root (directory containing src/). Override for tests via PROJECT_ROOT.
ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))

# Data layout (override via env vars). Writers create these folders on demand.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "examples" / "sample_data"))
LANDING_DIR = DATA_DIR / "landing"
FEATURE_DIR = DATA_DIR / "features"

HASH_KEY_ENV = "HASH_SALT"
MIN_HASH_KEY_LENGTH = 16


def hash_key() -> str:
    """Return the HMAC key for customer pseudonyms from the environment.

    There is deliberately no built-in fallback: a key that ships with the code
    lets anyone recompute the pseudonyms from raw customer IDs.
    """
    key = os.environ.get(HASH_KEY_ENV, "")
    if len(key) < MIN_HASH_KEY_LENGTH:
        raise RuntimeError(
            f"{HASH_KEY_ENV} must be set to a secret of at least {MIN_HASH_KEY_LENGTH} "
            "characters (see .env.example)"
        )
    return key
