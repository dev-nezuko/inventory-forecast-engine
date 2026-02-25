import os
from pathlib import Path

# ROOT is the repo root (directory containing src/). Override for tests via PROJECT_ROOT.
ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))

# Data layout (override via env vars)
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "examples" / "sample_data"))
RAW_DIR = DATA_DIR / "raw"
LANDING_DIR = DATA_DIR / "landing"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
SALT = os.environ.get("HASH_SALT", "CHANGE_ME_SECRET_SALT")

# Ensure key dirs exist
for _p in [RAW_DIR, LANDING_DIR, FEATURE_DIR, MODEL_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
