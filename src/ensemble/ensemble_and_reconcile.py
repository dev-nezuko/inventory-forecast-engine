"""
Simple ensembling and placeholder reconciliation.
- Reads LightGBM and optional DeepAR forecasts
- Produces ensemble median and writes forecast_ensemble.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import FEATURE_DIR

logger = logging.getLogger(__name__)


def load_optional(path: Path) -> pd.DataFrame | None:
    """Load parquet at path; return None only if file is missing or unreadable (not for other errors)."""
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None
    except PermissionError:
        logger.warning("Permission denied reading %s", path)
        return None
    except Exception:
        raise


def ensemble() -> None:
    """Load LGB and/or DeepAR forecasts, merge quantiles, clip to nonnegative, write forecast_ensemble.parquet."""
    lgb_path = FEATURE_DIR / "forecast_lgb.parquet"
    deepar_path = FEATURE_DIR / "forecast_deepar.parquet"

    lgb = load_optional(lgb_path)
    dp = load_optional(deepar_path)

    # Use whichever forecast is available
    if lgb is not None:
        df = lgb.copy()
    elif dp is not None:
        df = dp.copy()
    else:
        raise FileNotFoundError("No forecast files found (neither LGB nor DeepAR)")

    if lgb is not None and dp is not None:
        # Both forecasts available: merge and average
        df = lgb.merge(dp, on=["sku", "region", "date"], how="outer")
        df["q0.1"] = 0.5 * df.get("q0.1_lgb", 0).fillna(0) + 0.5 * df.get(
            "q0.1_deepar", 0
        ).fillna(0)
        df["q0.5"] = 0.5 * df.get("q0.5_lgb", 0).fillna(0) + 0.5 * df.get(
            "q0.5_deepar", 0
        ).fillna(0)
        df["q0.9"] = 0.5 * df.get("q0.9_lgb", 0).fillna(0) + 0.5 * df.get(
            "q0.9_deepar", 0
        ).fillna(0)
    elif lgb is not None:
        # Only LGB available
        df["q0.1"] = df.get("q0.1_lgb", 0)
        df["q0.5"] = df.get("q0.5_lgb", 0)
        df["q0.9"] = df.get("q0.9_lgb", 0)
    else:
        # Only DeepAR available
        df["q0.1"] = df.get("q0.1_deepar", 0)
        df["q0.5"] = df.get("q0.5_deepar", 0)
        df["q0.9"] = df.get("q0.9_deepar", 0)

    # Clamp to nonnegative so forecasts are valid for downstream use.
    for c in ["q0.1", "q0.5", "q0.9"]:
        df[c] = df[c].clip(lower=0.0)

    out_cols = ["sku", "region", "date", "q0.1", "q0.5", "q0.9"]
    out = FEATURE_DIR / "forecast_ensemble.parquet"
    df[out_cols].to_parquet(out, index=False)
    print("Saved ensemble forecasts:", out)


if __name__ == "__main__":
    ensemble()
