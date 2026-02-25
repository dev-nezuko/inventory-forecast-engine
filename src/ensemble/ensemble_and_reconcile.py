"""
Combine model-specific quantile forecasts into forecast_ensemble.parquet.

Each available model file contributes q0.1/q0.5/q0.9 columns with a model
suffix. For every SKU x region x date the ensemble is the mean of the models
that cover that row, clipped at zero and re-sorted so quantiles never cross.
No hierarchical reconciliation is done yet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import FEATURE_DIR
from src.utils.quantiles import enforce_quantile_order

logger = logging.getLogger(__name__)

QUANTILE_COLS = ["q0.1", "q0.5", "q0.9"]
KEYS = ["sku", "region", "date"]
MODEL_FILES = {"lgb": "forecast_lgb.parquet", "deepar": "forecast_deepar.parquet"}


def load_optional(path: Path) -> pd.DataFrame | None:
    """Load parquet at path, or return None if the file does not exist."""
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None


def combine(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average suffixed quantile columns across models, row by row."""
    merged: pd.DataFrame | None = None
    for name, df in forecasts.items():
        cols = [f"{q}_{name}" for q in QUANTILE_COLS]
        missing = [c for c in KEYS + cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} forecast is missing columns {missing}")
        part = df[KEYS + cols].assign(date=pd.to_datetime(df["date"]))
        merged = part if merged is None else merged.merge(part, on=KEYS, how="outer")
    if merged is None:
        raise FileNotFoundError("No forecast files found")

    for q in QUANTILE_COLS:
        merged[q] = merged[[f"{q}_{name}" for name in forecasts]].mean(
            axis=1, skipna=True
        )
    out = merged[KEYS + QUANTILE_COLS].dropna(subset=QUANTILE_COLS)
    out[QUANTILE_COLS] = out[QUANTILE_COLS].clip(lower=0.0)
    out = enforce_quantile_order(out, QUANTILE_COLS)
    return out.sort_values(KEYS).reset_index(drop=True)


def ensemble() -> Path:
    """Combine whichever model forecasts exist and write forecast_ensemble.parquet."""
    forecasts = {}
    for name, fname in MODEL_FILES.items():
        df = load_optional(FEATURE_DIR / fname)
        if df is not None:
            forecasts[name] = df
    if not forecasts:
        raise FileNotFoundError(
            f"No forecast files found in {FEATURE_DIR} ({', '.join(MODEL_FILES.values())})"
        )
    out = combine(forecasts)
    path = FEATURE_DIR / "forecast_ensemble.parquet"
    out.to_parquet(path, index=False)
    logger.info("Combined %s into %d rows", ", ".join(forecasts), len(out))
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Saved ensemble forecasts:", ensemble())
