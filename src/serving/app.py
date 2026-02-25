"""FastAPI app serving stored quantile forecasts: /health and /forecast."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import FEATURE_DIR

app = FastAPI(title="Inventory forecast API")

FORECAST_COLS = ["q0.1", "q0.5", "q0.9"]
FORECAST_FILES = ["forecast_ensemble.parquet", "forecast_gbm.parquet"]
# Identifiers are only compared against artifact rows, never used to build a
# path, but they are still restricted to a plain token so nothing else can be.
IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=1, max_length=64, pattern=IDENTIFIER)
    region: str = Field(min_length=1, max_length=32, pattern=IDENTIFIER)
    horizon_weeks: int = Field(default=12, ge=1, le=52)


class ForecastRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sku: str
    region: str
    date: str
    q10: float = Field(alias="q0.1", serialization_alias="q0.1")
    q50: float = Field(alias="q0.5", serialization_alias="q0.5")
    q90: float = Field(alias="q0.9", serialization_alias="q0.9")


def _normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map model-suffixed quantile columns (q0.5_gbm, q0.5_deepar) to q0.1/q0.5/q0.9."""
    renames = {}
    for c in FORECAST_COLS:
        if c in df.columns:
            continue
        for suffix in ("_gbm", "_deepar"):
            if f"{c}{suffix}" in df.columns:
                renames[f"{c}{suffix}"] = c
                break
    return df.rename(columns=renames) if renames else df


@lru_cache(maxsize=4)
def _read_forecast(path: Path, mtime_ns: int) -> pd.DataFrame:
    """Parse an artifact once per (path, modification time)."""
    df = _normalize_forecast_columns(pd.read_parquet(path))
    missing = [
        c for c in ["sku", "region", "date", *FORECAST_COLS] if c not in df.columns
    ]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sku", "region", "date"]).reset_index(drop=True)


def _load_any_forecast() -> pd.DataFrame:
    """Load the first available forecast artifact (ensemble, then gbm)."""
    for fname in FORECAST_FILES:
        path = FEATURE_DIR / fname
        if path.is_file():
            return _read_forecast(path, path.stat().st_mtime_ns)
    raise FileNotFoundError(
        "No forecast file found. Run the trainer or the ensemble first."
    )


@app.get("/health")
def health() -> dict:
    """Liveness check: returns status ok."""
    return {"status": "ok"}


@app.post("/forecast", response_model=list[ForecastRow], response_model_by_alias=True)
def forecast(q: Query) -> list[dict]:
    """Return the first `horizon_weeks` stored quantile rows for one SKU and region."""
    try:
        df = _load_any_forecast()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail="No forecast data available. Please run the pipeline to generate forecasts.",
        ) from e

    rows = df[(df["sku"] == q.sku) & (df["region"] == q.region)].head(q.horizon_weeks)
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for sku={q.sku!r} and region={q.region!r}.",
        )
    out = rows[["sku", "region", "date", *FORECAST_COLS]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")
