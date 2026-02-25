"""FastAPI app for forecast queries: /health and /forecast."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import FEATURE_DIR

app = FastAPI(title="Forecast MVP API")

FORECAST_COLS = ["q0.1", "q0.5", "q0.9"]


class Query(BaseModel):
    sku: str
    region: str
    horizon_weeks: int = 12


def _normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has canonical quantile columns q0.1, q0.5, q0.9 (from _lgb or _deepar suffixes if needed)."""
    if "q0.1" in df.columns:
        return df
    renames = {}
    for c in FORECAST_COLS:
        if f"{c}_lgb" in df.columns:
            renames[f"{c}_lgb"] = c
        elif f"{c}_deepar" in df.columns:
            renames[f"{c}_deepar"] = c
    return df.rename(columns=renames) if renames else df


def _load_any_forecast() -> pd.DataFrame:
    """Load first available forecast parquet (ensemble then lgb). Raises FileNotFoundError if none found."""
    for fname in ["forecast_ensemble.parquet", "forecast_lgb.parquet"]:
        path = FEATURE_DIR / fname
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError("No forecast file found. Run ensemble or train LGB first.")


@app.get("/health")
async def health() -> dict:
    """Liveness check: returns status ok."""
    return {"status": "ok"}


@app.post("/forecast")
async def forecast(q: Query):
    """Return quantile forecasts (q0.1, q0.5, q0.9) for the given sku/region and horizon."""
    try:
        df = _load_any_forecast()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail="No forecast data available. Please run the pipeline to generate forecasts.",
        ) from e

    df = df[(df["sku"] == q.sku) & (df["region"] == q.region)].copy()
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for sku={q.sku!r} and region={q.region!r}.",
        )

    df = df.sort_values("date").head(q.horizon_weeks)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = _normalize_forecast_columns(df)
    return df[["sku", "region", "date"] + FORECAST_COLS].to_dict(orient="records")
