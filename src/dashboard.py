"""Data access for the Streamlit dashboard, kept free of Streamlit so it can be tested."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

QCOLS = ["q0.1", "q0.5", "q0.9"]


@dataclass
class Artifacts:
    """Whatever pipeline outputs exist under one DATA_DIR, plus load problems."""

    credit: pd.DataFrame | None = None
    panel: pd.DataFrame | None = None
    linked: pd.DataFrame | None = None
    weighted: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    forecast: pd.DataFrame | None = None
    forecast_source: str | None = None
    backtest: dict | None = None
    problems: list[str] = field(default_factory=list)

    def series(self) -> pd.DataFrame:
        """SKU x region pairs that have features, sorted."""
        if self.features is None:
            return pd.DataFrame(columns=["sku", "region"])
        pairs = self.features[["sku", "region"]].drop_duplicates()
        return pairs.sort_values(["sku", "region"]).reset_index(drop=True)

    def busiest_series(self) -> tuple[str, str]:
        """The SKU x region with the most units sold, used as the default selection."""
        totals = self.features.groupby(["sku", "region"])["units_sold"].sum()
        return totals.idxmax()


def _read(path: Path, problems: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as e:
        problems.append(f"Could not read {path.name}: {e}")
        return None


def _normalise_forecast(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    for c in QCOLS:
        for suffix in ("_gbm", "_deepar"):
            if c not in df.columns and f"{c}{suffix}" in df.columns:
                renames[f"{c}{suffix}"] = c
    return df.rename(columns=renames).assign(date=lambda d: pd.to_datetime(d["date"]))


def load_artifacts(data_dir: Path) -> Artifacts:
    landing, features_dir = data_dir / "landing", data_dir / "features"
    a = Artifacts()
    a.credit = _read(landing / "credit_txn.parquet", a.problems)
    a.panel = _read(landing / "panel.parquet", a.problems)
    a.linked = _read(landing / "linked_panel_credit.parquet", a.problems)
    a.weighted = _read(landing / "panel_weighted.parquet", a.problems)
    a.features = _read(features_dir / "features.parquet", a.problems)
    if a.features is not None:
        a.features["week"] = pd.to_datetime(a.features["week"])

    for name in ("forecast_ensemble.parquet", "forecast_gbm.parquet"):
        df = _read(features_dir / name, a.problems)
        if df is not None:
            a.forecast, a.forecast_source = _normalise_forecast(df), name
            break

    backtest = features_dir / "gbm_backtest.json"
    if backtest.is_file():
        try:
            a.backtest = json.loads(backtest.read_text())
        except (OSError, ValueError) as e:
            a.problems.append(f"Could not read {backtest.name}: {e}")
    return a


def series_view(
    a: Artifacts, sku: str, region: str, history_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recent weekly actuals and stored forecast rows for one series."""
    hist = pd.DataFrame(columns=["week", "units_sold"])
    if a.features is not None:
        hist = a.features[(a.features["sku"] == sku) & (a.features["region"] == region)]
        hist = hist.sort_values("week").tail(history_weeks)[["week", "units_sold"]]
    fc = pd.DataFrame(columns=["date", *QCOLS])
    if a.forecast is not None:
        fc = a.forecast[(a.forecast["sku"] == sku) & (a.forecast["region"] == region)]
        fc = fc.sort_values("date")[["date", *QCOLS]]
    return hist.reset_index(drop=True), fc.reset_index(drop=True)


def pipeline_stages(a: Artifacts) -> list[tuple[str, str, str]]:
    """(stage, status, detail) rows describing which outputs exist."""

    def rows(df: pd.DataFrame | None, unit: str) -> str:
        return f"{len(df):,} {unit}" if df is not None else "not produced"

    linked_detail = rows(a.linked, "linked transactions")
    if a.linked is not None and a.credit is not None and len(a.credit):
        linked_detail += f" ({len(a.linked) / len(a.credit):.0%} of all transactions)"
    return [
        (
            "Ingest",
            "done" if a.credit is not None else "missing",
            rows(a.credit, "transactions"),
        ),
        (
            "Panel",
            "done" if a.panel is not None else "missing",
            rows(a.panel, "panel members"),
        ),
        ("HMAC linkage", "done" if a.linked is not None else "missing", linked_detail),
        (
            "Raking weights",
            "done" if a.weighted is not None else "missing",
            rows(a.weighted, "weighted members"),
        ),
        (
            "Weekly features",
            "done" if a.features is not None else "missing",
            rows(a.features, "series-weeks"),
        ),
        (
            "GBM quantile model",
            "done" if a.backtest is not None else "missing",
            "holdout backtest available" if a.backtest else "not trained",
        ),
        ("DeepAR, PyMC", "stub", "entry points raise NotImplementedError"),
    ]


def call_forecast_api(
    api_url: str, sku: str, region: str, weeks: int
) -> tuple[list | None, str | None]:
    """POST /forecast; returns (rows, None) or (None, readable error)."""
    try:
        r = requests.post(
            f"{api_url.rstrip('/')}/forecast",
            json={"sku": sku, "region": region, "horizon_weeks": weeks},
            timeout=5,
        )
    except requests.RequestException as e:
        return None, f"API not reachable at {api_url} ({type(e).__name__})"
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"API answered {r.status_code}: {detail}"
    return r.json(), None
