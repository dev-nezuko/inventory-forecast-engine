"""
Build aggregated features at SKU x region x week level.
Produces features.parquet with weekly targets, past-only lags and calendar terms.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FEATURE_DIR, LANDING_DIR
from src.utils.validation import check_required_columns

REQUIRED_LINKED_COLS = [
    "txn_date",
    "sku",
    "region",
    "quantity",
    "amount",
    "txn_id",
    "price",
]
SERIES = ["sku", "region"]
ADDITIVE = ["units_sold", "sales_dollars", "n_transactions"]


def complete_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """Insert zero-sales rows for weeks with no transactions inside each series' span.

    Without this, "lag 1" and the trailing mean silently reached back to the
    last week that happened to have a sale.
    """
    parts = []
    for (sku, region), g in weekly.groupby(SERIES, sort=True):
        weeks = pd.date_range(g["week"].min(), g["week"].max(), freq="W-MON")
        full = g.drop(columns=SERIES).set_index("week").reindex(weeks)
        full = full.rename_axis("week").reset_index()
        full.insert(0, "region", region)
        full.insert(0, "sku", sku)
        parts.append(full)
    out = pd.concat(parts, ignore_index=True)
    present = [c for c in ADDITIVE if c in out.columns]
    out[present] = out[present].fillna(0)
    return out


def add_lag_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1_units, lag_2_units and rolling_4w_mean from strictly earlier weeks.

    Expects one row per series and consecutive week. Rows whose units_sold is
    unknown (future weeks) still get lags from the rows before them.
    """
    df = weekly.sort_values([*SERIES, "week"]).reset_index(drop=True)
    units = df.groupby(SERIES, sort=False)["units_sold"]
    df["lag_1_units"] = units.shift(1)
    df["lag_2_units"] = units.shift(2)
    df["rolling_4w_mean"] = units.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return df


def build() -> Path:
    linked = pd.read_parquet(LANDING_DIR / "linked_panel_credit.parquet")
    check_required_columns(linked, REQUIRED_LINKED_COLS, "linked_panel_credit")

    # Weeks start on Monday
    linked["week"] = (
        pd.to_datetime(linked["txn_date"]).dt.to_period("W-SUN").dt.start_time
    )

    grouped = (
        linked.groupby([*SERIES, "week"])
        .agg(
            units_sold=("quantity", "sum"),
            sales_dollars=("amount", "sum"),
            n_transactions=("txn_id", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )
    grouped = add_lag_features(complete_weeks(grouped))

    grouped["weekofyear"] = grouped["week"].dt.isocalendar().week.astype(int)
    grouped["month"] = grouped["week"].dt.month.astype(int)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURE_DIR / "features.parquet"
    grouped.to_parquet(out, index=False)
    print("Saved features:", out)
    return out


if __name__ == "__main__":
    build()
