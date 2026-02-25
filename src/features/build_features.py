"""
Build aggregated features at SKU x region x week level.
Produces features.parquet with historical targets and simple lags.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LANDING_DIR, FEATURE_DIR
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
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def build() -> Path:
    linked = pd.read_parquet(LANDING_DIR / "linked_panel_credit.parquet")
    check_required_columns(linked, REQUIRED_LINKED_COLS, "linked_panel_credit")

    # convert txn_date to weekly period start
    linked["txn_date"] = pd.to_datetime(linked["txn_date"])  # type: ignore[arg-type]
    linked["week"] = linked["txn_date"].dt.to_period("W").apply(lambda r: r.start_time)

    # Aggregate to SKU x region x week
    grouped = (
        linked.groupby(["sku", "region", "week"]).agg(
            units_sold=("quantity", "sum"),
            sales_dollars=("amount", "sum"),
            n_transactions=("txn_id", "nunique"),
            avg_price=("price", "mean"),
        )
    ).reset_index()

    # Sort chronologically inside groups
    grouped = grouped.sort_values(["sku", "region", "week"]).reset_index(drop=True)

    # Add rolling features per SKU×Region
    grouped["lag_1_units"] = grouped.groupby(["sku", "region"])["units_sold"].shift(1)
    grouped["rolling_4w_mean"] = (
        grouped.groupby(["sku", "region"])["units_sold"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Calendar features
    grouped["weekofyear"] = (
        pd.to_datetime(grouped["week"]).dt.isocalendar().week.astype(int)
    )
    grouped["month"] = pd.to_datetime(grouped["week"]).dt.month.astype(int)

    out = FEATURE_DIR / "features.parquet"
    grouped.to_parquet(out, index=False)
    print("Saved features:", out)
    return out


if __name__ == "__main__":
    build()
