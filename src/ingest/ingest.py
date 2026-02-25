"""
Ingest raw CSVs (credit card, panel, demographics) and write Parquet landing files.
Usage:
  python -m src.ingest.ingest --credit_csv path/to/credit.csv --panel_csv path/to/panel.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import LANDING_DIR
from src.utils.validation import check_required_columns

# Required columns per README / downstream steps
CREDIT_COLS = [
    "txn_id",
    "customer_id",
    "txn_date",
    "sku",
    "region",
    "amount",
    "price",
    "quantity",
]
PANEL_COLS = [
    "panel_id",
    "customer_id",
    "age_group",
    "region",
    "income_bin",
    "household_size",
]


class IngestError(Exception):
    """Raised when ingest read/write or validation fails."""


def read_csv(path: str | Path) -> pd.DataFrame:
    """Thin wrapper: read CSV from path. Raises IngestError on file or parse errors."""
    path = Path(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        raise IngestError(f"File not found: {path}") from None
    except PermissionError:
        raise IngestError(f"Permission denied: {path}") from None
    except pd.errors.EmptyDataError:
        raise IngestError(f"Empty or invalid CSV: {path}") from None


def write_parquet(df: pd.DataFrame, name: str) -> None:
    """Thin wrapper: write DataFrame to LANDING_DIR / f'{name}.parquet'. Raises IngestError on failure."""
    out = LANDING_DIR / f"{name}.parquet"
    try:
        LANDING_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    except PermissionError:
        raise IngestError(f"Permission denied writing: {out}") from None
    except OSError as e:
        raise IngestError(f"Failed to write {out}: {e}") from e
    print(f"Wrote {out}")


def _examples(mask: pd.Series) -> str:
    rows = [str(i + 2) for i in mask[mask].index[:5]]  # +2: header line, 1-based
    return f"{int(mask.sum())} row(s), e.g. CSV line {', '.join(rows)}"


def validate_credit(credit: pd.DataFrame) -> pd.DataFrame:
    """Check types and keys of the transaction table; return it with parsed dates."""
    check_required_columns(credit, CREDIT_COLS, "credit_csv")
    out = credit.copy()
    dates = pd.to_datetime(out["txn_date"], errors="coerce", format="ISO8601")
    if dates.isna().any():
        raise ValueError(
            f"credit_csv txn_date is not an ISO date in {_examples(dates.isna())}"
        )
    out["txn_date"] = dates
    for col in ("amount", "price", "quantity"):
        values = pd.to_numeric(out[col], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"credit_csv {col} is not numeric in {_examples(values.isna())}"
            )
        out[col] = values
    if (out["price"] < 0).any():
        raise ValueError(
            f"credit_csv price is negative in {_examples(out['price'] < 0)}"
        )
    if out["txn_id"].duplicated().any():
        raise ValueError(
            f"credit_csv txn_id is duplicated in {_examples(out['txn_id'].duplicated())}"
        )
    return out


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Check the panel table: one row per panel member."""
    check_required_columns(panel, PANEL_COLS, "panel_csv")
    if panel["panel_id"].duplicated().any():
        raise ValueError(
            f"panel_csv panel_id is duplicated in {_examples(panel['panel_id'].duplicated())}"
        )
    return panel


def main(credit_csv: str, panel_csv: str, demo_csv: str | None = None) -> None:
    """Validate the credit and panel CSVs, then land both as Parquet (neither on failure)."""
    credit = validate_credit(read_csv(credit_csv))
    panel = validate_panel(read_csv(panel_csv))
    write_parquet(credit, "credit_txn")
    write_parquet(panel, "panel")

    if demo_csv:
        demo = read_csv(demo_csv)
        write_parquet(demo, "demographics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--credit_csv", required=True)
    parser.add_argument("--panel_csv", required=True)
    parser.add_argument("--demo_csv", required=False)
    args = parser.parse_args()
    main(args.credit_csv, args.panel_csv, args.demo_csv)
