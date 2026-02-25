"""
Link panel members to credit transactions using hashed customer IDs.
Input: landing/parquet files
Output: landing/linked_panel_credit.parquet
"""

from __future__ import annotations

import pandas as pd

from src.config import LANDING_DIR, hash_key
from src.utils.hashing import hash_id
from src.utils.validation import check_required_columns

REQUIRED_LINK_COLS = ["customer_id"]


def _canonical_region(linked: pd.DataFrame) -> pd.DataFrame:
    """Set canonical 'region' from credit or panel if missing."""
    if "region" in linked.columns:
        return linked
    if "region_credit" in linked.columns or "region_panel" in linked.columns:
        linked = linked.copy()
        linked["region"] = linked.get(
            "region_credit", pd.Series(index=linked.index)
        ).fillna(linked.get("region_panel", pd.Series(index=linked.index)))
    return linked


def _with_pseudonym(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Drop rows without a customer_id and replace the raw ID with its HMAC pseudonym."""
    ids = df["customer_id"].astype("string").str.strip()
    out = df.loc[ids.notna() & (ids != "")].copy()
    out["cust_hash"] = ids[out.index].map(lambda raw: hash_id(raw, key))
    return out.drop(columns="customer_id")


def run() -> None:
    credit = pd.read_parquet(LANDING_DIR / "credit_txn.parquet")
    panel = pd.read_parquet(LANDING_DIR / "panel.parquet")
    check_required_columns(credit, REQUIRED_LINK_COLS, "credit_txn")
    check_required_columns(panel, REQUIRED_LINK_COLS, "panel")

    key = hash_key()
    credit = _with_pseudonym(credit, key)
    panel = _with_pseudonym(panel, key)
    if panel["cust_hash"].duplicated().any():
        raise ValueError("panel has more than one row for the same customer_id")

    # Inner join: keep matched records only (simplified linkage). Raw IDs stay
    # in the landing files but are not carried into the linked table.
    linked = panel.merge(
        credit, on="cust_hash", suffixes=("_panel", "_credit"), validate="one_to_many"
    )

    linked = _canonical_region(linked)
    linked.to_parquet(LANDING_DIR / "linked_panel_credit.parquet", index=False)
    print("Linked records:", len(linked))


if __name__ == "__main__":
    run()
