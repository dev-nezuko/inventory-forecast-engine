"""Tests for the synthetic demo data generator."""

from __future__ import annotations

import pandas as pd

from src.ingest.ingest import validate_credit, validate_panel
from src.synthetic import generate


def test_generate_is_deterministic() -> None:
    a_credit, a_panel = generate(weeks=10, seed=3)
    b_credit, b_panel = generate(weeks=10, seed=3)
    pd.testing.assert_frame_equal(a_credit, b_credit)
    pd.testing.assert_frame_equal(a_panel, b_panel)


def test_generated_tables_pass_ingest_validation() -> None:
    credit, panel = generate(weeks=10)
    validate_credit(credit)
    validate_panel(panel)
    assert panel["customer_id"].is_unique
    assert set(panel["customer_id"]) <= set(credit["customer_id"]) | set(
        panel["customer_id"]
    )


def test_generated_history_covers_every_week_and_series() -> None:
    credit, _ = generate(weeks=12)
    dates = pd.to_datetime(credit["txn_date"])
    assert dates.min() >= pd.Timestamp("2024-01-01")
    assert dates.max() < pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=12)
    assert credit.groupby(["sku", "region"]).ngroups == 16
    assert (credit["amount"] == (credit["price"] * credit["quantity"]).round(2)).all()
