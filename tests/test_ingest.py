"""Tests for src.ingest: read_csv, write_parquet, main validation and error handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest.ingest import IngestError, read_csv


def test_read_csv_not_found() -> None:
    with pytest.raises(IngestError, match="File not found"):
        read_csv("/nonexistent/path.csv")


def test_write_parquet_and_read_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ingest_mod.write_parquet(df, "test_out")
    assert (tmp_path / "test_out.parquet").exists()
    back = pd.read_parquet(tmp_path / "test_out.parquet")
    pd.testing.assert_frame_equal(back, df)


def test_main_missing_columns_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    credit_csv = tmp_path / "credit.csv"
    panel_csv = tmp_path / "panel.csv"
    # Credit missing 'quantity'
    pd.DataFrame(
        {
            "txn_id": [1],
            "customer_id": ["x"],
            "txn_date": ["2025-01-01"],
            "sku": ["S1"],
            "region": ["NE"],
            "amount": [10.0],
            "price": [2.0],
        }
    ).to_csv(credit_csv, index=False)
    pd.DataFrame(
        {
            "panel_id": ["p1"],
            "customer_id": ["x"],
            "age_group": ["35-54"],
            "region": ["NE"],
            "income_bin": ["50k"],
            "household_size": [3],
        }
    ).to_csv(panel_csv, index=False)
    with pytest.raises(ValueError, match="credit_csv missing required columns"):
        ingest_mod.main(str(credit_csv), str(panel_csv))


def test_main_missing_columns_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    credit_csv = tmp_path / "credit.csv"
    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "txn_id": [1],
            "customer_id": ["x"],
            "txn_date": ["2025-01-01"],
            "sku": ["S1"],
            "region": ["NE"],
            "amount": [10.0],
            "price": [2.0],
            "quantity": [5],
        }
    ).to_csv(credit_csv, index=False)
    # Panel missing household_size
    pd.DataFrame(
        {
            "panel_id": ["p1"],
            "customer_id": ["x"],
            "age_group": ["35-54"],
            "region": ["NE"],
            "income_bin": ["50k"],
        }
    ).to_csv(panel_csv, index=False)
    with pytest.raises(ValueError, match="panel_csv missing required columns"):
        ingest_mod.main(str(credit_csv), str(panel_csv))


def test_main_success_writes_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path)
    credit_csv = tmp_path / "credit.csv"
    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "txn_id": [1],
            "customer_id": ["x"],
            "txn_date": ["2025-01-01"],
            "sku": ["S1"],
            "region": ["NE"],
            "amount": [10.0],
            "price": [2.0],
            "quantity": [5],
        }
    ).to_csv(credit_csv, index=False)
    pd.DataFrame(
        {
            "panel_id": ["p1"],
            "customer_id": ["x"],
            "age_group": ["35-54"],
            "region": ["NE"],
            "income_bin": ["50k"],
            "household_size": [3],
        }
    ).to_csv(panel_csv, index=False)
    ingest_mod.main(str(credit_csv), str(panel_csv))
    assert (tmp_path / "credit_txn.parquet").exists()
    assert (tmp_path / "panel.parquet").exists()


def _credit_frame(**overrides) -> pd.DataFrame:
    data = {
        "txn_id": [1, 2],
        "customer_id": ["x", "y"],
        "txn_date": ["2025-01-01", "2025-01-08"],
        "sku": ["S1", "S1"],
        "region": ["NE", "NE"],
        "amount": [10.0, 4.0],
        "price": [2.0, 2.0],
        "quantity": [5, 2],
    }
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"txn_date": ["2025-01-01", "next tuesday"]},
            "txn_date is not an ISO date in 1 row",
        ),
        ({"quantity": [5, "two"]}, "quantity is not numeric in 1 row"),
        ({"price": [2.0, -1.0]}, "price is negative"),
        ({"txn_id": [7, 7]}, "txn_id is duplicated"),
    ],
)
def test_validate_credit_rejects_bad_rows(overrides: dict, message: str) -> None:
    from src.ingest.ingest import validate_credit

    with pytest.raises(ValueError, match=message):
        validate_credit(_credit_frame(**overrides))


def test_validate_credit_parses_dates() -> None:
    from src.ingest.ingest import validate_credit

    out = validate_credit(_credit_frame())
    assert str(out["txn_date"].dtype).startswith("datetime64")


def test_nothing_is_landed_when_panel_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "LANDING_DIR", tmp_path / "landing")
    _credit_frame().to_csv(tmp_path / "credit.csv", index=False)
    pd.DataFrame({"panel_id": ["p1"]}).to_csv(tmp_path / "panel.csv", index=False)
    with pytest.raises(ValueError, match="panel_csv missing"):
        ingest_mod.main(str(tmp_path / "credit.csv"), str(tmp_path / "panel.csv"))
    assert not (tmp_path / "landing").exists()
