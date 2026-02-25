"""Tests for build_features: build() with minimal linked DataFrame, required columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_build_requires_linked_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.features.build_features as bf

    monkeypatch.setattr(bf, "LANDING_DIR", tmp_path)
    monkeypatch.setattr(bf, "FEATURE_DIR", tmp_path / "feat")
    (tmp_path / "feat").mkdir(parents=True, exist_ok=True)
    # Linked missing 'price'
    linked = pd.DataFrame(
        {
            "txn_date": ["2025-01-06"],
            "sku": ["S1"],
            "region": ["NE"],
            "quantity": [5],
            "amount": [12.5],
            "txn_id": [1],
        }
    )
    linked.to_parquet(tmp_path / "linked_panel_credit.parquet", index=False)
    with pytest.raises(
        ValueError, match="linked_panel_credit missing required columns"
    ):
        bf.build()


def test_build_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.features.build_features as bf

    monkeypatch.setattr(bf, "LANDING_DIR", tmp_path)
    monkeypatch.setattr(bf, "FEATURE_DIR", tmp_path / "feat")
    (tmp_path / "feat").mkdir(parents=True, exist_ok=True)
    linked = pd.DataFrame(
        {
            "txn_date": ["2025-01-06", "2025-01-13"],
            "sku": ["S1", "S1"],
            "region": ["NE", "NE"],
            "quantity": [5, 3],
            "amount": [12.5, 7.5],
            "txn_id": [1, 2],
            "price": [2.5, 2.5],
        }
    )
    linked.to_parquet(tmp_path / "linked_panel_credit.parquet", index=False)
    out_path = bf.build()
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert "units_sold" in df.columns and "lag_1_units" in df.columns


def test_lags_use_calendar_weeks_and_only_past_values() -> None:
    from src.features.build_features import add_lag_features, complete_weeks

    weekly = pd.DataFrame(
        {
            "sku": ["S1"] * 3,
            "region": ["NE"] * 3,
            "week": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-27"]),
            "units_sold": [4.0, 6.0, 10.0],
            "sales_dollars": [8.0, 12.0, 20.0],
            "n_transactions": [1, 2, 3],
            "avg_price": [2.0, 2.0, 2.0],
        }
    )
    out = add_lag_features(complete_weeks(weekly))
    assert out["week"].dt.strftime("%m-%d").tolist() == [
        "01-06",
        "01-13",
        "01-20",
        "01-27",
    ]
    assert out["units_sold"].tolist() == [4.0, 6.0, 0.0, 10.0]
    assert pd.isna(out.loc[2, "avg_price"])
    # 2025-01-27 follows a zero week, not the 6-unit week
    assert out.loc[3, "lag_1_units"] == 0.0
    assert out.loc[3, "lag_2_units"] == 6.0
    # trailing mean never includes the current week's target
    assert pd.isna(out.loc[0, "rolling_4w_mean"])
    assert out.loc[1, "rolling_4w_mean"] == 4.0
    assert out.loc[3, "rolling_4w_mean"] == (4.0 + 6.0 + 0.0) / 3


def test_series_do_not_leak_into_each_other() -> None:
    from src.features.build_features import add_lag_features

    weekly = pd.DataFrame(
        {
            "sku": ["S1", "S2"],
            "region": ["NE", "NE"],
            "week": pd.to_datetime(["2025-01-06", "2025-01-06"]),
            "units_sold": [4.0, 9.0],
        }
    )
    out = add_lag_features(weekly)
    assert out["lag_1_units"].isna().all()
    assert out["rolling_4w_mean"].isna().all()


def test_every_series_runs_to_the_last_week_in_the_data() -> None:
    from src.features.build_features import complete_weeks

    weekly = pd.DataFrame(
        {
            "sku": ["S1", "S1", "S2"],
            "region": ["NE", "NE", "NE"],
            "week": pd.to_datetime(["2025-01-06", "2025-01-20", "2025-01-13"]),
            "units_sold": [1.0, 2.0, 5.0],
        }
    )
    out = complete_weeks(weekly)
    s2 = out[out["sku"] == "S2"]
    assert s2["week"].dt.strftime("%m-%d").tolist() == ["01-13", "01-20"]
    assert s2["units_sold"].tolist() == [5.0, 0.0]
