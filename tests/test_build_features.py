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
