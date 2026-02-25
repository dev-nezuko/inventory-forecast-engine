"""Tests for ensemble: load_optional (missing file -> None), ensemble() with mock DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ensemble.ensemble_and_reconcile import load_optional, ensemble


def test_load_optional_missing_returns_none(tmp_path: Path) -> None:
    assert load_optional(tmp_path / "nonexistent.parquet") is None


def test_load_optional_reads_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "x.parquet"
    df.to_parquet(path, index=False)
    out = load_optional(path)
    assert out is not None
    pd.testing.assert_frame_equal(out, df)


def test_ensemble_with_lgb_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ensemble.ensemble_and_reconcile as ens

    monkeypatch.setattr(ens, "FEATURE_DIR", tmp_path)
    lgb = pd.DataFrame(
        {
            "sku": ["S1"],
            "region": ["NE"],
            "date": ["2025-02-01"],
            "q0.1_lgb": [1.0],
            "q0.5_lgb": [2.0],
            "q0.9_lgb": [3.0],
        }
    )
    lgb.to_parquet(tmp_path / "forecast_lgb.parquet", index=False)
    ensemble()
    out = pd.read_parquet(tmp_path / "forecast_ensemble.parquet")
    assert list(out.columns) == ["sku", "region", "date", "q0.1", "q0.5", "q0.9"]
    assert (out["q0.1"] >= 0).all()


def test_ensemble_no_files_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ensemble.ensemble_and_reconcile as ens

    monkeypatch.setattr(ens, "FEATURE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No forecast files"):
        ens.ensemble()
