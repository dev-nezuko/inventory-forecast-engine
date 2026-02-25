"""Tests for ensemble: load_optional (missing file -> None), ensemble() with mock DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ensemble.ensemble_and_reconcile import ensemble, load_optional


def test_load_optional_missing_returns_none(tmp_path: Path) -> None:
    assert load_optional(tmp_path / "nonexistent.parquet") is None


def test_load_optional_reads_file(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "x.parquet"
    df.to_parquet(path, index=False)
    out = load_optional(path)
    assert out is not None
    pd.testing.assert_frame_equal(out, df)


def test_ensemble_with_gbm_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ensemble.ensemble_and_reconcile as ens

    monkeypatch.setattr(ens, "FEATURE_DIR", tmp_path)
    gbm = pd.DataFrame(
        {
            "sku": ["S1"],
            "region": ["NE"],
            "date": ["2025-02-01"],
            "q0.1_gbm": [1.0],
            "q0.5_gbm": [2.0],
            "q0.9_gbm": [3.0],
        }
    )
    gbm.to_parquet(tmp_path / "forecast_gbm.parquet", index=False)
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


def _model_frame(name: str, dates: list[str], q10, q50, q90) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["S1"] * len(dates),
            "region": ["NE"] * len(dates),
            "date": dates,
            f"q0.1_{name}": q10,
            f"q0.5_{name}": q50,
            f"q0.9_{name}": q90,
        }
    )


def test_partial_overlap_is_not_averaged_with_zero() -> None:
    from src.ensemble.ensemble_and_reconcile import combine

    gbm = _model_frame("gbm", ["2025-02-03", "2025-02-10"], [8, 8], [10, 10], [12, 12])
    deepar = _model_frame("deepar", ["2025-02-03"], [4], [6], [8])
    out = combine({"gbm": gbm, "deepar": deepar})
    assert out["q0.5"].tolist() == [8.0, 10.0]


def test_crossing_quantiles_are_reordered_and_clipped() -> None:
    from src.ensemble.ensemble_and_reconcile import combine

    gbm = _model_frame("gbm", ["2025-02-03"], [9.0], [5.0], [-1.0])
    out = combine({"gbm": gbm})
    row = out.iloc[0]
    assert row["q0.1"] <= row["q0.5"] <= row["q0.9"]
    assert (row["q0.1"], row["q0.5"], row["q0.9"]) == (0.0, 5.0, 9.0)


def test_missing_quantile_column_is_reported() -> None:
    from src.ensemble.ensemble_and_reconcile import combine

    bad = _model_frame("gbm", ["2025-02-03"], [1], [2], [3]).drop(columns="q0.9_gbm")
    with pytest.raises(ValueError, match="missing columns"):
        combine({"gbm": bad})


def test_ensemble_with_both_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ensemble.ensemble_and_reconcile as ens

    monkeypatch.setattr(ens, "FEATURE_DIR", tmp_path)
    _model_frame("gbm", ["2025-02-03"], [2], [4], [6]).to_parquet(
        tmp_path / "forecast_gbm.parquet", index=False
    )
    _model_frame("deepar", ["2025-02-03"], [4], [6], [8]).to_parquet(
        tmp_path / "forecast_deepar.parquet", index=False
    )
    ens.ensemble()
    out = pd.read_parquet(tmp_path / "forecast_ensemble.parquet")
    assert out[["q0.1", "q0.5", "q0.9"]].iloc[0].tolist() == [3.0, 5.0, 7.0]
