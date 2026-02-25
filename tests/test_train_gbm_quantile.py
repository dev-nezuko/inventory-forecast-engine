"""Tests for the gradient-boosted quantile baseline, on small seeded synthetic data."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import weekly_features
from src.models.train_gbm_quantile import (
    QCOLS,
    QuantileGBM,
    backtest,
    main,
    train_and_forecast,
)
from src.synthetic import generate


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    credit, _ = generate(weeks=40, n_customers=120, seed=11)
    return weekly_features(credit)


def _weekly(features: pd.DataFrame) -> pd.DataFrame:
    return features[["sku", "region", "week", "units_sold"]]


def _skus_regions(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    return sorted(features["sku"].unique()), sorted(features["region"].unique())


def test_forecast_shape_dates_and_quantile_order(features: pd.DataFrame) -> None:
    weekly = _weekly(features)
    model = QuantileGBM(*_skus_regions(features)).fit(weekly)
    fc = model.forecast(weekly, horizon=5)

    assert len(fc) == 16 * 5
    last = weekly["week"].max()
    expected = [last + pd.Timedelta(weeks=i) for i in range(1, 6)]
    for _, g in fc.groupby(["sku", "region"]):
        assert g["date"].tolist() == expected
    values = fc[QCOLS].to_numpy()
    assert (values >= 0).all()
    assert (np.diff(values, axis=1) >= 0).all()


def test_training_is_deterministic(features: pd.DataFrame) -> None:
    weekly = _weekly(features)
    a = QuantileGBM(*_skus_regions(features)).fit(weekly).forecast(weekly, 3)
    b = QuantileGBM(*_skus_regions(features)).fit(weekly).forecast(weekly, 3)
    pd.testing.assert_frame_equal(a, b)


def test_backtest_ignores_weeks_after_the_cutoff(features: pd.DataFrame) -> None:
    holdout = 6
    base = backtest(features, holdout)
    cutoff = pd.Timestamp(base["holdout_weeks"].split(" to ")[0])

    tampered = features.copy()
    after = tampered["week"] >= cutoff
    tampered.loc[after, "units_sold"] = tampered.loc[after, "units_sold"] * 3
    again = backtest(tampered, holdout)

    # Only the actuals changed, so every score moves but the trained models and
    # forecasts (and therefore the training window) are identical.
    assert again["train_weeks"] == base["train_weeks"]
    assert again["wape"]["gbm_median"] != base["wape"]["gbm_median"]
    weekly = _weekly(features)
    train = weekly[weekly["week"] < cutoff]
    fc_a = QuantileGBM(*_skus_regions(features)).fit(train).forecast(train, holdout)
    train_t = _weekly(tampered)[_weekly(tampered)["week"] < cutoff]
    fc_b = QuantileGBM(*_skus_regions(features)).fit(train_t).forecast(train_t, holdout)
    pd.testing.assert_frame_equal(fc_a, fc_b)


def test_backtest_reports_labelled_metrics(features: pd.DataFrame) -> None:
    m = backtest(features, holdout=6)
    assert m["data"].startswith("synthetic")
    assert m["series"] == 16 and m["rows_scored"] == 16 * 6
    assert set(m["wape"]) == {
        "gbm_median",
        "trailing_4w_mean",
    }  # 40 weeks: no last-year baseline
    assert set(m["pinball_loss"]) == set(QCOLS)
    assert 0.0 <= m["interval_coverage_q0.1_q0.9"] <= 1.0


def test_backtest_needs_enough_history(features: pd.DataFrame) -> None:
    short = features[features["week"] < features["week"].min() + pd.Timedelta(weeks=10)]
    with pytest.raises(ValueError, match="Need at least"):
        backtest(short, holdout=6)


def test_train_and_forecast_writes_artifacts(
    features: pd.DataFrame, tmp_path: Path
) -> None:
    features.to_parquet(tmp_path / "features.parquet", index=False)
    forecast_path, metrics_path, metrics = train_and_forecast(
        tmp_path, horizon=4, holdout=6
    )

    fc = pd.read_parquet(forecast_path)
    assert list(fc.columns) == [
        "sku",
        "region",
        "date",
        "q0.1_gbm",
        "q0.5_gbm",
        "q0.9_gbm",
    ]
    assert len(fc) == 16 * 4
    assert json.loads(metrics_path.read_text()) == metrics


def test_cli_rejects_out_of_range_horizon() -> None:
    with pytest.raises(SystemExit):
        main(["--horizon", "0"])


def test_mlflow_is_only_used_when_tracking_uri_is_set(
    features: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.models.train_gbm_quantile as mod

    calls: list[tuple] = []

    class _Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = types.SimpleNamespace(
        start_run=lambda: (calls.append(("start",)), _Run())[1],
        log_param=lambda k, v: calls.append(("param", k, v)),
        log_metric=lambda k, v: calls.append(("metric", k, v)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    sys.modules.pop("src.mlops.mlflow_utils", None)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    monkeypatch.setattr(mod, "FEATURE_DIR", tmp_path)

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    main(["--horizon", "2", "--holdout", "6"])
    assert calls == []

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/unused")
    main(["--horizon", "2", "--holdout", "6"])
    assert ("start",) in calls
    assert any(c[0] == "metric" and c[1] == "wape_gbm_median" for c in calls)
    sys.modules.pop("src.mlops.mlflow_utils", None)
