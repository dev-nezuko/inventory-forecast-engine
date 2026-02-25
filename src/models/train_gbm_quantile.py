"""
Gradient-boosted quantile baseline on the weekly SKU x region features.

One scikit-learn HistGradientBoostingRegressor (pinball loss) per quantile,
trained on leakage-safe inputs: the previous two weeks' units, the mean of the
four weeks before, a yearly sine/cosine and SKU/region codes. Multi-week
forecasts are recursive: each week's median is fed back as the next week's lag.

Writes two files to FEATURE_DIR:
- gbm_backtest.json: a time-based holdout. The last --holdout weeks are hidden,
  forecast recursively from the cutoff and scored against two naive baselines.
- forecast_gbm.parquet: a refit on every week, forecasting --horizon weeks ahead.

Run from repo root: python -m src.models.train_gbm_quantile --horizon 12 --holdout 12
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import FEATURE_DIR
from src.features.build_features import SERIES, add_lag_features
from src.utils.metrics import quantile_loss, wape
from src.utils.quantiles import enforce_quantile_order

QUANTILES = (0.1, 0.5, 0.9)
QCOLS = [f"q{q}" for q in QUANTILES]
FEATURES = ["lag_1_units", "lag_2_units", "rolling_4w_mean", "week_sin", "week_cos"]
CATEGORICAL = ["sku_code", "region_code"]
SEED = 7
MIN_TRAIN_WEEKS = 8


def _design(df: pd.DataFrame, skus: list[str], regions: list[str]) -> pd.DataFrame:
    """Add calendar terms and integer SKU/region codes to a frame with lag features."""
    weekofyear = df["week"].dt.isocalendar().week.astype(float)
    return df.assign(
        week_sin=np.sin(2 * np.pi * weekofyear / 52.0),
        week_cos=np.cos(2 * np.pi * weekofyear / 52.0),
        sku_code=pd.Categorical(df["sku"], categories=skus).codes,
        region_code=pd.Categorical(df["region"], categories=regions).codes,
    )


class QuantileGBM:
    """One pinball-loss gradient-boosting model per quantile."""

    def __init__(self, skus: list[str], regions: list[str], quantiles=QUANTILES):
        self.skus, self.regions, self.quantiles = skus, regions, tuple(quantiles)
        self.models: dict[float, HistGradientBoostingRegressor] = {}

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _design(frame, self.skus, self.regions)[FEATURES + CATEGORICAL]

    def fit(self, weekly: pd.DataFrame) -> QuantileGBM:
        """Fit on weekly rows that have two weeks of lags."""
        train = add_lag_features(weekly).dropna(subset=["lag_2_units", "units_sold"])
        if len(train) < 20:
            raise ValueError(
                f"Not enough weekly history to train ({len(train)} usable rows)"
            )
        X, y = self._matrix(train), train["units_sold"].astype(float)
        cat_mask = [c in CATEGORICAL for c in X.columns]
        for q in self.quantiles:
            self.models[q] = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                categorical_features=cat_mask,
                early_stopping=False,
                random_state=SEED,
            ).fit(X, y)
        return self

    def forecast(self, weekly: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast `horizon` weeks after each series' last week, feeding medians back."""
        frame = weekly[[*SERIES, "week", "units_sold"]].copy()
        produced = []
        for _ in range(horizon):
            nxt = frame.groupby(SERIES, as_index=False)["week"].max()
            nxt["week"] = nxt["week"] + pd.Timedelta(weeks=1)
            nxt["units_sold"] = np.nan
            frame = add_lag_features(pd.concat([frame, nxt], ignore_index=True))
            step = frame[frame["units_sold"].isna()].copy()
            X = self._matrix(step)
            for q in self.quantiles:
                step[f"q{q}"] = self.models[q].predict(X)
            step[QCOLS] = step[QCOLS].clip(lower=0.0)
            step = enforce_quantile_order(step, QCOLS)
            produced.append(step[[*SERIES, "week", *QCOLS]])
            frame.loc[step.index, "units_sold"] = step["q0.5"]
            frame = frame[[*SERIES, "week", "units_sold"]]
        out = pd.concat(produced, ignore_index=True).rename(columns={"week": "date"})
        return out.sort_values([*SERIES, "date"]).reset_index(drop=True)


def _weekly(features: pd.DataFrame) -> pd.DataFrame:
    return features[[*SERIES, "week", "units_sold"]].assign(
        week=pd.to_datetime(features["week"])
    )


def backtest(features: pd.DataFrame, holdout: int) -> dict:
    """Hide the last `holdout` weeks of every series, forecast them and score the result."""
    weekly = _weekly(features)
    weeks = np.sort(weekly["week"].unique())
    if len(weeks) < holdout + MIN_TRAIN_WEEKS:
        raise ValueError(
            f"Need at least {holdout + MIN_TRAIN_WEEKS} weeks for this holdout"
        )
    cutoff = pd.Timestamp(weeks[-holdout])
    train = weekly[weekly["week"] < cutoff]
    test = weekly[weekly["week"] >= cutoff]

    skus, regions = sorted(weekly["sku"].unique()), sorted(weekly["region"].unique())
    model = QuantileGBM(skus, regions).fit(train)
    fc = model.forecast(train, holdout).rename(columns={"date": "week"})
    scored = test.merge(fc, on=[*SERIES, "week"], how="inner")

    trailing = train.groupby(SERIES)["units_sold"].apply(lambda s: s.tail(4).mean())
    scored = scored.join(trailing.rename("trailing_4w_mean"), on=SERIES)
    last_year = weekly.assign(week=weekly["week"] + pd.Timedelta(weeks=52))
    scored = scored.merge(
        last_year.rename(columns={"units_sold": "same_week_last_year"}),
        on=[*SERIES, "week"],
        how="left",
    )

    y = scored["units_sold"]
    result = {
        "data": "synthetic (examples/sample_data, generated by src.synthetic)",
        "train_weeks": f"{train['week'].min():%Y-%m-%d} to {train['week'].max():%Y-%m-%d}",
        "holdout_weeks": f"{cutoff:%Y-%m-%d} to {test['week'].max():%Y-%m-%d}",
        "series": int(scored.groupby(SERIES).ngroups),
        "rows_scored": len(scored),
        "wape": {"gbm_median": round(wape(y, scored["q0.5"]), 4)},
        "pinball_loss": {
            c: round(quantile_loss(y, scored[c], q), 4)
            for q, c in zip(QUANTILES, QCOLS, strict=True)
        },
        "interval_coverage_q0.1_q0.9": round(
            float(((y >= scored["q0.1"]) & (y <= scored["q0.9"])).mean()), 4
        ),
    }
    result["wape"]["trailing_4w_mean"] = round(wape(y, scored["trailing_4w_mean"]), 4)
    seasonal = scored.dropna(subset=["same_week_last_year"])
    if len(seasonal) == len(scored):
        result["wape"]["same_week_last_year"] = round(
            wape(y, seasonal["same_week_last_year"]), 4
        )
    return result


def train_and_forecast(
    feature_dir: Path, horizon: int, holdout: int
) -> tuple[Path, Path, dict]:
    """Run the holdout, refit on all weeks and write the forecast and metrics files."""
    features = pd.read_parquet(feature_dir / "features.parquet")
    metrics = backtest(features, holdout)
    metrics_path = feature_dir / "gbm_backtest.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    weekly = _weekly(features)
    skus, regions = sorted(weekly["sku"].unique()), sorted(weekly["region"].unique())
    fc = QuantileGBM(skus, regions).fit(weekly).forecast(weekly, horizon)
    fc = fc.rename(columns={c: f"{c}_gbm" for c in QCOLS})
    forecast_path = feature_dir / "forecast_gbm.parquet"
    fc.to_parquet(forecast_path, index=False)
    return forecast_path, metrics_path, metrics


def _log_to_mlflow(params: dict, metrics: dict) -> None:
    """Log the run only when MLFLOW_TRACKING_URI is set; mlflow is an optional install."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    from src.mlops.mlflow_utils import log_metrics, start_run

    flat = {f"wape_{k}": v for k, v in metrics["wape"].items()}
    flat.update({f"pinball_{k}": v for k, v in metrics["pinball_loss"].items()})
    flat["interval_coverage"] = metrics["interval_coverage_q0.1_q0.9"]
    with start_run({k: str(v) for k, v in params.items()}):
        log_metrics(flat)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the gradient-boosted quantile baseline"
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--holdout", type=int, default=12)
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= 52 or not 1 <= args.holdout <= 52:
        parser.error("--horizon and --holdout must be between 1 and 52")
    forecast_path, metrics_path, metrics = train_and_forecast(
        FEATURE_DIR, args.horizon, args.holdout
    )
    _log_to_mlflow(vars(args), metrics)
    print("Saved forecasts:", forecast_path)
    print("Saved holdout metrics:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
