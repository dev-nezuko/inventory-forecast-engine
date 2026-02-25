"""
Simple raking / iterative proportional fitting to compute weights.
Inputs:
 - panel_parquet with demographic columns (age_group, region, income_bin)
 - population_marginals: a dict of {col: {value: total_count}}
Outputs:
 - landing/panel_weighted.parquet
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import LANDING_DIR

logger = logging.getLogger(__name__)


def raking(
    df: pd.DataFrame, marginals: dict, max_iter: int = 50, tol: float = 1e-6
) -> pd.DataFrame:
    """Iterative proportional fitting: scale weights so group sums match marginals.

    Every category present in the data needs a target, and all marginals must
    describe the same population total, otherwise IPF cannot converge.
    """
    if df.empty:
        raise ValueError("raking requires a non-empty DataFrame")
    totals = []
    for col, target in marginals.items():
        if col not in df.columns:
            raise ValueError(f"marginals key '{col}' is not a column in df")
        if df[col].isna().any():
            raise ValueError(f"column '{col}' has missing values")
        unknown = sorted(set(df[col].astype(str)) - set(map(str, target)))
        if unknown:
            raise ValueError(f"no '{col}' target for categories {unknown}")
        totals.append(float(sum(target.values())))
    if totals and max(totals) - min(totals) > 1e-9 * max(totals):
        raise ValueError(f"marginal totals differ: {totals}")

    df = df.copy()
    df["weight"] = 1.0
    for it in range(max_iter):
        max_change = 0.0
        for col, target in marginals.items():
            by_name = {str(k): v for k, v in target.items()}
            current = df.groupby(col)["weight"].sum()
            scale = {k: by_name[str(k)] / current[k] for k in current.index}
            m = df[col].map(scale)
            df["weight"] *= m
            max_change = max(max_change, float((1 - m).abs().max()))
        if max_change < tol:
            logger.info("Raking converged after %d iterations", it + 1)
            break
    else:
        logger.warning("Raking did not converge in %d iterations", max_iter)
    return df


def main():
    panel = pd.read_parquet(LANDING_DIR / "panel.parquet")
    # Example marginals (toy). Replace with census/retailer derived targets.
    marginals = {
        "region": {"NE": 1000000, "SE": 800000, "MW": 1200000, "W": 900000},
        "age_group": {"18-34": 1200000, "35-54": 1800000, "55+": 900000},
    }
    panel_w = raking(panel, marginals)
    panel_w.to_parquet(LANDING_DIR / "panel_weighted.parquet", index=False)
    print("Saved weighted panel.")


if __name__ == "__main__":
    main()
