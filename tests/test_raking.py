"""Tests for raking(): convergence with simple marginals, validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.weighting.raking import raking


def test_raking_converges_simple() -> None:
    df = pd.DataFrame(
        {"region": ["NE", "NE", "SE"], "age_group": ["18-34", "35-54", "35-54"]}
    )
    marginals = {
        "region": {"NE": 2.0, "SE": 1.0},
        "age_group": {"18-34": 1.0, "35-54": 2.0},
    }
    out = raking(df, marginals, max_iter=50, tol=1e-6)
    assert "weight" in out.columns
    assert len(out) == 3
    # Weights should sum by region to match marginals (approximately)
    by_region = out.groupby("region")["weight"].sum()
    assert abs(by_region["NE"] - 2.0) < 0.01
    assert abs(by_region["SE"] - 1.0) < 0.01


def test_raking_empty_df_raises() -> None:
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="non-empty"):
        raking(df, {"region": {"NE": 1}})


def test_raking_marginal_key_not_column_raises() -> None:
    df = pd.DataFrame({"region": ["NE"]})
    with pytest.raises(ValueError, match="not a column"):
        raking(df, {"region": {"NE": 1}, "nonexistent": {"x": 1}})
