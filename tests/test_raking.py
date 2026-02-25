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


def test_raking_matches_both_marginals() -> None:
    df = pd.DataFrame(
        {
            "region": ["NE", "NE", "SE", "SE", "SE"],
            "age_group": ["18-34", "55+", "18-34", "55+", "55+"],
        }
    )
    marginals = {
        "region": {"NE": 60.0, "SE": 40.0},
        "age_group": {"18-34": 30.0, "55+": 70.0},
    }
    out = raking(df, marginals, max_iter=200, tol=1e-9)
    assert out.groupby("region")["weight"].sum().round(6).to_dict() == {
        "NE": 60.0,
        "SE": 40.0,
    }
    assert out.groupby("age_group")["weight"].sum().round(6).to_dict() == {
        "18-34": 30.0,
        "55+": 70.0,
    }


def test_raking_category_without_target_raises() -> None:
    df = pd.DataFrame({"region": ["NE", "MW"]})
    with pytest.raises(ValueError, match=r"no 'region' target for categories \['MW'\]"):
        raking(df, {"region": {"NE": 1.0}})


def test_raking_inconsistent_totals_raise() -> None:
    df = pd.DataFrame({"region": ["NE"], "age_group": ["55+"]})
    with pytest.raises(ValueError, match="totals differ"):
        raking(df, {"region": {"NE": 10.0}, "age_group": {"55+": 7.0}})


def test_main_writes_weights_for_example_marginals(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.weighting.raking as mod

    monkeypatch.setattr(mod, "LANDING_DIR", tmp_path)
    grid = [(r, a) for r in ("NE", "SE", "MW", "W") for a in ("18-34", "35-54", "55+")]
    pd.DataFrame(grid, columns=["region", "age_group"]).to_parquet(
        tmp_path / "panel.parquet", index=False
    )
    mod.main()
    out = pd.read_parquet(tmp_path / "panel_weighted.parquet")
    assert abs(out.groupby("region")["weight"].sum()["MW"] - 1_200_000) < 1
    assert abs(out["weight"].sum() - 3_900_000) < 1
