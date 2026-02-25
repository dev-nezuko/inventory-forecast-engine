"""Tests for linking: required columns, link logic with small DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_link_requires_customer_id_in_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.linking.link_panel_credit as link_mod

    monkeypatch.setattr(link_mod, "LANDING_DIR", tmp_path)
    # Write panel without customer_id
    pd.DataFrame({"panel_id": ["p1"], "age_group": ["35-54"]}).to_parquet(
        tmp_path / "panel.parquet", index=False
    )
    pd.DataFrame({"customer_id": ["c1"], "txn_id": [1]}).to_parquet(
        tmp_path / "credit_txn.parquet", index=False
    )
    with pytest.raises(ValueError, match="panel missing required columns"):
        link_mod.run()


def test_link_with_minimal_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.linking.link_panel_credit as link_mod

    monkeypatch.setattr(link_mod, "LANDING_DIR", tmp_path)
    panel = pd.DataFrame({"customer_id": ["c1", "c2"], "region": ["NE", "SE"]})
    credit = pd.DataFrame(
        {"customer_id": ["c1", "c2"], "txn_id": [1, 2], "region": ["NE", "SE"]}
    )
    panel.to_parquet(tmp_path / "panel.parquet", index=False)
    credit.to_parquet(tmp_path / "credit_txn.parquet", index=False)
    link_mod.run()
    out = pd.read_parquet(tmp_path / "linked_panel_credit.parquet")
    assert "cust_hash" in out.columns
    assert len(out) >= 1


def test_canonical_region_helper() -> None:
    from src.linking.link_panel_credit import _canonical_region

    df = pd.DataFrame({"region_credit": ["NE"], "region_panel": ["SE"]})
    out = _canonical_region(df)
    assert "region" in out.columns


def _write_landing(tmp_path: Path, panel: pd.DataFrame, credit: pd.DataFrame) -> None:
    panel.to_parquet(tmp_path / "panel.parquet", index=False)
    credit.to_parquet(tmp_path / "credit_txn.parquet", index=False)


def test_linked_output_has_no_raw_customer_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.linking.link_panel_credit as link_mod

    monkeypatch.setattr(link_mod, "LANDING_DIR", tmp_path)
    _write_landing(
        tmp_path,
        pd.DataFrame({"customer_id": ["C001"], "region": ["NE"]}),
        pd.DataFrame(
            {"customer_id": ["C001", "C001"], "txn_id": [1, 2], "region": ["NE"] * 2}
        ),
    )
    link_mod.run()
    out = pd.read_parquet(tmp_path / "linked_panel_credit.parquet")
    assert not [c for c in out.columns if c.startswith("customer_id")]
    assert len(out) == 2 and out["cust_hash"].nunique() == 1
    assert "C001" not in out.astype(str).to_numpy()


def test_missing_customer_ids_are_not_linked_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.linking.link_panel_credit as link_mod

    monkeypatch.setattr(link_mod, "LANDING_DIR", tmp_path)
    _write_landing(
        tmp_path,
        pd.DataFrame({"customer_id": ["C001", None], "region": ["NE", "SE"]}),
        pd.DataFrame(
            {
                "customer_id": [None, "C001", " "],
                "txn_id": [1, 2, 3],
                "region": ["SE", "NE", "W"],
            }
        ),
    )
    link_mod.run()
    out = pd.read_parquet(tmp_path / "linked_panel_credit.parquet")
    assert out["txn_id"].tolist() == [2]


def test_duplicate_panel_customers_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.linking.link_panel_credit as link_mod

    monkeypatch.setattr(link_mod, "LANDING_DIR", tmp_path)
    _write_landing(
        tmp_path,
        pd.DataFrame({"customer_id": ["C001", "C001"], "region": ["NE", "NE"]}),
        pd.DataFrame({"customer_id": ["C001"], "txn_id": [1], "region": ["NE"]}),
    )
    with pytest.raises(ValueError, match="more than one row"):
        link_mod.run()


def test_link_requires_hash_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.linking.link_panel_credit as link_mod

    monkeypatch.setattr(link_mod, "LANDING_DIR", tmp_path)
    monkeypatch.delenv("HASH_SALT")
    _write_landing(
        tmp_path,
        pd.DataFrame({"customer_id": ["C001"], "region": ["NE"]}),
        pd.DataFrame({"customer_id": ["C001"], "txn_id": [1], "region": ["NE"]}),
    )
    with pytest.raises(RuntimeError, match="HASH_SALT"):
        link_mod.run()
