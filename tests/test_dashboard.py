"""Tests for the dashboard data helpers and a headless run of streamlit_demo.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from src.config import ROOT
from src.dashboard import (
    backtest_note,
    call_forecast_api,
    load_artifacts,
    pipeline_stages,
    series_view,
)

SAMPLE = ROOT / "examples" / "sample_data"


def test_load_artifacts_reads_the_sample() -> None:
    a = load_artifacts(SAMPLE)
    assert a.problems == []
    assert a.forecast_source == "forecast_ensemble.parquet"
    assert len(a.series()) == 16
    assert a.backtest["data"].startswith("synthetic")


def test_series_view_returns_history_then_forecast() -> None:
    a = load_artifacts(SAMPLE)
    hist, fc = series_view(a, "SKU-ANCHOR", "NE", history_weeks=10)
    assert len(hist) == 10 and len(fc) == 12
    assert fc["date"].min() > hist["week"].max()


def test_empty_folder_reports_missing_stages(tmp_path: Path) -> None:
    a = load_artifacts(tmp_path)
    assert a.series().empty
    statuses = {stage: status for stage, status, _ in pipeline_stages(a)}
    assert statuses["Weekly features"] == "missing"
    assert statuses["DeepAR, PyMC"] == "stub"


def test_corrupt_artifact_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "features.parquet").write_bytes(b"not parquet")
    a = load_artifacts(tmp_path)
    assert a.features is None
    assert any("features.parquet" in p for p in a.problems)


class _Response:
    def __init__(self, status: int, payload):
        self.status_code, self._payload, self.text = status, payload, str(payload)

    def json(self):
        return self._payload


def test_call_forecast_api_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = {}

    def ok(url, json, timeout):
        sent.update(url=url, json=json)
        return _Response(200, [{"sku": "S1"}])

    monkeypatch.setattr(requests, "post", ok)
    assert call_forecast_api("http://api:8000/", "S1", "NE", 3) == (
        [{"sku": "S1"}],
        None,
    )
    assert sent == {
        "url": "http://api:8000/forecast",
        "json": {"sku": "S1", "region": "NE", "horizon_weeks": 3},
    }

    monkeypatch.setattr(
        requests, "post", lambda *a, **k: _Response(404, {"detail": "nope"})
    )
    assert call_forecast_api("http://api:8000", "S1", "NE", 3) == (
        None,
        "API answered 404: nope",
    )

    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", down)
    rows, error = call_forecast_api("http://api:8000", "S1", "NE", 3)
    assert rows is None and "not reachable" in error


@pytest.mark.parametrize("folder", ["sample", "empty"])
def test_streamlit_app_renders(
    folder: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    import src.config

    data_dir = SAMPLE if folder == "sample" else tmp_path
    monkeypatch.setattr(src.config, "DATA_DIR", data_dir)
    at = AppTest.from_file(str(ROOT / "streamlit_demo.py"), default_timeout=60).run()
    assert not at.exception
    if folder == "sample":
        assert [s.label for s in at.selectbox] == ["SKU", "Region"]
        assert any("synthetic data" in str(h.proto.body) for h in at.get("html"))
        assert pd.notna(at.metric[0].value)
    else:
        assert "No weekly features found" in at.warning[0].value


def test_busiest_series_is_the_default() -> None:
    a = load_artifacts(SAMPLE)
    sku, region = a.busiest_series()
    totals = a.features.groupby(["sku", "region"])["units_sold"].sum()
    assert totals[(sku, region)] == totals.max()


def test_backtest_note_follows_the_data_label() -> None:
    bt = {"holdout_weeks": "2025-10-06 to 2025-12-22", "rows_scored": 192}
    assert "synthetic data" in backtest_note({**bt, "data": "synthetic"})
    other = backtest_note({**bt, "data": "<b>store 12</b>"})
    assert "synthetic" not in other
    assert "&lt;b&gt;store 12&lt;/b&gt;" in other
