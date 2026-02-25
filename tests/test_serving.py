"""Tests for serving API: /health, /forecast with TestClient and mocked parquet."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.app import _normalize_forecast_columns, app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_forecast_404_when_no_file(client: TestClient) -> None:
    with patch("src.serving.app.FEATURE_DIR", Path("/nonexistent_empty_dir")):
        r = client.post(
            "/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 4}
        )
    assert r.status_code == 404
    assert "No forecast" in r.json()["detail"]


def test_forecast_200_with_mock_parquet(client: TestClient, tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sku": ["S1", "S1"],
            "region": ["NE", "NE"],
            "date": ["2025-02-01", "2025-02-08"],
            "q0.1": [1.0, 2.0],
            "q0.5": [2.0, 3.0],
            "q0.9": [3.0, 4.0],
        }
    )
    df.to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post(
            "/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 2}
        )
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert data[0]["sku"] == "S1" and data[0]["region"] == "NE"
    assert "q0.1" in data[0] and "q0.5" in data[0] and "q0.9" in data[0]


def test_forecast_404_sku_region_not_in_file(
    client: TestClient, tmp_path: Path
) -> None:
    df = pd.DataFrame(
        {
            "sku": ["S2"],
            "region": ["SE"],
            "date": ["2025-02-01"],
            "q0.1": [1.0],
            "q0.5": [2.0],
            "q0.9": [3.0],
        }
    )
    df.to_parquet(tmp_path / "forecast_ensemble.parquet", index=False)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post(
            "/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 4}
        )
    assert r.status_code == 404
    assert "No forecast found" in r.json()["detail"]


def test_normalize_forecast_columns_gbm_suffix() -> None:
    df = pd.DataFrame({"q0.1_gbm": [1], "q0.5_gbm": [2], "q0.9_gbm": [3]})
    out = _normalize_forecast_columns(df)
    assert "q0.1" in out.columns and out["q0.1"].iloc[0] == 1


def _write_forecast(path: Path, n: int = 3) -> None:
    pd.DataFrame(
        {
            "sku": ["S1"] * n,
            "region": ["NE"] * n,
            "date": pd.date_range("2025-02-03", periods=n, freq="W-MON")[::-1],
            "q0.1": [1.0] * n,
            "q0.5": [2.0] * n,
            "q0.9": [3.0] * n,
        }
    ).to_parquet(path / "forecast_ensemble.parquet", index=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"sku": "../../etc/passwd", "region": "NE"},
        {"sku": "S1", "region": "NE/../../x"},
        {"sku": "S1", "region": ""},
        {"sku": "S" * 65, "region": "NE"},
        {"sku": "S1", "region": "NE", "horizon_weeks": 0},
        {"sku": "S1", "region": "NE", "horizon_weeks": -1},
        {"sku": "S1", "region": "NE", "horizon_weeks": 53},
        {"sku": "S1", "region": "NE", "extra": 1},
    ],
)
def test_forecast_rejects_bad_input(
    client: TestClient, tmp_path: Path, payload: dict
) -> None:
    _write_forecast(tmp_path)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json=payload)
    assert r.status_code == 422


def test_forecast_returns_earliest_weeks_in_order(
    client: TestClient, tmp_path: Path
) -> None:
    _write_forecast(tmp_path, n=3)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post(
            "/forecast", json={"sku": "S1", "region": "NE", "horizon_weeks": 2}
        )
    assert r.status_code == 200
    assert [row["date"] for row in r.json()] == ["2025-02-03", "2025-02-10"]


def test_forecast_reloads_artifact_after_it_changes(
    client: TestClient, tmp_path: Path
) -> None:
    import os

    _write_forecast(tmp_path, n=1)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        first = client.post("/forecast", json={"sku": "S1", "region": "NE"}).json()
        _write_forecast(tmp_path, n=4)
        stat = (tmp_path / "forecast_ensemble.parquet").stat()
        os.utime(
            tmp_path / "forecast_ensemble.parquet",
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9),
        )
        second = client.post("/forecast", json={"sku": "S1", "region": "NE"}).json()
    assert len(first) == 1 and len(second) == 4


def test_gbm_artifact_columns_are_normalised(
    client: TestClient, tmp_path: Path
) -> None:
    pd.DataFrame(
        {
            "sku": ["S1"],
            "region": ["NE"],
            "date": ["2025-02-03"],
            "q0.1_gbm": [1.0],
            "q0.5_gbm": [2.0],
            "q0.9_gbm": [3.0],
        }
    ).to_parquet(tmp_path / "forecast_gbm.parquet", index=False)
    with patch("src.serving.app.FEATURE_DIR", tmp_path):
        r = client.post("/forecast", json={"sku": "S1", "region": "NE"})
    assert r.status_code == 200
    assert r.json() == [
        {
            "sku": "S1",
            "region": "NE",
            "date": "2025-02-03",
            "q0.1": 1.0,
            "q0.5": 2.0,
            "q0.9": 3.0,
        }
    ]
