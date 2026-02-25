"""Tests for serving API: /health, /forecast with TestClient and mocked parquet."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app, _normalize_forecast_columns


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


def test_normalize_forecast_columns_lgb_suffix() -> None:
    df = pd.DataFrame({"q0.1_lgb": [1], "q0.5_lgb": [2], "q0.9_lgb": [3]})
    out = _normalize_forecast_columns(df)
    assert "q0.1" in out.columns and out["q0.1"].iloc[0] == 1
