"""Tests for src.utils.metrics: wape and quantile_loss with known values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.metrics import quantile_loss, wape


def test_wape_known() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    # sum|y_true - y_pred| = 2+2+0 = 4, sum|y_true| = 60
    assert abs(wape(y_true, y_pred) - 4 / 60) < 1e-9


def test_wape_accepts_series() -> None:
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([10.0, 20.0])
    assert wape(y_true, y_pred) == 0.0


def test_wape_zero_actual_guarded() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    # denom uses max(sum|y_true|, 1e-9) so we get a finite value
    w = wape(y_true, y_pred)
    assert np.isfinite(w)


def test_quantile_loss_known() -> None:
    # e = y_true - y_pred. For q=0.5, loss = mean(|e|)/2 * something; pinball: max(0.5*e, -0.5*e) = 0.5|e|
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert quantile_loss(y_true, y_pred, 0.5) == 0.0
    e = np.array([1.0, -1.0, 0.0])
    q = 0.5
    expected = np.mean(np.maximum(q * e, (q - 1) * e))
    assert abs(quantile_loss(y_true, y_pred + (-e), 0.5) - expected) < 1e-9


def test_quantile_loss_accepts_series() -> None:
    y_true = pd.Series([1.0, 2.0])
    y_pred = pd.Series([1.0, 2.0])
    assert quantile_loss(y_true, y_pred, 0.1) == 0.0
