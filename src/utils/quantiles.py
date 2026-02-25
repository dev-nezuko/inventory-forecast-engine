"""Helpers for keeping quantile forecasts coherent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def enforce_quantile_order(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Sort each row's quantile values so q0.1 <= q0.5 <= q0.9.

    Separately fitted or averaged quantiles can cross; sorting the values per
    row (monotone rearrangement) never increases pinball loss.
    """
    out = df.copy()
    out[cols] = np.sort(out[cols].to_numpy(dtype=float), axis=1)
    return out
