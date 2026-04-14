from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_returns(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def compute_ncskew(values: pd.Series | np.ndarray | list[float]) -> float:
    returns = _clean_returns(values)
    n_obs = len(returns)
    if n_obs < 3:
        return np.nan
    sum_sq = np.sum(returns**2)
    if sum_sq <= 0:
        return np.nan
    numerator = n_obs * (n_obs - 1) ** 1.5 * np.sum(returns**3)
    denominator = (n_obs - 1) * (n_obs - 2) * (sum_sq**1.5)
    return float(-numerator / denominator)


def compute_duvol(values: pd.Series | np.ndarray | list[float]) -> float:
    returns = _clean_returns(values)
    if len(returns) < 4:
        return np.nan
    mean_return = np.mean(returns)
    down = returns[returns < mean_return]
    up = returns[returns >= mean_return]
    n_down = len(down)
    n_up = len(up)
    if n_down <= 1 or n_up <= 1:
        return np.nan
    down_sum = np.sum(down**2)
    up_sum = np.sum(up**2)
    if down_sum <= 0 or up_sum <= 0:
        return np.nan
    return float(np.log(((n_up - 1) * down_sum) / ((n_down - 1) * up_sum)))


def add_lagged_crash_features(
    panel: pd.DataFrame,
    window: int,
    min_periods: int,
    residual_col: str = "firm_specific_return",
) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    grouped = panel.groupby("ticker", sort=False)[residual_col]
    panel["lagged_ncskew"] = grouped.transform(
        lambda series: series.rolling(window, min_periods=min_periods).apply(compute_ncskew, raw=False)
    )
    panel["lagged_duvol"] = grouped.transform(
        lambda series: series.rolling(window, min_periods=max(4, min_periods)).apply(compute_duvol, raw=False)
    )
    return panel

