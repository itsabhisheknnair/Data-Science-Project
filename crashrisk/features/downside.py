from __future__ import annotations

import numpy as np
import pandas as pd


def _beta(stock: np.ndarray, market: np.ndarray) -> float:
    valid = np.isfinite(stock) & np.isfinite(market)
    stock = stock[valid]
    market = market[valid]
    if len(stock) < 3:
        return np.nan
    market_var = np.var(market, ddof=1)
    if market_var <= 0:
        return np.nan
    return float(np.cov(stock, market, ddof=1)[0, 1] / market_var)


def add_downside_features(
    panel: pd.DataFrame,
    window: int,
    min_periods: int,
    return_col: str = "weekly_return",
    benchmark_col: str = "benchmark_return",
) -> pd.DataFrame:
    output_frames: list[pd.DataFrame] = []
    for _, group in panel.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.copy()
        stock = group[return_col].to_numpy(dtype=float)
        market = group[benchmark_col].to_numpy(dtype=float)
        betas: list[float] = []
        downside_betas: list[float] = []
        for idx in range(len(group)):
            start = max(0, idx - window + 1)
            stock_window = stock[start : idx + 1]
            market_window = market[start : idx + 1]
            valid = np.isfinite(stock_window) & np.isfinite(market_window)
            if valid.sum() < min_periods:
                betas.append(np.nan)
                downside_betas.append(np.nan)
                continue
            betas.append(_beta(stock_window, market_window))
            downside_mask = valid & (market_window < 0)
            if downside_mask.sum() < 3:
                downside_betas.append(np.nan)
            else:
                downside_betas.append(_beta(stock_window[downside_mask], market_window[downside_mask]))
        group["beta"] = betas
        group["downside_beta"] = downside_betas
        group["relative_downside_beta"] = group["downside_beta"] - group["beta"]
        output_frames.append(group)
    if not output_frames:
        return panel.assign(beta=np.nan, downside_beta=np.nan, relative_downside_beta=np.nan)
    return pd.concat(output_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

