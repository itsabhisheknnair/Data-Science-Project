from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weekly_returns(prices: pd.DataFrame, week_rule: str = "W-FRI") -> pd.DataFrame:
    weekly_frames: list[pd.DataFrame] = []
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        weekly = (
            group.set_index("date")
            .resample(week_rule)
            .agg(adj_close=("adj_close", "last"), weekly_volume=("volume", "sum"))
            .dropna(subset=["adj_close"])
        )
        weekly["ticker"] = ticker
        weekly["weekly_return"] = weekly["adj_close"].pct_change()
        weekly_frames.append(weekly.reset_index())
    if not weekly_frames:
        return pd.DataFrame(columns=["ticker", "date", "adj_close", "weekly_volume", "weekly_return"])
    return pd.concat(weekly_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_benchmark_returns(benchmark_prices: pd.DataFrame, week_rule: str = "W-FRI") -> pd.DataFrame:
    weekly = (
        benchmark_prices.sort_values("date")
        .set_index("date")
        .resample(week_rule)
        .agg(benchmark_close=("benchmark_close", "last"))
        .dropna(subset=["benchmark_close"])
    )
    weekly["benchmark_return"] = weekly["benchmark_close"].pct_change()
    return weekly.reset_index().sort_values("date").reset_index(drop=True)


def add_trailing_return_volatility(
    panel: pd.DataFrame,
    window: int,
    min_periods: int,
    return_col: str = "weekly_return",
) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()

    def cumulative_return(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < min_periods:
            return np.nan
        return float(np.prod(1.0 + finite) - 1.0)

    grouped = panel.groupby("ticker", sort=False)[return_col]
    panel["trailing_return"] = grouped.transform(
        lambda series: series.rolling(window, min_periods=min_periods).apply(cumulative_return, raw=True)
    )
    panel["realized_volatility"] = grouped.transform(
        lambda series: series.rolling(window, min_periods=min_periods).std()
    ) * np.sqrt(52.0)
    return panel

