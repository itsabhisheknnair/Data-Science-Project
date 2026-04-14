from __future__ import annotations

import numpy as np
import pandas as pd


def add_turnover_features(panel: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    shares = panel["shares_outstanding"].replace(0, np.nan)
    panel["turnover"] = panel["weekly_volume"] / shares
    grouped = panel.groupby("ticker", sort=False)["turnover"]
    trend = grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).mean())
    std = grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).std())
    panel["detrended_turnover"] = panel["turnover"] - trend
    panel["turnover_zscore"] = panel["detrended_turnover"] / std.replace(0, np.nan)
    return panel

