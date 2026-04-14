from __future__ import annotations

import numpy as np
import pandas as pd


def align_controversies(panel: pd.DataFrame, controversies: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    right_columns = ["date", "sector", "controversy_score"]
    for ticker, group in panel.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.copy().sort_values("date")
        ticker_controversies = controversies.loc[controversies["ticker"] == ticker, right_columns].sort_values("date")
        if ticker_controversies.empty:
            group["sector"] = pd.NA
            group["controversy_score"] = np.nan
        else:
            group = pd.merge_asof(
                group,
                ticker_controversies,
                on="date",
                direction="backward",
                allow_exact_matches=True,
            )
        frames.append(group)
    if not frames:
        return panel.assign(sector=pd.NA, controversy_score=np.nan)
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def add_controversy_features(panel: pd.DataFrame, windows: tuple[int, ...] = (4, 13, 26)) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    grouped = panel.groupby("ticker", sort=False)["controversy_score"]
    for window in windows:
        panel[f"controversy_change_{window}w"] = grouped.transform(lambda series: series.diff(window))
        panel[f"controversy_rolling_mean_{window}w"] = grouped.transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        panel[f"controversy_rolling_std_{window}w"] = grouped.transform(
            lambda series: series.rolling(window, min_periods=2).std()
        )

    long_mean_col = f"controversy_rolling_mean_{max(windows)}w"
    long_std_col = f"controversy_rolling_std_{max(windows)}w"
    panel["controversy_spike_flag"] = (
        panel["controversy_score"] > panel[long_mean_col] + 2.0 * panel[long_std_col].fillna(0.0)
    ).astype(int)
    panel["sector"] = panel["sector"].fillna("Unknown")
    panel["controversy_sector_percentile"] = panel.groupby(["date", "sector"], sort=False)[
        "controversy_score"
    ].rank(pct=True)
    return panel

