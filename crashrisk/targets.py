from __future__ import annotations

import pandas as pd

from crashrisk.features.crash_metrics import compute_duvol, compute_ncskew


def _future_metric(values, index: int, horizon: int, metric) -> float:
    future_values = values[index + 1 : index + 1 + horizon]
    return metric(future_values)


def make_targets(
    df: pd.DataFrame,
    horizon_weeks: int = 13,
    top_quantile: float = 0.20,
    residual_col: str = "firm_specific_return",
) -> pd.DataFrame:
    if not 0 < top_quantile < 1:
        raise ValueError("top_quantile must be between 0 and 1")

    frames: list[pd.DataFrame] = []
    for _, group in df.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.copy()
        values = group[residual_col].to_numpy(dtype=float)
        group["future_ncskew"] = [
            _future_metric(values, index, horizon_weeks, compute_ncskew) for index in range(len(group))
        ]
        group["future_duvol"] = [
            _future_metric(values, index, horizon_weeks, compute_duvol) for index in range(len(group))
        ]
        frames.append(group)

    dataset = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    def label_top_bucket(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        labels = pd.Series(pd.NA, index=series.index, dtype="Int64")
        if valid.empty:
            return labels
        threshold = valid.quantile(1.0 - top_quantile)
        labels.loc[valid.index] = (valid >= threshold).astype(int)
        return labels

    dataset["high_crash_risk"] = dataset.groupby("date", group_keys=False)["future_ncskew"].apply(label_top_bucket)
    return dataset.sort_values(["ticker", "date"]).reset_index(drop=True)

