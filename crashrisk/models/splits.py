from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ChronologicalSplits:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    sorted_df = df.sort_values(date_col).copy()
    unique_dates = pd.Series(sorted_df[date_col].dropna().sort_values().unique())
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique dates for chronological train/validation/test split")

    train_end = max(1, int(len(unique_dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(unique_dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(unique_dates) - 1)

    train_dates = set(unique_dates.iloc[:train_end])
    validation_dates = set(unique_dates.iloc[train_end:validation_end])
    test_dates = set(unique_dates.iloc[validation_end:])

    return ChronologicalSplits(
        train=sorted_df.loc[sorted_df[date_col].isin(train_dates)].copy(),
        validation=sorted_df.loc[sorted_df[date_col].isin(validation_dates)].copy(),
        test=sorted_df.loc[sorted_df[date_col].isin(test_dates)].copy(),
    )

