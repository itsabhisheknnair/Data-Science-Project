from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def assign_risk_buckets(probabilities: pd.Series, high_share: float = 0.20, medium_share: float = 0.40) -> pd.Series:
    if probabilities.empty:
        return pd.Series(dtype="object")
    n_obs = len(probabilities)
    high_count = max(1, ceil(n_obs * high_share))
    medium_count = max(0, ceil(n_obs * medium_share))
    rank = probabilities.rank(method="first", ascending=False)
    labels = pd.Series("Low", index=probabilities.index, dtype="object")
    labels.loc[rank <= high_count] = "High"
    labels.loc[(rank > high_count) & (rank <= high_count + medium_count)] = "Medium"
    return labels


def _top_driver_strings(model: Pipeline, features: pd.DataFrame, top_n: int) -> list[str]:
    feature_columns = list(model.feature_columns_)
    transformed = model[:-1].transform(features[feature_columns])
    coefficients = model.named_steps["classifier"].coef_[0]
    contributions = transformed * coefficients
    driver_strings: list[str] = []
    for row in contributions:
        top_indices = np.argsort(np.abs(row))[::-1][:top_n]
        driver_strings.append(";".join(feature_columns[index] for index in top_indices))
    return driver_strings


def score_latest(
    model: Pipeline,
    panel: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
    top_n_drivers: int = 3,
) -> pd.DataFrame:
    feature_columns = list(model.feature_columns_)
    missing = [column for column in feature_columns if column not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing feature column(s): {', '.join(missing)}")

    dates = pd.to_datetime(panel["date"])
    if as_of_date is None:
        scoring_date = dates.max()
    else:
        requested = pd.Timestamp(as_of_date)
        eligible = dates.loc[dates <= requested]
        if eligible.empty:
            raise ValueError(f"No panel rows exist on or before as_of_date={requested.date()}")
        scoring_date = eligible.max()

    latest = panel.loc[dates == scoring_date].copy().sort_values("ticker").reset_index(drop=True)
    if latest.empty:
        raise ValueError("No rows available for scoring")

    positive_index = list(model.named_steps["classifier"].classes_).index(1)
    probabilities = pd.Series(
        model.predict_proba(latest[feature_columns])[:, positive_index],
        index=latest.index,
        name="crash_probability",
    )
    latest["crash_probability"] = probabilities
    latest["risk_bucket"] = assign_risk_buckets(probabilities)
    latest["top_drivers"] = _top_driver_strings(model, latest, top_n=top_n_drivers)
    latest["as_of_date"] = pd.Timestamp(scoring_date).date().isoformat()
    return latest[["ticker", "as_of_date", "crash_probability", "risk_bucket", "top_drivers"]]

