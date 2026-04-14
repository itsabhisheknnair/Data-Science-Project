from __future__ import annotations

import numpy as np
import pandas as pd

from crashrisk.config import CrashRiskConfig
from crashrisk.models.score import assign_risk_buckets, score_latest
from crashrisk.models.splits import chronological_split
from crashrisk.models.train import train_classifier


MODEL_FEATURES = ("lagged_ncskew", "lagged_duvol", "detrended_turnover")


def _model_dataset() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-03", periods=12, freq="W-FRI")
    for i, date in enumerate(dates):
        for ticker, offset in {"AAA": 0.0, "BBB": 0.4, "CCC": 0.8}.items():
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "lagged_ncskew": i * 0.05 + offset,
                    "lagged_duvol": i * 0.03 - offset,
                    "detrended_turnover": offset - i * 0.01,
                    "high_crash_risk": int(ticker == "CCC"),
                }
            )
    return pd.DataFrame(rows)


def test_chronological_split_has_ordered_non_overlapping_blocks():
    dataset = _model_dataset()
    splits = chronological_split(dataset, train_fraction=0.5, validation_fraction=0.25)

    assert splits.train["date"].max() < splits.validation["date"].min()
    assert splits.validation["date"].max() < splits.test["date"].min()


def test_train_classifier_and_score_latest_emit_probability_schema():
    dataset = _model_dataset()
    config = CrashRiskConfig(feature_columns=MODEL_FEATURES, train_fraction=0.5, validation_fraction=0.25)

    model = train_classifier(dataset, config=config)
    scores = score_latest(model, dataset, as_of_date=dataset["date"].max())

    assert set(scores.columns) == {"ticker", "as_of_date", "crash_probability", "risk_bucket", "top_drivers"}
    assert scores["crash_probability"].between(0, 1).all()
    assert set(scores["risk_bucket"]).issubset({"Low", "Medium", "High"})
    assert scores["top_drivers"].str.len().gt(0).all()


def test_assign_risk_buckets_uses_top_probability_as_high():
    probabilities = pd.Series([0.1, 0.8, 0.4, 0.2, 0.7])
    buckets = assign_risk_buckets(probabilities)

    assert buckets.loc[1] == "High"
    assert (buckets == "High").sum() == 1
    assert (buckets == "Medium").sum() == 2
    assert (buckets == "Low").sum() == 2


def test_train_classifier_rejects_single_class_target():
    dataset = _model_dataset()
    dataset["high_crash_risk"] = 0
    config = CrashRiskConfig(feature_columns=MODEL_FEATURES)

    try:
        train_classifier(dataset, config=config)
    except ValueError as exc:
        assert "two target classes" in str(exc)
    else:
        raise AssertionError("Expected train_classifier to reject single-class target")

