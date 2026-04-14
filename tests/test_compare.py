from __future__ import annotations

import pandas as pd

from crashrisk.config import CrashRiskConfig
from crashrisk.models.compare import baseline_feature_columns, compare_esg_lift, is_esg_feature


COMPARISON_FEATURES = (
    "lagged_ncskew",
    "lagged_duvol",
    "detrended_turnover",
    "controversy_score",
    "controversy_change_13w",
)


def _comparison_dataset() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-03", periods=12, freq="W-FRI")
    for date_index, date in enumerate(dates):
        for ticker, is_positive in {"AAA": 0, "BBB": 0, "CCC": 1, "DDD": 1}.items():
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "lagged_ncskew": 0.1 * date_index,
                    "lagged_duvol": 0.05 * date_index,
                    "detrended_turnover": 0.01 * date_index,
                    "controversy_score": 5.0 if is_positive else 0.2,
                    "controversy_change_13w": 2.0 if is_positive else -0.1,
                    "high_crash_risk": is_positive,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_feature_columns_remove_controversy_features():
    config = CrashRiskConfig(feature_columns=COMPARISON_FEATURES)

    baseline = baseline_feature_columns(config)

    assert "lagged_ncskew" in baseline
    assert "controversy_score" not in baseline
    assert "controversy_change_13w" not in baseline
    assert is_esg_feature("controversy_spike_flag")


def test_compare_esg_lift_reports_baseline_full_and_delta_rows():
    config = CrashRiskConfig(feature_columns=COMPARISON_FEATURES, train_fraction=0.5, validation_fraction=0.25)

    report = compare_esg_lift(_comparison_dataset(), config=config)

    assert {"baseline_no_esg", "full_with_esg", "full_minus_baseline"}.issubset(set(report["model"]))
    assert {"validation", "test"}.issubset(set(report["split"]))
    assert {"roc_auc", "precision_at_top_bucket", "crash_capture_at_top_bucket"}.issubset(report.columns)
    assert report.loc[report["model"] == "full_with_esg", "feature_count"].iloc[0] == len(COMPARISON_FEATURES)
    assert report.loc[report["model"] == "baseline_no_esg", "feature_count"].iloc[0] == 3

