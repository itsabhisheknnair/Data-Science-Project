from __future__ import annotations

import pandas as pd
import pytest

from crashrisk.features.crash_metrics import compute_ncskew
from crashrisk.targets import make_targets


def test_targets_use_future_window_and_exclude_current_week():
    dates = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "firm_specific_return": [0.50, -0.20, 0.03, 0.04, 0.05, -0.01],
        }
    )

    targets = make_targets(panel, horizon_weeks=3)

    first_target = targets.loc[targets["date"] == dates[0], "future_ncskew"].item()
    expected = compute_ncskew([-0.20, 0.03, 0.04])
    assert first_target == pytest.approx(expected)
    assert first_target != pytest.approx(compute_ncskew([0.50, -0.20, 0.03]))


def test_targets_rank_top_cross_section_as_high_risk():
    dates = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "date": list(dates) + list(dates),
            "firm_specific_return": [0.01, -0.30, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )

    targets = make_targets(panel, horizon_weeks=3, top_quantile=0.20)
    first_date = targets.loc[targets["date"] == dates[0], ["ticker", "high_crash_risk"]]

    assert set(first_date["high_crash_risk"].dropna().astype(int)) == {0, 1}

