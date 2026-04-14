from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crashrisk.features.controversy import add_controversy_features, align_controversies
from crashrisk.features.crash_metrics import compute_duvol, compute_ncskew
from crashrisk.features.downside import add_downside_features
from crashrisk.features.pipeline import align_fundamentals
from crashrisk.features.returns import compute_weekly_returns


def test_weekly_returns_and_volume_are_resampled_to_friday():
    prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-10"]),
            "adj_close": [90.0, 100.0, 110.0],
            "volume": [10, 20, 30],
        }
    )

    weekly = compute_weekly_returns(prices)

    assert list(weekly["date"]) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-10")]
    assert weekly.loc[0, "weekly_volume"] == 30
    assert weekly.loc[1, "weekly_return"] == pytest.approx(0.10)


def test_ncskew_and_duvol_match_formula():
    values = np.array([-0.20, -0.08, 0.03, 0.04, 0.05])
    n_obs = len(values)
    expected_ncskew = -(
        n_obs * (n_obs - 1) ** 1.5 * np.sum(values**3)
    ) / ((n_obs - 1) * (n_obs - 2) * (np.sum(values**2) ** 1.5))
    mean_return = np.mean(values)
    down = values[values < mean_return]
    up = values[values >= mean_return]
    expected_duvol = np.log(((len(up) - 1) * np.sum(down**2)) / ((len(down) - 1) * np.sum(up**2)))

    assert compute_ncskew(values) == pytest.approx(expected_ncskew)
    assert compute_duvol(values) == pytest.approx(expected_duvol)


def test_downside_beta_uses_negative_market_window():
    market = np.array([0.01, -0.02, -0.03, 0.02, -0.01, 0.03])
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(market),
            "date": pd.date_range("2020-01-03", periods=len(market), freq="W-FRI"),
            "weekly_return": 2.0 * market,
            "benchmark_return": market,
        }
    )

    features = add_downside_features(panel, window=6, min_periods=3)

    assert features.loc[len(features) - 1, "beta"] == pytest.approx(2.0)
    assert features.loc[len(features) - 1, "downside_beta"] == pytest.approx(2.0)
    assert features.loc[len(features) - 1, "relative_downside_beta"] == pytest.approx(0.0)


def test_controversy_alignment_does_not_use_future_scores():
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "date": pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"]),
        }
    )
    controversies = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-20"]),
            "sector": ["Tech", "Tech"],
            "controversy_score": [5.0, 99.0],
        }
    )

    aligned = add_controversy_features(align_controversies(panel, controversies))

    assert aligned.loc[aligned["date"] == pd.Timestamp("2020-01-17"), "controversy_score"].item() == 5.0
    assert aligned.loc[aligned["date"] == pd.Timestamp("2020-01-24"), "controversy_score"].item() == 99.0


def test_fundamental_alignment_waits_until_available_date():
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2021-02-01", "2021-02-19"]),
        }
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "period_end": [pd.Timestamp("2020-12-31")],
            "available_date": [pd.Timestamp("2021-02-14")],
            "market_cap": [1_000.0],
            "shares_outstanding": [100.0],
            "market_to_book": [2.0],
            "leverage": [0.3],
            "roa": [0.1],
        }
    )

    aligned = align_fundamentals(panel, fundamentals)

    assert pd.isna(aligned.loc[aligned["date"] == pd.Timestamp("2021-02-01"), "market_cap"]).item()
    assert aligned.loc[aligned["date"] == pd.Timestamp("2021-02-19"), "market_cap"].item() == 1_000.0

