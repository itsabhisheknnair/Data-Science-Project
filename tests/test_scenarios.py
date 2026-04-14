from __future__ import annotations

import pandas as pd

from crashrisk.models.scenarios import make_price_history, make_price_scenarios


def _scenario_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-03", "2020-01-10"]),
            "adj_close": [100.0, 110.0, 100.0, 110.0],
            "realized_volatility": [0.30, 0.30, 0.30, 0.30],
        }
    )


def test_make_price_history_exports_recent_prices():
    history = make_price_history(_scenario_panel(), max_weeks=1)

    assert set(history.columns) == {"ticker", "date", "adj_close"}
    assert len(history) == 2
    assert set(history["ticker"]) == {"AAA", "BBB"}
    assert history["date"].eq(pd.Timestamp("2020-01-10")).all()


def test_make_price_scenarios_exports_one_row_per_score_and_ordered_range():
    scores = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "as_of_date": ["2020-01-10", "2020-01-10"],
            "crash_probability": [0.10, 0.80],
            "risk_bucket": ["Low", "High"],
        }
    )

    scenarios = make_price_scenarios(_scenario_panel(), scores, horizon_weeks=13)

    assert len(scenarios) == 2
    assert set(scenarios["ticker"]) == {"AAA", "BBB"}
    assert scenarios["price_p05"].le(scenarios["price_p50"]).all()
    assert scenarios["price_p50"].le(scenarios["price_p95"]).all()
    assert scenarios["price_p50"].eq(scenarios["latest_price"]).all()


def test_higher_crash_probability_widens_downside_when_volatility_matches():
    scores = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "as_of_date": ["2020-01-10", "2020-01-10"],
            "crash_probability": [0.10, 0.80],
            "risk_bucket": ["Low", "High"],
        }
    )

    scenarios = make_price_scenarios(_scenario_panel(), scores, horizon_weeks=13)
    aaa = scenarios.loc[scenarios["ticker"] == "AAA"].iloc[0]
    bbb = scenarios.loc[scenarios["ticker"] == "BBB"].iloc[0]

    assert aaa["latest_price"] == bbb["latest_price"]
    assert bbb["price_p05"] < aaa["price_p05"]
    assert bbb["price_p95"] == aaa["price_p95"]

