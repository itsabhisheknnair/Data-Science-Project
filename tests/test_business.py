from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from crashrisk.analysis.business import build_weekly_forward_portfolio_returns


class DummyRiskModel:
    feature_columns_ = ["risk_signal"]
    named_steps = {"classifier": SimpleNamespace(classes_=np.array([0, 1]))}

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = features["risk_signal"].astype(float).to_numpy()
        return np.column_stack([1.0 - probabilities, probabilities])


def test_weekly_forward_overlay_excludes_high_risk_name_for_next_week_return():
    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
    next_returns = {"A": -0.10, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05}
    risk_signals = {"A": 0.90, "B": 0.10, "C": 0.20, "D": 0.30, "E": 0.40}

    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "weekly_return": next_returns[ticker] if date == dates[2] else 0.0,
                    "risk_signal": risk_signals[ticker] if date == dates[1] else 0.10,
                }
            )
    panel = pd.DataFrame(rows)

    portfolio = build_weekly_forward_portfolio_returns(
        panel,
        DummyRiskModel(),
        eval_start_quantile=0.0,
        high_share=0.20,
    )

    assert len(portfolio) == 1
    row = portfolio.iloc[0]
    assert row["date"] == dates[1]
    assert row["return_date"] == dates[2]
    assert row["excluded_tickers"] == "A"
    assert row["n_excluded"] == 1
    assert row["n_holdings"] == 4
    assert np.isclose(row["benchmark"], np.mean(list(next_returns.values())))
    assert np.isclose(row["strategy"], np.mean([next_returns[ticker] for ticker in ["B", "C", "D", "E"]]))
