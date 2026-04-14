from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_HISTORY_COLUMNS = ["ticker", "date", "adj_close"]
PRICE_SCENARIO_COLUMNS = [
    "ticker",
    "as_of_date",
    "latest_price",
    "horizon_weeks",
    "price_p05",
    "price_p50",
    "price_p95",
    "crash_probability",
    "risk_bucket",
    "scenario_method",
]


def make_price_history(panel: pd.DataFrame, max_weeks: int = 104) -> pd.DataFrame:
    required = {"ticker", "date", "adj_close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required price history column(s): {', '.join(sorted(missing))}")

    history = panel.sort_values(["ticker", "date"]).copy()
    history = history.groupby("ticker", group_keys=False).tail(max_weeks)
    return history[PRICE_HISTORY_COLUMNS].reset_index(drop=True)


def make_price_scenarios(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    horizon_weeks: int = 13,
    z_score: float = 1.645,
) -> pd.DataFrame:
    required_panel = {"ticker", "date", "adj_close", "realized_volatility"}
    missing_panel = required_panel - set(panel.columns)
    if missing_panel:
        raise ValueError(f"panel is missing required scenario column(s): {', '.join(sorted(missing_panel))}")

    required_scores = {"ticker", "as_of_date", "crash_probability", "risk_bucket"}
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(f"scores are missing required scenario column(s): {', '.join(sorted(missing_scores))}")

    latest = panel.sort_values(["ticker", "date"]).groupby("ticker", as_index=False, group_keys=False).tail(1)
    scenario_base = scores.merge(
        latest[["ticker", "adj_close", "realized_volatility"]],
        on="ticker",
        how="left",
        validate="one_to_one",
    )
    scenario_base["latest_price"] = pd.to_numeric(scenario_base["adj_close"], errors="coerce")
    scenario_base["crash_probability"] = pd.to_numeric(scenario_base["crash_probability"], errors="coerce").fillna(0.0)
    annual_vol = pd.to_numeric(scenario_base["realized_volatility"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weekly_vol = annual_vol / np.sqrt(52.0)
    horizon_vol = weekly_vol * np.sqrt(float(horizon_weeks))

    scenario_base["horizon_weeks"] = int(horizon_weeks)
    scenario_base["price_p05"] = scenario_base["latest_price"] * np.exp(
        -z_score * horizon_vol * (1.0 + scenario_base["crash_probability"].clip(lower=0.0, upper=1.0))
    )
    scenario_base["price_p50"] = scenario_base["latest_price"]
    scenario_base["price_p95"] = scenario_base["latest_price"] * np.exp(z_score * horizon_vol)
    scenario_base["scenario_method"] = "historical_volatility_crash_adjusted"

    return scenario_base[PRICE_SCENARIO_COLUMNS].sort_values("ticker").reset_index(drop=True)

