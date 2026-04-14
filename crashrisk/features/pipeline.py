from __future__ import annotations

import pandas as pd

from crashrisk.config import CrashRiskConfig, RawDataPaths
from crashrisk.data.loaders import load_raw_data
from crashrisk.features.controversy import add_controversy_features, align_controversies
from crashrisk.features.crash_metrics import add_lagged_crash_features
from crashrisk.features.downside import add_downside_features
from crashrisk.features.returns import add_trailing_return_volatility, compute_benchmark_returns, compute_weekly_returns
from crashrisk.features.turnover import add_turnover_features


FUNDAMENTAL_FEATURE_COLUMNS = [
    "period_end",
    "available_date",
    "market_cap",
    "shares_outstanding",
    "market_to_book",
    "leverage",
    "roa",
]


def align_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    right_columns = ["available_date", *FUNDAMENTAL_FEATURE_COLUMNS]
    right_columns = list(dict.fromkeys(right_columns))
    for ticker, group in panel.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.copy().sort_values("date")
        ticker_fundamentals = fundamentals.loc[fundamentals["ticker"] == ticker, right_columns].sort_values(
            "available_date"
        )
        if ticker_fundamentals.empty:
            for column in FUNDAMENTAL_FEATURE_COLUMNS:
                group[column] = pd.NA
        else:
            group = pd.merge_asof(
                group,
                ticker_fundamentals,
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        frames.append(group)
    if not frames:
        for column in FUNDAMENTAL_FEATURE_COLUMNS:
            panel[column] = pd.NA
        return panel
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_feature_panel(
    raw_paths: RawDataPaths | dict[str, str],
    config: CrashRiskConfig | None = None,
) -> pd.DataFrame:
    config = config or CrashRiskConfig()
    raw = load_raw_data(raw_paths, config=config)

    weekly_prices = compute_weekly_returns(raw["prices"], week_rule=config.week_rule)
    weekly_benchmark = compute_benchmark_returns(raw["benchmark_prices"], week_rule=config.week_rule)
    panel = weekly_prices.merge(
        weekly_benchmark[["date", "benchmark_return"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    panel["firm_specific_return"] = panel["weekly_return"] - panel["benchmark_return"]

    panel = align_fundamentals(panel, raw["fundamentals"])
    panel = add_turnover_features(
        panel,
        window=config.turnover_window_weeks,
        min_periods=max(3, min(config.min_crash_observations, config.turnover_window_weeks)),
    )
    panel = add_trailing_return_volatility(
        panel,
        window=config.trailing_window_weeks,
        min_periods=max(3, min(config.min_crash_observations, config.trailing_window_weeks)),
    )
    panel = add_downside_features(
        panel,
        window=config.beta_window_weeks,
        min_periods=config.min_beta_observations(),
    )
    panel = align_controversies(panel, raw["controversies"])
    panel = add_controversy_features(panel, windows=config.controversy_windows)
    panel = add_lagged_crash_features(
        panel,
        window=config.trailing_window_weeks,
        min_periods=max(3, min(config.min_crash_observations, config.trailing_window_weeks)),
    )
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)

