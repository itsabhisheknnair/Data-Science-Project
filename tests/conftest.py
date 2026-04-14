from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crashrisk.config import CrashRiskConfig, RawDataPaths


@pytest.fixture
def workspace_tmp_path(request) -> "Path":
    from pathlib import Path
    from uuid import uuid4

    safe_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_").replace("\\", "_")
    path = Path("data/processed/test-artifacts") / f"{safe_name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def small_config() -> CrashRiskConfig:
    return CrashRiskConfig(
        trailing_window_weeks=6,
        turnover_window_weeks=6,
        beta_window_weeks=6,
        min_crash_observations=3,
        target_horizon_weeks=6,
        train_fraction=0.60,
        validation_fraction=0.20,
    )


@pytest.fixture
def synthetic_raw_paths(workspace_tmp_path) -> RawDataPaths:
    raw_dir = workspace_tmp_path / "raw"
    raw_dir.mkdir()
    dates = pd.date_range("2020-01-03", periods=42, freq="W-FRI")
    benchmark_returns = np.array([0.0] + [0.012 * np.sin(i / 1.7) for i in range(1, len(dates))])
    benchmark_close = 100.0 * np.cumprod(1.0 + benchmark_returns)
    pd.DataFrame({"date": dates, "benchmark_close": benchmark_close}).to_csv(
        raw_dir / "benchmark_prices.csv", index=False
    )

    price_rows = []
    ticker_specs = {
        "AAA": (50.0, 0.9, lambda i: 0.014 * np.sin(i * 0.7)),
        "BBB": (75.0, 1.1, lambda i: -0.011 * np.cos(i * 0.5)),
        "CCC": (90.0, 0.8, lambda i: -0.065 if i % 9 == 0 else 0.010 * np.cos(i * 0.4)),
    }
    for ticker, (start_price, beta, residual_fn) in ticker_specs.items():
        returns = np.array([beta * benchmark_returns[i] + residual_fn(i) for i in range(len(dates))])
        returns[0] = 0.0
        prices = start_price * np.cumprod(1.0 + returns)
        for i, date in enumerate(dates):
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adj_close": prices[i],
                    "volume": 1_000_000 + i * 10_000 + len(ticker) * 100,
                }
            )
    pd.DataFrame(price_rows).to_csv(raw_dir / "prices.csv", index=False)

    fundamentals = []
    for ticker, market_cap in {"AAA": 10_000_000_000, "BBB": 15_000_000_000, "CCC": 8_000_000_000}.items():
        fundamentals.append(
            {
                "ticker": ticker,
                "period_end": "2019-09-30",
                "market_cap": market_cap,
                "shares_outstanding": 100_000_000,
                "market_to_book": 2.0,
                "leverage": 0.35,
                "roa": 0.08,
            }
        )
        fundamentals.append(
            {
                "ticker": ticker,
                "period_end": "2020-06-30",
                "market_cap": market_cap * 1.05,
                "shares_outstanding": 100_000_000,
                "market_to_book": 2.2,
                "leverage": 0.33,
                "roa": 0.09,
            }
        )
    pd.DataFrame(fundamentals).to_csv(raw_dir / "fundamentals.csv", index=False)

    controversy_rows = []
    for ticker, sector in {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}.items():
        for i, date in enumerate(dates[::4]):
            controversy_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "sector": sector,
                    "controversy_score": i + (4 if ticker == "CCC" else 1),
                }
            )
    pd.DataFrame(controversy_rows).to_csv(raw_dir / "controversies.csv", index=False)

    return RawDataPaths(
        prices=raw_dir / "prices.csv",
        benchmark_prices=raw_dir / "benchmark_prices.csv",
        fundamentals=raw_dir / "fundamentals.csv",
        controversies=raw_dir / "controversies.csv",
    )
