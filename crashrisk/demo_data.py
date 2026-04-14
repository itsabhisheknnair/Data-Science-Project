from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# 12 tickers across 6 sectors with varied risk/controversy profiles.
# Keys: sector, start price, market beta, controversy_base score (0–10 scale).
DEMO_TICKERS: dict[str, dict] = {
    # Technology — high beta, moderate controversy
    "ALPH": {"sector": "Technology",  "start": 84.0,  "beta": 1.10, "controversy_base": 2.0},
    "GLOB": {"sector": "Technology",  "start": 91.0,  "beta": 1.18, "controversy_base": 1.5},
    # Energy — high beta, elevated controversy (environmental concerns)
    "CYRX": {"sector": "Energy",      "start": 46.0,  "beta": 1.35, "controversy_base": 4.2},
    "HVST": {"sector": "Energy",      "start": 38.0,  "beta": 1.42, "controversy_base": 3.5},
    # Financials — moderate beta, occasional governance controversies
    "FINX": {"sector": "Financials",  "start": 52.0,  "beta": 1.22, "controversy_base": 2.2},
    "LUXE": {"sector": "Financials",  "start": 78.0,  "beta": 1.15, "controversy_base": 2.0},
    # Healthcare — low beta, generally low controversy
    "DYNM": {"sector": "Healthcare",  "start": 72.0,  "beta": 0.78, "controversy_base": 1.8},
    "IMUN": {"sector": "Healthcare",  "start": 67.0,  "beta": 0.72, "controversy_base": 1.2},
    # Industrials — moderate beta and controversy
    "BRCK": {"sector": "Industrials", "start": 58.0,  "beta": 0.86, "controversy_base": 1.4},
    "KNXT": {"sector": "Industrials", "start": 55.0,  "beta": 0.88, "controversy_base": 1.6},
    # Consumer — moderate profile
    "ECOR": {"sector": "Consumer",    "start": 63.0,  "beta": 1.05, "controversy_base": 2.5},
    "JETT": {"sector": "Consumer",    "start": 49.0,  "beta": 0.95, "controversy_base": 2.8},
}

# Tickers that experience idiosyncratic crash events (significant negative returns)
_CRASH_TICKERS = {"CYRX", "FINX", "HVST"}
# Tickers with ESG controversy spikes (simulate news-driven score jumps)
_CONTROVERSY_SPIKE_TICKERS = {"CYRX": [14, 24, 38], "FINX": [21, 35], "HVST": [19, 32]}


def _price_path(start: float, returns: np.ndarray) -> np.ndarray:
    returns = returns.copy()
    returns[0] = 0.0
    return start * np.cumprod(1.0 + returns)


def _make_benchmark(dates: pd.DatetimeIndex, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(dates)
    # Seasonal drift + noise; inject three market-wide stress periods
    seasonal = np.array([0.003 * np.sin(i / 6.0) for i in range(n)])
    shocks = rng.normal(loc=0.0010, scale=0.017, size=n)
    returns = seasonal + shocks
    # Market-wide drawdown episodes spread across the 5-year window
    stress_weeks = [28, 65, 104, 155, 210]
    for wk in stress_weeks:
        if wk < n:
            returns[wk] -= rng.uniform(0.045, 0.075)
    close = _price_path(100.0, returns)
    return pd.DataFrame({"date": dates, "benchmark_close": close}), returns


def _make_prices(
    dates: pd.DatetimeIndex,
    benchmark_returns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(dates)
    rows: list[dict] = []
    for ticker, spec in DEMO_TICKERS.items():
        idiosyncratic = rng.normal(loc=0.0006, scale=0.022, size=n)

        # Inject idiosyncratic crash events for selected tickers
        if ticker in _CRASH_TICKERS:
            crash_weeks = rng.choice(range(40, n - 20), size=3, replace=False)
            for wk in crash_weeks:
                idiosyncratic[wk] -= rng.uniform(0.09, 0.15)

        # Mild positive drift for "safe" healthcare names
        if spec["sector"] == "Healthcare":
            idiosyncratic += 0.0004

        returns = spec["beta"] * benchmark_returns + idiosyncratic
        prices = _price_path(spec["start"], returns)
        base_volume = rng.integers(500_000, 2_000_000)

        for idx, date in enumerate(dates):
            # Volume spikes coincide with controversy periods
            vol_multiplier = 1.0
            if ticker in _CRASH_TICKERS:
                spike_weeks = _CONTROVERSY_SPIKE_TICKERS.get(ticker, [])
                if any(abs(idx - 4 * sw) < 4 for sw in spike_weeks):
                    vol_multiplier = 1.6
            rows.append(
                {
                    "ticker": ticker,
                    "date": date.date().isoformat(),
                    "adj_close": round(float(prices[idx]), 4),
                    "volume": int(
                        (base_volume + idx * rng.integers(500, 5_000)) * vol_multiplier
                    ),
                }
            )
    return pd.DataFrame(rows)


def _make_fundamentals(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    period_ends = pd.date_range(
        dates.min() - pd.DateOffset(months=6), dates.max(), freq="QE"
    )
    for ticker, spec in DEMO_TICKERS.items():
        shares = rng.integers(60_000_000, 250_000_000)
        market_cap = shares * spec["start"]
        leverage_base = 0.35 if ticker in _CRASH_TICKERS else 0.22
        roa_base = 0.08 if ticker in _CRASH_TICKERS else 0.13
        for idx, period_end in enumerate(period_ends):
            leverage = leverage_base + 0.008 * idx + rng.normal(0, 0.015)
            roa = roa_base - 0.002 * idx + rng.normal(0, 0.008)
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": period_end.date().isoformat(),
                    "market_cap": round(float(market_cap * (1.0 + 0.014 * idx)), 2),
                    "shares_outstanding": int(shares),
                    "market_to_book": round(float(max(0.5, 1.4 + 0.06 * idx + rng.normal(0, 0.05))), 3),
                    "leverage": round(float(max(0.05, leverage)), 3),
                    "roa": round(float(roa), 3),
                }
            )
    return pd.DataFrame(rows)


def _make_controversies(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    # Monthly controversy updates (every 4 weeks)
    monthly_dates = dates[::4]
    for ticker, spec in DEMO_TICKERS.items():
        score = float(spec["controversy_base"])
        spike_months = _CONTROVERSY_SPIKE_TICKERS.get(ticker, [])
        for idx, date in enumerate(monthly_dates):
            # Gradual random walk
            score = max(0.0, score + rng.normal(0.04, 0.30))
            # Inject ESG controversy spikes (news events)
            if idx in spike_months:
                spike = rng.uniform(2.5, 4.5)
                score += spike
            # Mean reversion: high scores drift back toward baseline
            score = score * 0.92 + spec["controversy_base"] * 0.08
            score = min(score, 10.0)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date.date().isoformat(),
                    "sector": spec["sector"],
                    "controversy_score": round(float(score), 3),
                }
            )
    return pd.DataFrame(rows)


def write_demo_data(
    raw_dir: str | Path = "data/raw",
    weeks: int = 260,
    seed: int = 7,
) -> dict[str, Path]:
    """
    Generate synthetic raw data and write four CSV files to raw_dir.

    Parameters
    ----------
    raw_dir:  Output directory (created if it does not exist).
    weeks:    Number of weekly observations (default 260 ≈ 5 years).
    seed:     Random seed for reproducibility.

    Returns
    -------
    Dict mapping dataset name to file path.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-03", periods=weeks, freq="W-FRI")

    benchmark, benchmark_returns = _make_benchmark(dates, rng)
    files = {
        "benchmark_prices": raw_dir / "benchmark_prices.csv",
        "prices":           raw_dir / "prices.csv",
        "fundamentals":     raw_dir / "fundamentals.csv",
        "controversies":    raw_dir / "controversies.csv",
    }
    benchmark.to_csv(files["benchmark_prices"], index=False)
    _make_prices(dates, benchmark_returns, rng).to_csv(files["prices"], index=False)
    _make_fundamentals(dates, rng).to_csv(files["fundamentals"], index=False)
    _make_controversies(dates, rng).to_csv(files["controversies"], index=False)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic raw data for the crash-risk demo.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory to write the four raw CSV inputs.")
    parser.add_argument("--weeks", type=int, default=260, help="Number of weekly observations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    files = write_demo_data(raw_dir=args.raw_dir, weeks=args.weeks, seed=args.seed)
    for name, path in files.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
