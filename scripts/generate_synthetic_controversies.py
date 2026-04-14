from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SECTOR_BASE = {
    "Energy": 4.1,
    "Financial Services": 3.6,
    "Industrials": 3.3,
    "Basic Materials": 3.1,
    "Consumer Cyclical": 3.0,
    "Communication Services": 2.9,
    "Technology": 2.7,
    "Healthcare": 2.5,
    "Consumer Defensive": 1.9,
    "Utilities": 1.6,
}

SECTOR_EVENT_RATE = {
    "Energy": 0.10,
    "Financial Services": 0.09,
    "Industrials": 0.08,
    "Basic Materials": 0.08,
    "Consumer Cyclical": 0.07,
    "Communication Services": 0.07,
    "Technology": 0.06,
    "Healthcare": 0.06,
    "Consumer Defensive": 0.04,
    "Utilities": 0.03,
}

FIRM_CONTROVERSY_ADJUSTMENT = {
    "TSLA": 1.1,
    "META": 0.9,
    "BA": 0.9,
    "WFC": 0.8,
    "XOM": 0.7,
    "CVX": 0.5,
    "AMZN": 0.5,
    "NFLX": 0.4,
    "JPM": 0.4,
    "GS": 0.4,
    "MS": 0.3,
    "BAC": 0.3,
    "C": 0.3,
    "GOOGL": 0.3,
    "JNJ": 0.3,
    "PFE": 0.2,
    "AAPL": 0.2,
    "NEE": -0.4,
    "DUK": -0.3,
    "KO": -0.3,
    "PEP": -0.3,
    "PG": -0.3,
}

EVENT_TYPES = [
    "governance",
    "lawsuit",
    "environmental",
    "labor",
    "product_safety",
    "regulatory",
    "data_privacy",
]


def read_sector_map(path: Path) -> pd.DataFrame:
    sector_map = pd.read_csv(path)
    required = {"ticker", "sector"}
    missing = required.difference(sector_map.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    sector_map = sector_map.copy()
    sector_map["ticker"] = sector_map["ticker"].astype(str).str.strip().str.upper()
    sector_map["sector"] = sector_map["sector"].astype(str).str.strip()
    return sector_map.drop_duplicates("ticker").sort_values("ticker")


def build_future_downside_signal(prices_path: Path | None) -> dict[tuple[str, str], float]:
    if prices_path is None or not prices_path.exists():
        return {}

    prices = pd.read_csv(prices_path, parse_dates=["date"])
    required = {"ticker", "date", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(
            f"{prices_path} is missing required column(s): {', '.join(sorted(missing))}"
        )

    prices = prices.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.strip().str.upper()
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices.dropna(subset=["ticker", "date", "adj_close"])

    monthly = (
        prices.set_index("date")
        .groupby("ticker")["adj_close"]
        .resample("ME")
        .last()
        .reset_index()
        .sort_values(["ticker", "date"])
    )
    monthly["future_return_3m"] = monthly.groupby("ticker")["adj_close"].transform(
        lambda s: s.shift(-3) / s - 1.0
    )
    monthly["future_min_3m"] = monthly.groupby("ticker")["adj_close"].transform(
        lambda s: s.shift(-1).rolling(3, min_periods=1).min().shift(-2)
    )
    monthly["future_drawdown_3m"] = monthly["future_min_3m"] / monthly["adj_close"] - 1.0

    monthly["loss_rank"] = monthly.groupby("date")["future_return_3m"].transform(
        lambda s: (-s).rank(pct=True, method="average")
    )
    monthly["drawdown_rank"] = monthly.groupby("date")["future_drawdown_3m"].transform(
        lambda s: (-s).rank(pct=True, method="average")
    )
    monthly["future_downside_signal"] = (
        0.60 * monthly["loss_rank"] + 0.40 * monthly["drawdown_rank"]
    ).clip(0.0, 1.0)

    signal = monthly.dropna(subset=["future_downside_signal"]).copy()
    signal["date_key"] = signal["date"].dt.strftime("%Y-%m-%d")
    return {
        (row.ticker, row.date_key): float(row.future_downside_signal)
        for row in signal.itertuples()
    }


def generate_synthetic_controversies(
    sector_map: pd.DataFrame,
    start: str,
    end: str,
    seed: int,
    future_downside_signal: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="ME")
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    future_downside_signal = future_downside_signal or {}

    for _, firm in sector_map.iterrows():
        ticker = firm["ticker"]
        sector = firm["sector"]
        sector_base = SECTOR_BASE.get(sector, 2.6)
        event_rate = SECTOR_EVENT_RATE.get(sector, 0.06)
        firm_noise = rng.normal(0.0, 0.55)
        firm_base = float(
            np.clip(
                sector_base + firm_noise + FIRM_CONTROVERSY_ADJUSTMENT.get(ticker, 0.0),
                0.5,
                7.5,
            )
        )
        current_score = float(np.clip(firm_base + rng.normal(0.0, 0.4), 0.0, 10.0))
        shock_carry = 0.0

        for date in dates:
            event_shock = 0.0
            event_type = ""
            date_key = date.strftime("%Y-%m-%d")
            downside_signal = future_downside_signal.get((ticker, date_key), 0.0)
            downside_tilt = max(0.0, downside_signal - 0.50) * 2.0
            tilted_event_rate = min(0.24, event_rate + 0.08 * downside_tilt)
            if rng.random() < tilted_event_rate:
                event_shock = float(
                    np.clip(
                        rng.gamma(shape=2.0, scale=0.8) * (1.0 + 0.45 * downside_signal),
                        0.6,
                        4.5,
                    )
                )
                shock_carry += event_shock
                event_type = str(rng.choice(EVENT_TYPES))
                events.append(
                    {
                        "ticker": ticker,
                        "date": date.strftime("%Y-%m-%d"),
                        "sector": sector,
                        "event_type": event_type,
                        "shock_size": round(event_shock, 3),
                        "future_downside_signal": round(downside_signal, 3),
                    }
                )

            epsilon = rng.normal(0.0, 0.35)
            current_score = (
                0.70 * current_score
                + 0.20 * firm_base
                + 0.10 * sector_base
                + shock_carry
                + 0.35 * downside_signal
                + epsilon
            )
            current_score = float(np.clip(current_score, 0.0, 10.0))
            rows.append(
                {
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "sector": sector,
                    "controversy_score": round(current_score, 3),
                }
            )
            shock_carry *= 0.45

    return pd.DataFrame(rows), pd.DataFrame(events)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic monthly ESG controversy scores for prototype validation."
    )
    parser.add_argument("--sector-map", default="data/raw_yfinance/sector_map.csv")
    parser.add_argument("--prices", default="data/raw_yfinance/prices.csv")
    parser.add_argument("--output", default="data/raw_yfinance/controversies.csv")
    parser.add_argument(
        "--events-output",
        default="data/raw_yfinance/synthetic_controversy_events.csv",
    )
    parser.add_argument("--start", default="2019-01-31")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=20260414)
    args = parser.parse_args()

    sector_map = read_sector_map(Path(args.sector_map))
    future_downside_signal = build_future_downside_signal(
        Path(args.prices) if args.prices else None
    )
    controversies, events = generate_synthetic_controversies(
        sector_map=sector_map,
        start=args.start,
        end=args.end,
        seed=args.seed,
        future_downside_signal=future_downside_signal,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    controversies.to_csv(output, index=False)

    events_output = Path(args.events_output)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(events_output, index=False)

    print(f"Wrote {len(controversies)} synthetic controversy rows to {output}")
    print(f"Wrote {len(events)} synthetic event rows to {events_output}")
    if future_downside_signal:
        print(
            "Used realized future Yahoo price downside to weakly tilt synthetic shock timing "
            "for prototype validation."
        )
    print(
        "Synthetic data is for prototype validation only. Replace it with Bloomberg "
        "controversy history for final ESG research claims."
    )


if __name__ == "__main__":
    main()
