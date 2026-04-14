from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "yfinance is not installed. Install it with: pip install yfinance"
        ) from exc
    return yf


def configure_yfinance_cache(yf, output_dir: Path) -> None:
    cache_dir = output_dir / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))


def read_ticker_universe(path: Path) -> pd.DataFrame:
    tickers = pd.read_csv(path)
    required = {"bloomberg_ticker", "ticker"}
    missing = required.difference(tickers.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    tickers = tickers.copy()
    tickers["ticker"] = tickers["ticker"].astype(str).str.strip().str.upper()
    tickers["yahoo_ticker"] = tickers.get("yahoo_ticker", tickers["ticker"]).astype(str).str.strip()
    return tickers


def download_history(yf, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        actions=False,
        threads=True,
    )


def extract_symbol_history(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["date", "adj_close", "volume"])
    if isinstance(history.columns, pd.MultiIndex):
        if symbol not in history.columns.get_level_values(0):
            return pd.DataFrame(columns=["date", "adj_close", "volume"])
        frame = history[symbol].copy()
    else:
        frame = history.copy()

    price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if price_col not in frame.columns or "Volume" not in frame.columns:
        return pd.DataFrame(columns=["date", "adj_close", "volume"])

    out = frame[[price_col, "Volume"]].reset_index()
    out = out.rename(columns={"Date": "date", price_col: "adj_close", "Volume": "volume"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    return out.dropna(subset=["date", "adj_close"])


def build_prices(yf, tickers: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    symbols = tickers["yahoo_ticker"].tolist()
    history = download_history(yf, symbols, start, end)
    rows = []
    for _, row in tickers.iterrows():
        frame = extract_symbol_history(history, row["yahoo_ticker"])
        if frame.empty:
            print(f"WARNING: no Yahoo price history for {row['yahoo_ticker']}")
            continue
        frame.insert(0, "ticker", row["ticker"])
        rows.append(frame[["ticker", "date", "adj_close", "volume"]])
    if not rows:
        raise RuntimeError("No price history downloaded.")
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])


def build_benchmark(yf, benchmark: str, start: str, end: str) -> pd.DataFrame:
    history = download_history(yf, [benchmark], start, end)
    frame = extract_symbol_history(history, benchmark)
    if frame.empty:
        raise RuntimeError(f"No benchmark history downloaded for {benchmark}.")
    return frame.rename(columns={"adj_close": "benchmark_close"})[["date", "benchmark_close"]]


def _ratio(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(number):
        return np.nan
    return float(number / 100.0 if abs(number) > 10 else number)


def build_fundamentals(
    yf,
    tickers: pd.DataFrame,
    period_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fundamentals = []
    sectors = []
    for _, row in tickers.iterrows():
        symbol = row["yahoo_ticker"]
        ticker_obj = yf.Ticker(symbol)
        try:
            info = ticker_obj.info or {}
        except Exception as exc:
            print(f"WARNING: could not read Yahoo info for {symbol}: {exc}")
            info = {}

        sector = info.get("sector") or "Unknown"
        sectors.append({"ticker": row["ticker"], "sector": sector})
        fundamentals.append(
            {
                "ticker": row["ticker"],
                "period_end": period_end,
                "market_cap": pd.to_numeric(info.get("marketCap"), errors="coerce"),
                "shares_outstanding": pd.to_numeric(info.get("sharesOutstanding"), errors="coerce"),
                "market_to_book": pd.to_numeric(info.get("priceToBook"), errors="coerce"),
                "leverage": _ratio(info.get("debtToEquity")),
                "roa": _ratio(info.get("returnOnAssets")),
            }
        )
    return pd.DataFrame(fundamentals), pd.DataFrame(sectors)


def build_placeholder_controversies(sectors: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="ME")
    rows = []
    for _, row in sectors.iterrows():
        for date in dates:
            rows.append(
                {
                    "ticker": row["ticker"],
                    "date": date.strftime("%Y-%m-%d"),
                    "sector": row["sector"],
                    "controversy_score": 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull Yahoo Finance fallback data for the crash-risk pipeline."
    )
    parser.add_argument("--tickers-file", default="bloomberg/ticker_universe.csv")
    parser.add_argument("--output-dir", default="data/raw_yfinance")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--benchmark", default="^GSPC", help="Yahoo symbol for benchmark, e.g. ^GSPC or SPY.")
    parser.add_argument(
        "--fundamentals-period-end",
        default=None,
        help=(
            "Period_end to assign to Yahoo snapshot fundamentals. Defaults to 60 days "
            "before --start so the values are available throughout the demo panel. "
            "Replace with Bloomberg historical fundamentals for research results."
        ),
    )
    parser.add_argument(
        "--placeholder-controversies",
        action="store_true",
        help="Write controversies.csv with zero scores. This is only for plumbing tests, not the final ESG research claim.",
    )
    args = parser.parse_args()

    yf = _import_yfinance()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_yfinance_cache(yf, output_dir)
    tickers = read_ticker_universe(Path(args.tickers_file))

    prices = build_prices(yf, tickers, args.start, args.end)
    benchmark = build_benchmark(yf, args.benchmark, args.start, args.end)
    fundamentals_period_end = args.fundamentals_period_end
    if fundamentals_period_end is None:
        fundamentals_period_end = (
            pd.to_datetime(args.start) - pd.Timedelta(days=60)
        ).strftime("%Y-%m-%d")
    fundamentals, sectors = build_fundamentals(yf, tickers, fundamentals_period_end)

    prices.to_csv(output_dir / "prices.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_prices.csv", index=False)
    fundamentals.to_csv(output_dir / "fundamentals.csv", index=False)
    sectors.to_csv(output_dir / "sector_map.csv", index=False)

    print(f"Wrote {len(prices)} price rows to {output_dir / 'prices.csv'}")
    print(f"Wrote {len(benchmark)} benchmark rows to {output_dir / 'benchmark_prices.csv'}")
    print(f"Wrote {len(fundamentals)} fundamentals rows to {output_dir / 'fundamentals.csv'}")
    print(f"Wrote sector map to {output_dir / 'sector_map.csv'}")

    if args.placeholder_controversies:
        controversies = build_placeholder_controversies(sectors, args.start, args.end)
        controversies.to_csv(output_dir / "controversies.csv", index=False)
        print(
            f"Wrote placeholder controversies to {output_dir / 'controversies.csv'}; "
            "replace with Bloomberg/MSCI controversy history for the final project."
        )
    else:
        print(
            "Did not write controversies.csv. Pull Bloomberg/MSCI controversy history separately, "
            "or rerun with --placeholder-controversies only for pipeline plumbing tests."
        )


if __name__ == "__main__":
    main()
