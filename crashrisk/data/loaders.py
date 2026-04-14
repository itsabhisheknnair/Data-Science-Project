from __future__ import annotations

from pathlib import Path

import pandas as pd

from crashrisk.config import CrashRiskConfig, RawDataPaths
from crashrisk.data.validators import normalize_columns, require_columns, require_non_empty


PRICE_COLUMNS = ("ticker", "date", "adj_close", "volume")
BENCHMARK_COLUMNS = ("date", "benchmark_close")
FUNDAMENTAL_COLUMNS = (
    "ticker",
    "period_end",
    "market_cap",
    "shares_outstanding",
    "market_to_book",
    "leverage",
    "roa",
)
CONTROVERSY_COLUMNS = ("ticker", "date", "sector", "controversy_score")


def read_tabular(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input file extension for {path}. Use .csv, .xlsx, or .xls")
    return normalize_columns(df)


def _standardize_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def _coerce_numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    df = read_tabular(path)
    require_columns(df, PRICE_COLUMNS, "prices")
    require_non_empty(df, "prices")
    df = _standardize_ticker(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _coerce_numeric(df, ("adj_close", "volume"))
    return df.dropna(subset=["ticker", "date", "adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_benchmark_prices(path: str | Path) -> pd.DataFrame:
    df = read_tabular(path)
    require_columns(df, BENCHMARK_COLUMNS, "benchmark_prices")
    require_non_empty(df, "benchmark_prices")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _coerce_numeric(df, ("benchmark_close",))
    return df.dropna(subset=["date", "benchmark_close"]).sort_values("date").reset_index(drop=True)


def load_fundamentals(path: str | Path, config: CrashRiskConfig | None = None) -> pd.DataFrame:
    config = config or CrashRiskConfig()
    df = read_tabular(path)
    require_columns(df, FUNDAMENTAL_COLUMNS, "fundamentals")
    require_non_empty(df, "fundamentals")
    df = _standardize_ticker(df)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["available_date"] = df["period_end"] + pd.to_timedelta(config.fundamentals_lag_days, unit="D")
    df = _coerce_numeric(
        df,
        ("market_cap", "shares_outstanding", "market_to_book", "leverage", "roa"),
    )
    return df.dropna(subset=["ticker", "period_end", "available_date"]).sort_values(
        ["ticker", "available_date"]
    ).reset_index(drop=True)


def load_controversies(path: str | Path) -> pd.DataFrame:
    df = read_tabular(path)
    require_columns(df, CONTROVERSY_COLUMNS, "controversies")
    require_non_empty(df, "controversies")
    df = _standardize_ticker(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sector"] = df["sector"].astype(str).str.strip()
    df = _coerce_numeric(df, ("controversy_score",))
    return df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_raw_data(
    raw_paths: RawDataPaths | dict[str, str | Path],
    config: CrashRiskConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or CrashRiskConfig()
    paths = RawDataPaths.from_mapping(raw_paths)
    return {
        "prices": load_prices(paths.prices),
        "benchmark_prices": load_benchmark_prices(paths.benchmark_prices),
        "fundamentals": load_fundamentals(paths.fundamentals, config=config),
        "controversies": load_controversies(paths.controversies),
    }

