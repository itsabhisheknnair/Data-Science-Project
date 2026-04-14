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


def _parse_dates(values: pd.Series) -> pd.Series:
    """
    Parse ISO, month-first, and day-first dates.

    Bloomberg/Excel exports often arrive as DD-MM-YYYY on machines using
    European locale settings. Pandas defaults to month-first parsing for
    ambiguous strings, which can silently scramble daily price histories.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    text = values.astype(str).str.strip()
    parts = text.str.extract(r"^(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})(?:\s.*)?$")
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    third = pd.to_numeric(parts[2], errors="coerce")

    # ISO YYYY-MM-DD needs no locale inference.
    iso_like = first.gt(31) & third.gt(31).eq(False)
    if iso_like.mean(skipna=True) > 0.8:
        return pd.to_datetime(text, errors="coerce")

    dayfirst_votes = ((first > 12) & (first <= 31) & (third > 31)).sum()
    monthfirst_votes = ((second > 12) & (second <= 31) & (third > 31)).sum()
    dayfirst = bool(dayfirst_votes > monthfirst_votes)
    return pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)


def load_prices(path: str | Path) -> pd.DataFrame:
    df = read_tabular(path)
    require_columns(df, PRICE_COLUMNS, "prices")
    require_non_empty(df, "prices")
    df = _standardize_ticker(df)
    df["date"] = _parse_dates(df["date"])
    df = _coerce_numeric(df, ("adj_close", "volume"))
    return df.dropna(subset=["ticker", "date", "adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_benchmark_prices(path: str | Path) -> pd.DataFrame:
    df = read_tabular(path)
    require_columns(df, BENCHMARK_COLUMNS, "benchmark_prices")
    require_non_empty(df, "benchmark_prices")
    df["date"] = _parse_dates(df["date"])
    df = _coerce_numeric(df, ("benchmark_close",))
    return df.dropna(subset=["date", "benchmark_close"]).sort_values("date").reset_index(drop=True)


def load_fundamentals(path: str | Path, config: CrashRiskConfig | None = None) -> pd.DataFrame:
    config = config or CrashRiskConfig()
    df = read_tabular(path)
    require_columns(df, FUNDAMENTAL_COLUMNS, "fundamentals")
    require_non_empty(df, "fundamentals")
    df = _standardize_ticker(df)
    df["period_end"] = _parse_dates(df["period_end"])
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
    df["date"] = _parse_dates(df["date"])
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
