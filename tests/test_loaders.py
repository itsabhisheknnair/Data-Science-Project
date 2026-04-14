from __future__ import annotations

import pandas as pd
import pytest

from crashrisk.config import CrashRiskConfig
from crashrisk.data.loaders import load_controversies, load_fundamentals, load_prices
from crashrisk.data.validators import SchemaError


def test_load_prices_csv_standardizes_tickers_and_dates(workspace_tmp_path):
    path = workspace_tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Ticker": [" aaa "],
            "Date": ["2020-01-03"],
            "Adj_Close": [100],
            "Volume": [10_000],
        }
    ).to_csv(path, index=False)

    prices = load_prices(path)

    assert prices.loc[0, "ticker"] == "AAA"
    assert prices.loc[0, "date"] == pd.Timestamp("2020-01-03")
    assert prices.loc[0, "adj_close"] == 100


def test_load_controversies_supports_excel(workspace_tmp_path):
    path = workspace_tmp_path / "controversies.xlsx"
    pd.DataFrame(
        {
            "ticker": ["bbb"],
            "date": ["2020-01-03"],
            "sector": ["Tech"],
            "controversy_score": [3],
        }
    ).to_excel(path, index=False)

    controversies = load_controversies(path)

    assert controversies.loc[0, "ticker"] == "BBB"
    assert controversies.loc[0, "controversy_score"] == 3


def test_missing_required_columns_fail_clearly(workspace_tmp_path):
    path = workspace_tmp_path / "bad_prices.csv"
    pd.DataFrame({"ticker": ["AAA"], "date": ["2020-01-03"]}).to_csv(path, index=False)

    with pytest.raises(SchemaError, match="adj_close"):
        load_prices(path)


def test_fundamentals_apply_availability_lag(workspace_tmp_path):
    path = workspace_tmp_path / "fundamentals.csv"
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "period_end": ["2020-12-31"],
            "market_cap": [1_000],
            "shares_outstanding": [100],
            "market_to_book": [2.0],
            "leverage": [0.3],
            "roa": [0.1],
        }
    ).to_csv(path, index=False)

    fundamentals = load_fundamentals(path, config=CrashRiskConfig(fundamentals_lag_days=45))

    assert fundamentals.loc[0, "available_date"] == pd.Timestamp("2021-02-14")
