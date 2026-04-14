from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class SchemaError(ValueError):
    """Raised when an input file does not match the expected schema."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().lower() for column in df.columns}
    return df.rename(columns=renamed)


def require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SchemaError(f"{dataset_name} is missing required column(s): {', '.join(missing)}")


def require_non_empty(df: pd.DataFrame, dataset_name: str) -> None:
    if df.empty:
        raise SchemaError(f"{dataset_name} is empty")

