from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_FEATURE_COLUMNS = (
    "lagged_ncskew",
    "lagged_duvol",
    "detrended_turnover",
    "trailing_return",
    "realized_volatility",
    "beta",
    "downside_beta",
    "relative_downside_beta",
    "market_cap",
    "market_to_book",
    "leverage",
    "roa",
    "controversy_score",
    "controversy_change_4w",
    "controversy_change_13w",
    "controversy_change_26w",
    "controversy_rolling_mean_13w",
    "controversy_rolling_std_13w",
    "controversy_spike_flag",
    "controversy_sector_percentile",
)


@dataclass(frozen=True)
class RawDataPaths:
    prices: Path
    benchmark_prices: Path
    fundamentals: Path
    controversies: Path

    @classmethod
    def from_mapping(cls, raw_paths: Mapping[str, str | Path] | "RawDataPaths") -> "RawDataPaths":
        if isinstance(raw_paths, cls):
            return raw_paths
        required = ("prices", "benchmark_prices", "fundamentals", "controversies")
        missing = [name for name in required if name not in raw_paths]
        if missing:
            raise ValueError(f"Missing raw path(s): {', '.join(missing)}")
        return cls(**{name: Path(raw_paths[name]) for name in required})


@dataclass(frozen=True)
class CrashRiskConfig:
    week_rule: str = "W-FRI"
    fundamentals_lag_days: int = 45
    trailing_window_weeks: int = 26
    turnover_window_weeks: int = 26
    beta_window_weeks: int = 26
    min_crash_observations: int = 8
    target_horizon_weeks: int = 13
    target_top_quantile: float = 0.20
    controversy_windows: tuple[int, ...] = (4, 13, 26)
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_state: int = 42
    feature_columns: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURE_COLUMNS)
    # ML model selection: "logistic_regression", "random_forest", "gradient_boosting"
    model_type: str = "logistic_regression"
    # Number of time-series CV folds for hyperparameter tuning
    n_cv_splits: int = 5

    def min_beta_observations(self) -> int:
        return max(3, min(self.beta_window_weeks, self.min_crash_observations))


def discover_raw_paths(raw_dir: str | Path) -> RawDataPaths:
    raw_dir = Path(raw_dir)

    def find_file(stem: str) -> Path:
        candidates = [raw_dir / f"{stem}.csv", raw_dir / f"{stem}.xlsx", raw_dir / f"{stem}.xls"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        allowed = ", ".join(candidate.name for candidate in candidates)
        raise FileNotFoundError(f"Could not find {stem} input. Expected one of: {allowed}")

    return RawDataPaths(
        prices=find_file("prices"),
        benchmark_prices=find_file("benchmark_prices"),
        fundamentals=find_file("fundamentals"),
        controversies=find_file("controversies"),
    )


def ensure_columns_exist(columns: Iterable[str], available: Iterable[str], context: str) -> list[str]:
    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"{context} is missing required column(s): {', '.join(missing)}")
    return list(columns)

