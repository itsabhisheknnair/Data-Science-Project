from __future__ import annotations

import html
import json
import re
import shutil
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from wordcloud import WordCloud

from crashrisk.config import CrashRiskConfig, RawDataPaths
from crashrisk.data.loaders import (
    BENCHMARK_COLUMNS,
    CONTROVERSY_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    PRICE_COLUMNS,
    _parse_dates,
    load_raw_data,
    read_tabular,
)


TEXT_FILE_STEMS = ("controversy_text", "news_text", "textual_data")
TEXT_COLUMNS = ("headline", "title", "description", "body", "text", "summary")
TEXT_COLUMN_ALIASES = {
    "ticker": ("ticker", "symbol", "security", "stock", "companyticker", "tickercode"),
    "date": ("date", "publishdate", "publisheddate", "publicationdate", "storydate", "newsdate", "datetime", "timestamp"),
    "source": ("source", "provider", "newssource", "publisher", "vendor"),
    "headline": ("headline", "headlinetext", "newsheadline"),
    "title": ("title", "storytitle", "newstitle"),
    "description": ("description", "subtitle", "snippet", "abstract", "deck"),
    "body": ("body", "story", "storytext", "article", "articletext", "content"),
    "text": ("text", "fulltext", "documenttext", "newstext"),
    "summary": ("summary", "brief", "synopsis"),
}
NEGATIVE_WORDS = {
    "abuse",
    "accident",
    "allegation",
    "breach",
    "bribery",
    "collapse",
    "controversy",
    "corruption",
    "crash",
    "crisis",
    "default",
    "downgrade",
    "emissions",
    "fraud",
    "investigation",
    "lawsuit",
    "loss",
    "misconduct",
    "pollution",
    "probe",
    "recall",
    "risk",
    "scandal",
    "strike",
    "violation",
}
POSITIVE_WORDS = {
    "award",
    "benefit",
    "clean",
    "improve",
    "improved",
    "positive",
    "progress",
    "resolve",
    "resolved",
    "safe",
    "settle",
    "settled",
    "upgrade",
}
CONTROVERSY_KEYWORDS = {
    "bribery",
    "corruption",
    "emissions",
    "fraud",
    "governance",
    "investigation",
    "lawsuit",
    "misconduct",
    "pollution",
    "scandal",
    "social",
    "violation",
}
STOP_WORDS = {
    "a",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}
BIGRAM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    STOP_WORDS,
    {
        "analysts",
        "article",
        "commentary",
        "company",
        "coverage",
        "esg",
        "external",
        "faces",
        "firm",
        "follow",
        "group",
        "headlines",
        "highlights",
        "linked",
        "market",
        "monitor",
        "month",
        "news",
        "note",
        "ongoing",
        "reports",
        "review",
        "risk",
        "risks",
        "said",
        "score",
        "signal",
        "signals",
        "text",
    },
)
FEATURE_GROUPS = {
    "lagged_ncskew": "Crash history",
    "lagged_duvol": "Crash history",
    "detrended_turnover": "Trading activity",
    "trailing_return": "Downside risk",
    "realized_volatility": "Downside risk",
    "beta": "Downside risk",
    "downside_beta": "Downside risk",
    "relative_downside_beta": "Downside risk",
    "market_cap": "Fundamentals",
    "market_to_book": "Fundamentals",
    "leverage": "Fundamentals",
    "roa": "Fundamentals",
    "controversy_score": "ESG controversy",
    "controversy_change_4w": "ESG controversy",
    "controversy_change_13w": "ESG controversy",
    "controversy_change_26w": "ESG controversy",
    "controversy_rolling_mean_13w": "ESG controversy",
    "controversy_rolling_std_13w": "ESG controversy",
    "controversy_spike_flag": "ESG controversy",
    "controversy_sector_percentile": "ESG controversy",
    "negative_esg_controversy_score_0_100": "Text signal",
    "rolling_sentiment_13w": "Text signal",
}
TEXT_SIGNAL_COLUMNS = (
    "negative_esg_controversy_score_0_100",
    "rolling_sentiment_13w",
)


def build_report_artifacts(
    raw_paths: RawDataPaths | dict[str, str | Path],
    feature_panel: pd.DataFrame,
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    price_history: pd.DataFrame,
    price_scenarios: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_comparison: pd.DataFrame,
    outputs_dir: str | Path,
    config: CrashRiskConfig | None = None,
    text_outputs: dict[str, pd.DataFrame] | None = None,
    calibration_curve: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build FDS report-ready outputs without changing the model signal."""
    config = config or CrashRiskConfig()
    paths = RawDataPaths.from_mapping(raw_paths)
    raw = load_raw_data(paths, config=config)
    outputs_dir = Path(outputs_dir)

    data_summary, cleaning_log = build_data_summary(paths, raw, feature_panel, dataset, scores, config)
    feature_descriptive_stats = build_feature_descriptive_stats(feature_panel, config)
    feature_correlation_matrix = build_feature_correlation_matrix(feature_panel, config)
    sql_summary = build_sql_summary(raw, feature_panel, dataset, scores)
    if text_outputs is None:
        text_outputs = build_text_analysis_outputs(paths, config=config)
    text_coverage = build_text_coverage(text_outputs["weekly"], dataset, config=config)
    lda_outputs = build_lda_topic_outputs(paths, config=config)
    write_sql_summary_markdown(sql_summary, outputs_dir / "sql_summary.md")
    write_report_figures(
        feature_panel=feature_panel,
        scores=scores,
        price_scenarios=price_scenarios,
        feature_importance=feature_importance,
        model_comparison=model_comparison,
        figures_dir=outputs_dir / "figures",
        feature_correlation_matrix=feature_correlation_matrix,
        calibration_curve=calibration_curve,
        lda_ticker_topics=lda_outputs["ticker_topics"],
    )
    write_text_word_cloud(paths, outputs_dir / "figures" / "text_word_cloud.svg")
    write_price_time_series(feature_panel, scores, outputs_dir / "figures" / "price_time_series.png")
    write_report_outline(
        data_summary=data_summary,
        model_comparison=model_comparison,
        outputs_dir=outputs_dir,
    )

    return {
        "data_summary": data_summary,
        "cleaning_log": cleaning_log,
        "feature_descriptive_stats": feature_descriptive_stats,
        "feature_correlation_matrix": feature_correlation_matrix,
        "sql_summary": sql_summary,
        "textual_analysis": text_outputs["weekly"],
        "textual_ticker_summary": text_outputs["ticker_summary"],
        "text_coverage": text_coverage,
        "lda_topic_words": lda_outputs["topic_words"],
        "lda_ticker_topics": lda_outputs["ticker_topics"],
    }


def build_data_summary(
    paths: RawDataPaths,
    raw: dict[str, pd.DataFrame],
    feature_panel: pd.DataFrame,
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    config: CrashRiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    cleaning_rows: list[dict[str, object]] = []

    specs = {
        "prices": (paths.prices, PRICE_COLUMNS, "date"),
        "benchmark_prices": (paths.benchmark_prices, BENCHMARK_COLUMNS, "date"),
        "fundamentals": (paths.fundamentals, FUNDAMENTAL_COLUMNS, "period_end"),
        "controversies": (paths.controversies, CONTROVERSY_COLUMNS, "date"),
    }

    for name, (path, required_columns, date_column) in specs.items():
        original = read_tabular(path)
        loaded = raw[name]
        add_summary(summary_rows, name, "source_file", Path(path).name, "Input file used for this run.")
        add_summary(summary_rows, name, "raw_rows", len(original), "Rows before loader cleaning.")
        add_summary(summary_rows, name, "loaded_rows", len(loaded), "Rows after parsing, type coercion, and required-date filtering.")
        if "ticker" in loaded:
            add_summary(summary_rows, name, "ticker_count", loaded["ticker"].nunique(), "Distinct tickers after loading.")
        if "sector" in loaded:
            add_summary(summary_rows, name, "sector_count", loaded["sector"].nunique(), "Distinct sectors after loading.")
        if date_column in loaded:
            dates = pd.to_datetime(loaded[date_column], errors="coerce").dropna()
            add_summary(summary_rows, name, "date_start", dates.min().date() if not dates.empty else "", "Earliest loaded date.")
            add_summary(summary_rows, name, "date_end", dates.max().date() if not dates.empty else "", "Latest loaded date.")

        blank_required = 0
        missing_required_rows = pd.Series(False, index=original.index)
        for column in required_columns:
            if column in original:
                values = original[column]
                blank_mask = values.isna()
                if values.dtype == object:
                    blank_mask = blank_mask | values.astype(str).str.strip().eq("")
                blank_required += int(blank_mask.sum())
                missing_required_rows = missing_required_rows | blank_mask
        duplicates = int(original.duplicated().sum())
        dropped = max(0, len(original) - len(loaded))
        add_cleaning(
            cleaning_rows,
            name,
            "missing_required_values",
            blank_required,
            "Blank or NA cells in required columns before loading; zero means the raw required fields were complete.",
            denominator=max(1, len(original) * len(required_columns)),
        )
        add_cleaning(
            cleaning_rows,
            name,
            "missing_required_rows",
            int(missing_required_rows.sum()),
            "Rows containing at least one blank required field before loading.",
            denominator=max(1, len(original)),
        )
        add_cleaning(
            cleaning_rows,
            name,
            "duplicate_rows",
            duplicates,
            "Fully duplicated raw rows; zero means no exact duplicate records were removed.",
            denominator=max(1, len(original)),
        )
        add_cleaning(
            cleaning_rows,
            name,
            "rows_removed_or_invalid",
            dropped,
            "Rows not retained by the validated loader after parsing, type coercion, and required-date filtering.",
            denominator=max(1, len(original)),
        )
        if date_column in original:
            parsed_dates = _parse_dates(original[date_column])
            add_cleaning(
                cleaning_rows,
                name,
                f"invalid_{date_column}_values",
                int(parsed_dates.isna().sum()),
                f"Rows where {date_column} could not be parsed as a date.",
                denominator=max(1, len(original)),
            )
        id_columns = ["ticker", date_column] if "ticker" in original.columns else [date_column]
        if all(column in original.columns for column in id_columns):
            add_cleaning(
                cleaning_rows,
                name,
                "duplicate_entity_date_rows",
                int(original.duplicated(subset=id_columns).sum()),
                f"Duplicate rows by {', '.join(id_columns)} before loading.",
                denominator=max(1, len(original)),
            )
        for column in numeric_columns_for_dataset(name):
            if column in original:
                raw_values = original[column]
                nonblank = raw_values.notna()
                if raw_values.dtype == object:
                    nonblank = nonblank & raw_values.astype(str).str.strip().ne("")
                coerced = pd.to_numeric(raw_values, errors="coerce")
                add_cleaning(
                    cleaning_rows,
                    name,
                    f"{column}_numeric_coercion_failures",
                    int((nonblank & coerced.isna()).sum()),
                    f"Nonblank {column} values that could not be coerced to numeric.",
                    denominator=max(1, int(nonblank.sum())),
                )

    prices_original = read_tabular(paths.prices)
    if "adj_close" in prices_original:
        prices_numeric = pd.to_numeric(prices_original["adj_close"], errors="coerce")
        add_cleaning(
            cleaning_rows,
            "prices",
            "zero_or_negative_adj_close",
            int((prices_numeric <= 0).sum()),
            "Rows with non-positive adjusted prices; zero confirms all retained price levels are positive.",
            denominator=max(1, len(prices_original)),
        )
    if "volume" in prices_original:
        volume_numeric = pd.to_numeric(prices_original["volume"], errors="coerce")
        add_cleaning(
            cleaning_rows,
            "prices",
            "zero_or_negative_volume",
            int((volume_numeric <= 0).sum()),
            "Rows with non-positive trading volume; zero confirms volume inputs are positive.",
            denominator=max(1, len(prices_original)),
        )

    benchmark_original = read_tabular(paths.benchmark_prices)
    if "benchmark_close" in benchmark_original:
        benchmark_numeric = pd.to_numeric(benchmark_original["benchmark_close"], errors="coerce")
        add_cleaning(
            cleaning_rows,
            "benchmark_prices",
            "zero_or_negative_benchmark_close",
            int((benchmark_numeric <= 0).sum()),
            "Rows with non-positive benchmark prices; zero confirms the market proxy is positive.",
            denominator=max(1, len(benchmark_original)),
        )

    fundamentals_original = read_tabular(paths.fundamentals)
    for column in ("market_cap", "shares_outstanding", "market_to_book", "leverage", "roa"):
        if column in fundamentals_original:
            values = pd.to_numeric(fundamentals_original[column], errors="coerce")
            missing_count = int(values.isna().sum())
            add_cleaning(
                cleaning_rows,
                "fundamentals",
                f"retained_missing_{column}",
                missing_count,
                f"Missing {column} values retained for model-stage median imputation; zero means no imputation was needed for this field.",
                denominator=max(1, len(fundamentals_original)),
            )
    for column in ("market_cap", "shares_outstanding"):
        if column in fundamentals_original:
            values = pd.to_numeric(fundamentals_original[column], errors="coerce")
            add_cleaning(
                cleaning_rows,
                "fundamentals",
                f"zero_or_negative_{column}",
                int((values <= 0).sum()),
                f"Rows with non-positive {column}; zero confirms the scale input is positive.",
                denominator=max(1, len(fundamentals_original)),
            )

    controversies_original = read_tabular(paths.controversies)
    if "controversy_score" in controversies_original:
        controversy_numeric = pd.to_numeric(controversies_original["controversy_score"], errors="coerce")
        add_cleaning(
            cleaning_rows,
            "controversies",
            "negative_controversy_score",
            int((controversy_numeric < 0).sum()),
            "Rows with negative controversy scores; zero confirms scores are non-negative.",
            denominator=max(1, len(controversies_original)),
        )

    add_summary(summary_rows, "feature_panel", "rows", len(feature_panel), "Weekly ticker-date feature rows.")
    add_summary(summary_rows, "feature_panel", "ticker_count", feature_panel["ticker"].nunique(), "Distinct tickers in the engineered panel.")
    add_summary(summary_rows, "model_dataset", "rows", len(dataset), "Rows with future crash-risk labels.")
    add_summary(summary_rows, "stock_scores", "rows", len(scores), "Latest scored names shown in the dashboard.")
    add_summary(summary_rows, "configuration", "fundamentals_lag_days", config.fundamentals_lag_days, "Fundamentals availability lag used to reduce look-ahead bias.")
    add_summary(summary_rows, "configuration", "target_horizon_weeks", config.target_horizon_weeks, "Future window used for crash-risk target creation.")

    add_cleaning(
        cleaning_rows,
        "fundamentals",
        "availability_lag_days",
        config.fundamentals_lag_days,
        "Fundamentals become usable only after this many calendar days from period_end.",
    )
    add_cleaning(
        cleaning_rows,
        "feature_engineering",
        "date_alignment_method",
        "weekly Friday observations",
        "Daily raw inputs are aligned to Friday week-end observations before modeling.",
    )
    add_cleaning(
        cleaning_rows,
        "target_creation",
        "future_window",
        f"t+1 through t+{config.target_horizon_weeks}",
        "Targets use future weeks only; features at t use data available at or before t.",
    )

    return pd.DataFrame(summary_rows), pd.DataFrame(cleaning_rows)


def add_summary(rows: list[dict[str, object]], section: str, metric: str, value: object, detail: str) -> None:
    rows.append({"section": section, "metric": metric, "value": value, "detail": detail})


def add_cleaning(
    rows: list[dict[str, object]],
    dataset: str,
    check: str,
    value: object,
    detail: str,
    denominator: int | None = None,
) -> None:
    percent = ""
    if denominator is not None:
        try:
            percent = round(float(value) / denominator * 100.0, 4)
        except (TypeError, ValueError, ZeroDivisionError):
            percent = ""
    rows.append(
        {
            "dataset": dataset,
            "check": check,
            "value": value,
            "denominator": denominator if denominator is not None else "",
            "percent": percent,
            "detail": detail,
        }
    )


def numeric_columns_for_dataset(name: str) -> tuple[str, ...]:
    if name == "prices":
        return ("adj_close", "volume")
    if name == "benchmark_prices":
        return ("benchmark_close",)
    if name == "fundamentals":
        return ("market_cap", "shares_outstanding", "market_to_book", "leverage", "roa")
    if name == "controversies":
        return ("controversy_score",)
    return ()


def build_feature_descriptive_stats(feature_panel: pd.DataFrame, config: CrashRiskConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_rows = len(feature_panel)
    for feature in config.feature_columns:
        if feature not in feature_panel.columns:
            rows.append(
                {
                    "feature": feature,
                    "feature_group": FEATURE_GROUPS.get(feature, "Other"),
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "null_count": n_rows,
                    "null_percent": 100.0 if n_rows else np.nan,
                }
            )
            continue
        values = pd.to_numeric(feature_panel[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "feature_group": FEATURE_GROUPS.get(feature, "Other"),
                "count": int(values.notna().sum()),
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
                "min": float(values.min()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
                "null_count": int(values.isna().sum()),
                "null_percent": round(float(values.isna().mean() * 100.0), 4) if n_rows else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_feature_correlation_matrix(feature_panel: pd.DataFrame, config: CrashRiskConfig) -> pd.DataFrame:
    features = list(config.feature_columns)
    matrix_input = pd.DataFrame(index=feature_panel.index)
    for feature in features:
        matrix_input[feature] = pd.to_numeric(
            feature_panel[feature] if feature in feature_panel.columns else pd.Series(np.nan, index=feature_panel.index),
            errors="coerce",
        )
    corr = matrix_input.corr(method="pearson")
    corr = corr.reindex(index=features, columns=features)
    corr.insert(0, "feature", corr.index)
    return corr.reset_index(drop=True)


def join_text_signals_to_panel(feature_panel: pd.DataFrame, weekly_text: pd.DataFrame) -> pd.DataFrame:
    panel = feature_panel.copy()
    for column in (*TEXT_SIGNAL_COLUMNS, "text_article_count"):
        if column not in panel.columns:
            panel[column] = np.nan

    if weekly_text.empty or "status" not in weekly_text.columns or not weekly_text["status"].eq("ok").any():
        return panel

    required = {"ticker", "date", *TEXT_SIGNAL_COLUMNS}
    if not required.issubset(weekly_text.columns):
        return panel

    text_columns = ["ticker", "date", *TEXT_SIGNAL_COLUMNS]
    if "article_count" in weekly_text.columns:
        text_columns.append("article_count")
    text_features = weekly_text.loc[weekly_text["status"].eq("ok"), text_columns].copy()
    text_features["ticker"] = text_features["ticker"].astype(str).str.upper().str.strip()
    text_features["date"] = pd.to_datetime(text_features["date"], errors="coerce")
    text_features = text_features.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    if "article_count" in text_features.columns:
        text_features = text_features.rename(columns={"article_count": "text_article_count"})

    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.drop(columns=[column for column in (*TEXT_SIGNAL_COLUMNS, "text_article_count") if column in panel.columns])
    panel = panel.merge(text_features, on=["ticker", "date"], how="left")
    return panel


def build_text_coverage(
    weekly_text: pd.DataFrame,
    dataset: pd.DataFrame,
    config: CrashRiskConfig | None = None,
) -> pd.DataFrame:
    config = config or CrashRiskConfig()
    columns = [
        "split",
        "status",
        "source_file",
        "article_count",
        "weekly_text_rows",
        "unique_tickers",
        "date_start",
        "date_end",
        "avg_articles_per_ticker",
        "matched_model_rows",
    ]
    if weekly_text.empty or "status" not in weekly_text.columns or not weekly_text["status"].eq("ok").any():
        status = weekly_text.iloc[0].get("status", "no_text_file") if not weekly_text.empty else "no_text_file"
        source_file = weekly_text.iloc[0].get("source_file", "") if not weekly_text.empty else ""
        return pd.DataFrame(
            [
                {
                    "split": "overall",
                    "status": status,
                    "source_file": source_file,
                    "article_count": 0,
                    "weekly_text_rows": 0,
                    "unique_tickers": 0,
                    "date_start": "",
                    "date_end": "",
                    "avg_articles_per_ticker": np.nan,
                    "matched_model_rows": 0,
                }
            ],
            columns=columns,
        )

    text_ok = weekly_text.loc[weekly_text["status"].eq("ok")].copy()
    text_ok["date"] = pd.to_datetime(text_ok["date"], errors="coerce")
    text_ok = text_ok.dropna(subset=["ticker", "date"])

    labeled = dataset.dropna(subset=["high_crash_risk"]).copy()
    labeled["date"] = pd.to_datetime(labeled["date"], errors="coerce")
    splits = chronological_split_for_coverage(labeled, config)

    rows = [summarize_text_coverage("overall", text_ok, labeled)]
    for split_name, split_frame in splits.items():
        split_dates = set(split_frame["date"].dropna().unique())
        split_text = text_ok.loc[text_ok["date"].isin(split_dates)].copy()
        rows.append(summarize_text_coverage(split_name, split_text, split_frame))
    source_file = str(text_ok["source_file"].dropna().iloc[0]) if "source_file" in text_ok and text_ok["source_file"].notna().any() else ""
    for row in rows:
        row["status"] = "ok"
        row["source_file"] = source_file
    return pd.DataFrame(rows, columns=columns)


def chronological_split_for_coverage(dataset: pd.DataFrame, config: CrashRiskConfig) -> dict[str, pd.DataFrame]:
    sorted_df = dataset.sort_values("date").copy()
    unique_dates = pd.Series(sorted_df["date"].dropna().sort_values().unique())
    if len(unique_dates) < 3:
        return {"train": sorted_df.iloc[0:0], "validation": sorted_df.iloc[0:0], "test": sorted_df}
    train_end = max(1, int(len(unique_dates) * config.train_fraction))
    validation_end = max(train_end + 1, int(len(unique_dates) * (config.train_fraction + config.validation_fraction)))
    validation_end = min(validation_end, len(unique_dates) - 1)
    train_dates = set(unique_dates.iloc[:train_end])
    validation_dates = set(unique_dates.iloc[train_end:validation_end])
    test_dates = set(unique_dates.iloc[validation_end:])
    return {
        "train": sorted_df.loc[sorted_df["date"].isin(train_dates)].copy(),
        "validation": sorted_df.loc[sorted_df["date"].isin(validation_dates)].copy(),
        "test": sorted_df.loc[sorted_df["date"].isin(test_dates)].copy(),
    }


def summarize_text_coverage(split_name: str, text_frame: pd.DataFrame, model_frame: pd.DataFrame) -> dict[str, object]:
    if text_frame.empty:
        return {
            "split": split_name,
            "article_count": 0,
            "weekly_text_rows": 0,
            "unique_tickers": 0,
            "date_start": "",
            "date_end": "",
            "avg_articles_per_ticker": np.nan,
            "matched_model_rows": 0,
        }
    article_count = int(pd.to_numeric(text_frame.get("article_count", 0), errors="coerce").fillna(0).sum())
    matched = model_frame.merge(text_frame[["ticker", "date"]].drop_duplicates(), on=["ticker", "date"], how="inner")
    unique_tickers = int(text_frame["ticker"].nunique())
    return {
        "split": split_name,
        "article_count": article_count,
        "weekly_text_rows": int(len(text_frame)),
        "unique_tickers": unique_tickers,
        "date_start": text_frame["date"].min().date().isoformat(),
        "date_end": text_frame["date"].max().date().isoformat(),
        "avg_articles_per_ticker": round(article_count / unique_tickers, 4) if unique_tickers else np.nan,
        "matched_model_rows": int(len(matched)),
    }


def build_lda_topic_outputs(
    paths: RawDataPaths,
    config: CrashRiskConfig | None = None,
    n_topics: int = 5,
    top_words: int = 10,
) -> dict[str, pd.DataFrame]:
    config = config or CrashRiskConfig()
    prepared = prepare_text_records(paths)
    topic_columns = ["status", "source_file", "topic", "rank", "word", "weight"]
    ticker_columns = [
        "status",
        "source_file",
        "ticker",
        "article_count",
        "date_start",
        "date_end",
        "dominant_topic",
        "topic_probability",
    ]
    if prepared["status"] != "ok":
        row = prepared["status_row"]
        return {
            "topic_words": pd.DataFrame(
                [
                    {
                        "status": row.get("status", "no_text_file"),
                        "source_file": row.get("source_file", ""),
                        "topic": "",
                        "rank": "",
                        "word": "",
                        "weight": np.nan,
                    }
                ],
                columns=topic_columns,
            ),
            "ticker_topics": pd.DataFrame(
                [
                    {
                        "status": row.get("status", "no_text_file"),
                        "source_file": row.get("source_file", ""),
                        "ticker": "",
                        "article_count": 0,
                        "date_start": "",
                        "date_end": "",
                        "dominant_topic": "",
                        "topic_probability": np.nan,
                    }
                ],
                columns=ticker_columns,
            ),
        }

    text_df = prepared["text_df"].copy()
    source_file = prepared["text_path"].name
    ticker_stop_words = set(text_df["ticker"].astype(str).str.lower().unique())
    stop_words = BIGRAM_STOP_WORDS.union(ticker_stop_words)
    vectorizer = CountVectorizer(
        stop_words=sorted(stop_words),
        max_features=2000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        matrix = vectorizer.fit_transform(text_df["text"].fillna("").astype(str))
    except ValueError:
        return empty_lda_outputs("empty_vocabulary", source_file, topic_columns, ticker_columns)
    if matrix.shape[1] == 0 or matrix.shape[0] == 0:
        return empty_lda_outputs("empty_vocabulary", source_file, topic_columns, ticker_columns)

    topic_count = max(1, min(n_topics, matrix.shape[0]))
    lda = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=config.random_state,
        learning_method="batch",
    )
    doc_topics = lda.fit_transform(matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())

    topic_rows: list[dict[str, object]] = []
    for topic_idx, component in enumerate(lda.components_, start=1):
        top_indices = np.argsort(component)[::-1][:top_words]
        for rank, term_idx in enumerate(top_indices, start=1):
            topic_rows.append(
                {
                    "status": "ok",
                    "source_file": source_file,
                    "topic": f"topic_{topic_idx}",
                    "rank": rank,
                    "word": str(terms[term_idx]),
                    "weight": float(component[term_idx]),
                }
            )

    topic_labels = [f"topic_{idx}" for idx in range(1, topic_count + 1)]
    topic_frame = pd.DataFrame(doc_topics, columns=topic_labels)
    topic_frame.insert(0, "ticker", text_df["ticker"].astype(str).str.upper().str.strip().to_numpy())
    topic_frame.insert(1, "date", pd.to_datetime(text_df["date"], errors="coerce").to_numpy())
    topic_means = topic_frame.groupby("ticker", as_index=False)[topic_labels].mean()
    ticker_meta = (
        topic_frame.groupby("ticker", as_index=False)
        .agg(article_count=("ticker", "size"), date_start=("date", "min"), date_end=("date", "max"))
    )
    ticker_summary = ticker_meta.merge(topic_means, on="ticker", how="left")
    ticker_summary["dominant_topic"] = ticker_summary[topic_labels].idxmax(axis=1)
    ticker_summary["topic_probability"] = ticker_summary[topic_labels].max(axis=1)
    ticker_summary.insert(0, "status", "ok")
    ticker_summary.insert(1, "source_file", source_file)
    ticker_summary["date_start"] = ticker_summary["date_start"].dt.date.astype(str)
    ticker_summary["date_end"] = ticker_summary["date_end"].dt.date.astype(str)

    return {
        "topic_words": pd.DataFrame(topic_rows, columns=topic_columns),
        "ticker_topics": ticker_summary[
            ["status", "source_file", "ticker", "article_count", "date_start", "date_end", "dominant_topic", "topic_probability", *topic_labels]
        ],
    }


def empty_lda_outputs(
    status: str,
    source_file: str,
    topic_columns: list[str],
    ticker_columns: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        "topic_words": pd.DataFrame(
            [{"status": status, "source_file": source_file, "topic": "", "rank": "", "word": "", "weight": np.nan}],
            columns=topic_columns,
        ),
        "ticker_topics": pd.DataFrame(
            [
                {
                    "status": status,
                    "source_file": source_file,
                    "ticker": "",
                    "article_count": 0,
                    "date_start": "",
                    "date_end": "",
                    "dominant_topic": "",
                    "topic_probability": np.nan,
                }
            ],
            columns=ticker_columns,
        ),
    }


def build_sql_summary(
    raw: dict[str, pd.DataFrame],
    feature_panel: pd.DataFrame,
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    with sqlite3.connect(":memory:") as conn:
        raw["prices"].to_sql("raw_prices", conn, index=False, if_exists="replace")
        raw["controversies"].to_sql("raw_controversies", conn, index=False, if_exists="replace")
        feature_panel.to_sql("feature_panel", conn, index=False, if_exists="replace")
        dataset.to_sql("model_dataset", conn, index=False, if_exists="replace")

        scores_enriched = scores.copy()
        # Only merge sector from panel if scores doesn't already carry it;
        # merging when sector exists in both sides produces sector_x / sector_y.
        if "sector" not in scores_enriched.columns and "sector" in feature_panel.columns:
            latest_sector = (
                feature_panel.dropna(subset=["sector"])
                .sort_values(["ticker", "date"])
                .drop_duplicates("ticker", keep="last")[["ticker", "sector"]]
            )
            scores_enriched = scores_enriched.merge(latest_sector, on="ticker", how="left")
        scores_enriched.to_sql("stock_scores", conn, index=False, if_exists="replace")

        queries = {
            "observations_by_ticker": """
                select ticker, count(*) as observations, min(date) as start_date, max(date) as end_date
                from raw_prices
                group by ticker
                order by observations desc, ticker
                limit 20
            """,
            "sector_controversy_summary": """
                select sector, count(distinct ticker) as tickers, count(*) as records,
                       avg(controversy_score) as avg_controversy_score
                from raw_controversies
                group by sector
                order by avg_controversy_score desc
            """,
            "top_controversy_events": """
                select ticker, date, sector, controversy_score
                from raw_controversies
                order by controversy_score desc
                limit 10
            """,
            "target_class_balance": """
                select high_crash_risk, count(*) as rows
                from model_dataset
                group by high_crash_risk
                order by high_crash_risk desc
            """,
            "high_risk_names_by_sector": """
                select coalesce(sector, 'Unknown') as sector, count(*) as high_risk_names,
                       avg(crash_probability) as avg_crash_probability
                from stock_scores
                where risk_bucket = 'High'
                group by coalesce(sector, 'Unknown')
                order by high_risk_names desc, avg_crash_probability desc
            """,
            "model_feature_missingness": """
                select count(*) as rows,
                       sum(case when lagged_ncskew is null then 1 else 0 end) as missing_lagged_ncskew,
                       sum(case when downside_beta is null then 1 else 0 end) as missing_downside_beta,
                       sum(case when controversy_score is null then 1 else 0 end) as missing_controversy_score,
                       sum(case when market_cap is null then 1 else 0 end) as missing_market_cap
                from model_dataset
            """,
        }

        rows = []
        for name, query in queries.items():
            result = pd.read_sql_query(query, conn)
            rows.append(
                {
                    "query_name": name,
                    "query": clean_sql(query),
                    "result_json": result.to_json(orient="records"),
                    "row_count": len(result),
                }
            )
        return pd.DataFrame(rows)


def write_sql_summary_markdown(sql_summary: pd.DataFrame, path: Path) -> None:
    sections = ["# SQL Summary for Data Section", ""]
    for _, row in sql_summary.iterrows():
        result = pd.DataFrame(json.loads(row["result_json"]))
        sections.extend(
            [
                f"## {row['query_name']}",
                "",
                "```sql",
                row["query"],
                "```",
                "",
                "```csv",
                result.to_csv(index=False, lineterminator="\n").strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def clean_sql(query: str) -> str:
    return "\n".join(line.strip() for line in query.strip().splitlines())


def build_textual_analysis(paths: RawDataPaths, config: CrashRiskConfig | None = None) -> pd.DataFrame:
    return build_text_analysis_outputs(paths, config=config)["weekly"]


def build_text_ticker_summary(paths: RawDataPaths, config: CrashRiskConfig | None = None) -> pd.DataFrame:
    return build_text_analysis_outputs(paths, config=config)["ticker_summary"]


def build_text_analysis_outputs(paths: RawDataPaths, config: CrashRiskConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or CrashRiskConfig()
    prepared = prepare_text_records(paths)
    if prepared["status"] != "ok":
        status_row = pd.DataFrame([prepared["status_row"]])
        return {"weekly": status_row.copy(), "ticker_summary": status_row.copy()}

    text_df = prepared["text_df"]
    text_path = prepared["text_path"]
    scored_df = pd.DataFrame(text_df["text"].map(score_text).tolist(), index=text_df.index)
    text_df = pd.concat([text_df, scored_df], axis=1)
    text_df["week_end"] = text_df["date"].dt.to_period(config.week_rule).dt.end_time.dt.normalize()

    weekly = (
        text_df.groupby(["ticker", "week_end"], as_index=False)
        .agg(
            article_count=("text", "size"),
            text_sentiment_score=("text_sentiment_score", "mean"),
            negative_word_count=("negative_word_count", "sum"),
            positive_word_count=("positive_word_count", "sum"),
            controversy_keyword_count=("controversy_keyword_count", "sum"),
            total_token_count=("token_count", "sum"),
            negative_sentiment_intensity=("negative_sentiment_intensity", "mean"),
            controversy_keyword_density=("controversy_keyword_density", "mean"),
            news_pressure_component=("news_pressure_component", "mean"),
        )
        .sort_values(["ticker", "week_end"])
    )
    weekly["negative_esg_controversy_score_0_100"] = weekly.apply(score_negative_esg_controversy, axis=1)
    weekly["rolling_sentiment_13w"] = weekly.groupby("ticker", sort=False)["text_sentiment_score"].transform(
        lambda series: series.rolling(13, min_periods=1).mean()
    )
    weekly["score_band"] = weekly["negative_esg_controversy_score_0_100"].map(score_band)
    weekly.insert(0, "status", "ok")
    weekly.insert(1, "source_file", text_path.name)
    weekly = weekly.rename(columns={"week_end": "date"})

    ticker_summary = (
        weekly.sort_values(["ticker", "date"])
        .drop_duplicates("ticker", keep="last")
        .rename(columns={"date": "latest_text_date"})
        .sort_values(["negative_esg_controversy_score_0_100", "article_count", "ticker"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return {"weekly": weekly, "ticker_summary": ticker_summary}


def discover_text_path(paths: RawDataPaths) -> Path | None:
    raw_dir = Path(paths.controversies).parent
    for stem in TEXT_FILE_STEMS:
        for suffix in (".csv", ".xlsx", ".xls"):
            candidate = raw_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def prepare_text_records(paths: RawDataPaths) -> dict[str, object]:
    text_path = discover_text_path(paths)
    if text_path is None:
        status_row = {
            "status": "no_text_file",
            "note": "No controversy_text/news_text file found. Treat controversy_score as a vendor text-derived ESG signal until raw text is supplied.",
        }
        return {"status": "no_text_file", "status_row": status_row}

    text_df = harmonize_text_columns(read_tabular(text_path))
    missing = [column for column in ("ticker", "date") if column not in text_df.columns]
    text_columns = [column for column in TEXT_COLUMNS if column in text_df.columns]
    if missing or not text_columns:
        status_row = {
            "status": "invalid_text_file",
            "source_file": text_path.name,
            "note": f"Missing required text columns: {', '.join(missing or ['one of ' + ', '.join(TEXT_COLUMNS)])}.",
        }
        return {"status": "invalid_text_file", "status_row": status_row}

    text_df = text_df.copy()
    text_df["ticker"] = text_df["ticker"].astype(str).str.strip().str.upper()
    text_df["date"] = pd.to_datetime(text_df["date"], errors="coerce")
    text_df["text"] = text_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    source_column = "source" if "source" in text_df.columns else None
    if source_column:
        text_df["source"] = text_df["source"].astype(str).str.strip()
    text_df = text_df.dropna(subset=["ticker", "date"])
    text_df = text_df.loc[text_df["text"].str.strip().ne("")]
    if text_df.empty:
        status_row = {
            "status": "empty_text_file",
            "source_file": text_path.name,
            "note": "No usable text rows after parsing.",
        }
        return {"status": "empty_text_file", "status_row": status_row}

    columns = ["ticker", "date", "text"]
    if source_column:
        columns.append("source")
    return {"status": "ok", "text_df": text_df[columns].copy(), "text_path": text_path}


def harmonize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    canonical_lookup: dict[str, str] = {}
    for column in df.columns:
        canonical_lookup.setdefault(canonicalize_name(column), column)

    rename_map: dict[str, str] = {}
    for canonical, aliases in TEXT_COLUMN_ALIASES.items():
        for alias in aliases:
            source = canonical_lookup.get(alias)
            if source and source != canonical and canonical not in df.columns:
                rename_map[source] = canonical
                break
    return df.rename(columns=rename_map)


def canonicalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def score_text(text: str) -> dict[str, float | int]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return {
            "text_sentiment_score": 0.0,
            "negative_word_count": 0,
            "positive_word_count": 0,
            "controversy_keyword_count": 0,
            "token_count": 0,
            "negative_sentiment_intensity": 0.0,
            "controversy_keyword_density": 0.0,
            "news_pressure_component": 0.0,
        }
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    controversy = sum(token in CONTROVERSY_KEYWORDS for token in tokens)
    token_count = len(tokens)
    negative_sentiment_intensity = max(negative - positive, 0) / token_count
    controversy_keyword_density = controversy / token_count
    return {
        "text_sentiment_score": (positive - negative) / token_count,
        "negative_word_count": int(negative),
        "positive_word_count": int(positive),
        "controversy_keyword_count": int(controversy),
        "token_count": int(token_count),
        "negative_sentiment_intensity": float(negative_sentiment_intensity),
        "controversy_keyword_density": float(controversy_keyword_density),
        "news_pressure_component": 1.0,
    }


def score_negative_esg_controversy(row: pd.Series) -> float:
    negative_component = min(1.0, float(row.get("negative_sentiment_intensity", 0.0)) * 8.0)
    controversy_component = min(1.0, float(row.get("controversy_keyword_density", 0.0)) * 10.0)
    article_count = max(float(row.get("article_count", 0.0)), 0.0)
    news_pressure_component = min(1.0, np.log1p(article_count) / np.log(6.0))
    score = 100.0 * (
        0.5 * negative_component
        + 0.35 * controversy_component
        + 0.15 * news_pressure_component
    )
    return round(score, 2)


def score_band(score: float) -> str:
    if score >= 67:
        return "High"
    if score >= 34:
        return "Medium"
    return "Low"


def _write_placeholder_bigram_cloud(title: str, message: str, output_base: Path, colormap: str) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, fontsize=20, pad=20)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=18,
        color="#35524a" if colormap == "Greens" else "#6f2c2c",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_bigram_wordcloud(
    text_list: list[str],
    title: str,
    colormap: str,
    output_base: Path,
    stop_words: set[str],
) -> pd.DataFrame:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if not text_list:
        _write_placeholder_bigram_cloud(title, "No texts available to process for this sentiment bucket.", output_base, colormap)
        return pd.DataFrame(columns=["bigram", "tfidf_weight"])

    vectorizer = TfidfVectorizer(
        stop_words=sorted(stop_words),
        max_features=2000,
        ngram_range=(2, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        matrix = vectorizer.fit_transform(text_list)
    except ValueError as exc:
        _write_placeholder_bigram_cloud(title, f"Unable to build bigrams: {exc}", output_base, colormap)
        return pd.DataFrame(columns=["bigram", "tfidf_weight"])

    terms = vectorizer.get_feature_names_out()
    mean_weights = np.asarray(matrix.mean(axis=0)).ravel()
    freq_dict = dict(zip(terms, mean_weights))
    if not freq_dict:
        _write_placeholder_bigram_cloud(title, "No bigrams remained after filtering.", output_base, colormap)
        return pd.DataFrame(columns=["bigram", "tfidf_weight"])

    word_cloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100,
        colormap=colormap,
        collocations=False,
        prefer_horizontal=0.96,
        random_state=42,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.imshow(word_cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=20, pad=20)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return (
        pd.DataFrame({"bigram": terms, "tfidf_weight": mean_weights})
        .sort_values("tfidf_weight", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


def write_text_bigram_wordclouds(paths: RawDataPaths, figures_dir: Path, top_terms_path: Path | None = None) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_text_records(paths)
    if prepared["status"] != "ok":
        message = str(prepared["status_row"].get("note", "No text file supplied."))
        for output_base, title, colormap in (
            (figures_dir / "bullish_signals_bigrams", "Bullish Signals: Positive Sentiment TF-IDF Bigrams (Filtered)", "Greens"),
            (figures_dir / "bearish_signals_bigrams", "Bearish Signals: Negative Sentiment TF-IDF Bigrams (Filtered)", "Reds"),
        ):
            _write_placeholder_bigram_cloud(title, message, output_base, colormap)
        top_terms = pd.DataFrame(columns=["sentiment_bucket", "bigram", "tfidf_weight"])
    else:
        text_df = prepared["text_df"].copy()
        scored_df = pd.DataFrame(text_df["text"].map(score_text).tolist(), index=text_df.index)
        text_df = pd.concat([text_df, scored_df], axis=1)
        ticker_stop_words = set(text_df["ticker"].astype(str).str.lower().unique())
        stop_words = BIGRAM_STOP_WORDS.union(ticker_stop_words)

        positive_terms = generate_bigram_wordcloud(
            text_df.loc[text_df["text_sentiment_score"] > 0, "text"].dropna().tolist(),
            "Bullish Signals: Positive Sentiment TF-IDF Bigrams (Filtered)",
            "Greens",
            figures_dir / "bullish_signals_bigrams",
            stop_words,
        )
        positive_terms.insert(0, "sentiment_bucket", "bullish")

        negative_terms = generate_bigram_wordcloud(
            text_df.loc[text_df["text_sentiment_score"] < 0, "text"].dropna().tolist(),
            "Bearish Signals: Negative Sentiment TF-IDF Bigrams (Filtered)",
            "Reds",
            figures_dir / "bearish_signals_bigrams",
            stop_words,
        )
        negative_terms.insert(0, "sentiment_bucket", "bearish")
        term_frames = [frame for frame in (positive_terms, negative_terms) if not frame.empty]
        top_terms = (
            pd.concat(term_frames, ignore_index=True)
            if term_frames
            else pd.DataFrame(columns=["sentiment_bucket", "bigram", "tfidf_weight"])
        )

    if top_terms_path is not None:
        top_terms_path.parent.mkdir(parents=True, exist_ok=True)
        top_terms.to_csv(top_terms_path, index=False)

    primary_source = figures_dir / "bearish_signals_bigrams.svg"
    if not primary_source.exists() or primary_source.stat().st_size == 0:
        primary_source = figures_dir / "bullish_signals_bigrams.svg"
    if primary_source.exists():
        shutil.copyfile(primary_source, figures_dir / "text_word_cloud.svg")
    return top_terms


def write_text_word_cloud(paths: RawDataPaths, path: Path) -> None:
    write_text_bigram_wordclouds(
        paths=paths,
        figures_dir=path.parent,
        top_terms_path=path.parent.parent / "textual_bigram_terms.csv",
    )
    generated = path.parent / "text_word_cloud.svg"
    if generated != path and generated.exists():
        shutil.copyfile(generated, path)


def write_report_figures(
    feature_panel: pd.DataFrame,
    scores: pd.DataFrame,
    price_scenarios: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_comparison: pd.DataFrame,
    figures_dir: Path,
    feature_correlation_matrix: pd.DataFrame | None = None,
    calibration_curve: pd.DataFrame | None = None,
    lda_ticker_topics: pd.DataFrame | None = None,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    top_scores = scores.sort_values("crash_probability", ascending=False).head(10)
    write_bar_svg(
        figures_dir / "risk_probability_ranking.svg",
        "Crash-risk probability ranking",
        top_scores["ticker"].astype(str).tolist(),
        top_scores["crash_probability"].astype(float).tolist(),
        value_suffix="",
    )

    top_importance = feature_importance.sort_values("importance", ascending=False).head(10)
    write_bar_svg(
        figures_dir / "feature_importance.svg",
        "Top model feature importance",
        top_importance["feature"].astype(str).tolist(),
        top_importance["importance"].astype(float).tolist(),
        value_suffix="",
    )

    if {"sector", "controversy_score"}.issubset(feature_panel.columns):
        sector = (
            feature_panel.dropna(subset=["sector", "controversy_score"])
            .groupby("sector", as_index=False)["controversy_score"]
            .mean()
            .sort_values("controversy_score", ascending=False)
            .head(10)
        )
        write_bar_svg(
            figures_dir / "sector_controversy.svg",
            "Average controversy score by sector",
            sector["sector"].astype(str).tolist(),
            sector["controversy_score"].astype(float).tolist(),
            value_suffix="",
        )

        time_series = (
            feature_panel.dropna(subset=["date", "controversy_score"])
            .groupby("date", as_index=False)["controversy_score"]
            .mean()
            .sort_values("date")
        )
        write_line_svg(
            figures_dir / "controversy_over_time.svg",
            "Average controversy score over time",
            time_series["date"].astype(str).tolist(),
            time_series["controversy_score"].astype(float).tolist(),
        )

    delta = model_comparison.loc[model_comparison["model"].eq("full_minus_baseline")]
    if not delta.empty:
        row = delta.loc[delta["split"].eq("test")].head(1)
        if row.empty:
            row = delta.head(1)
        metrics = ["roc_auc", "precision_at_top_bucket", "crash_capture_at_top_bucket"]
        write_bar_svg(
            figures_dir / "esg_lift.svg",
            "ESG controversy model lift",
            metrics,
            [float(row.iloc[0].get(metric, 0.0)) for metric in metrics],
            value_suffix="",
        )

    if not price_scenarios.empty:
        high = price_scenarios.sort_values("crash_probability", ascending=False).head(1).iloc[0]
        write_bar_svg(
            figures_dir / "price_scenario_range.svg",
            f"{high['ticker']} 13-week price scenario",
            ["p05", "p50", "p95"],
            [float(high["price_p05"]), float(high["price_p50"]), float(high["price_p95"])],
            value_prefix="$",
        )

    if feature_correlation_matrix is not None and not feature_correlation_matrix.empty:
        write_feature_correlation_heatmap(feature_correlation_matrix, figures_dir / "feature_correlation_heatmap.png")
    if calibration_curve is not None and not calibration_curve.empty:
        write_probability_calibration_plot(calibration_curve, figures_dir / "probability_calibration.png")
    if lda_ticker_topics is not None and not lda_ticker_topics.empty:
        write_lda_topic_distribution(lda_ticker_topics, figures_dir / "lda_topic_distribution.png")


def write_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    value_prefix: str = "",
    value_suffix: str = "",
) -> None:
    width, row_h, left, right = 900, 32, 230, 90
    height = max(120, 72 + len(labels) * row_h)
    max_abs = max([abs(value) for value in values] or [1.0]) or 1.0
    zero_x = left if min(values or [0]) >= 0 else left + (width - left - right) / 2
    bar_area = width - left - right if min(values or [0]) >= 0 else (width - left - right) / 2
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#1a2025">{escape(title)}</text>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values, strict=False)):
        y = 62 + idx * row_h
        bar_w = max(2, abs(value) / max_abs * bar_area)
        x = zero_x if value >= 0 else zero_x - bar_w
        fill = "#1a7f68" if value >= 0 else "#c0332e"
        rows.extend(
            [
                f'<text x="24" y="{y + 16}" font-family="Arial" font-size="13" fill="#1a2025">{escape(label[:34])}</text>',
                f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="18" rx="4" fill="{fill}"/>',
                f'<text x="{width - 76}" y="{y + 15}" font-family="Arial" font-size="12" fill="#5c6770">{escape(format_chart_value(value, value_prefix, value_suffix))}</text>',
            ]
        )
    rows.append("</svg>")
    path.write_text("\n".join(rows), encoding="utf-8")


def write_line_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height, pad = 900, 320, 48
    if not values:
        path.write_text("", encoding="utf-8")
        return
    ymin, ymax = min(values), max(values)
    span = ymax - ymin or 1.0
    points = []
    for idx, value in enumerate(values):
        x = pad + (idx / max(1, len(values) - 1)) * (width - pad * 2)
        y = height - pad - ((value - ymin) / span) * (height - pad * 2)
        points.append(f"{x:.1f},{y:.1f}")
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#1a2025">{escape(title)}</text>',
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#0d6b62" stroke-width="3"/>',
        f'<text x="{pad}" y="{height - 14}" font-family="Arial" font-size="12" fill="#5c6770">{escape(labels[0])}</text>',
        f'<text x="{width - 150}" y="{height - 14}" font-family="Arial" font-size="12" fill="#5c6770">{escape(labels[-1])}</text>',
        f'<text x="{pad}" y="{pad}" font-family="Arial" font-size="12" fill="#5c6770">max {ymax:.3f}</text>',
        f'<text x="{pad}" y="{height - pad}" font-family="Arial" font-size="12" fill="#5c6770">min {ymin:.3f}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def write_feature_correlation_heatmap(correlation_matrix: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    corr = correlation_matrix.copy()
    if "feature" in corr.columns:
        corr = corr.set_index("feature")
    corr = corr.apply(pd.to_numeric, errors="coerce")
    if corr.empty:
        write_placeholder_png(path, "Feature Correlation Heatmap", "No feature correlations available.")
        return

    labels = list(corr.index.astype(str))
    values = corr.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_price_time_series(feature_panel: pd.DataFrame, scores: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if feature_panel.empty or "adj_close" not in feature_panel.columns or scores.empty:
        write_placeholder_png(path, "Representative Price Time Series", "No price history available.")
        return

    high = (
        scores.loc[scores["risk_bucket"].eq("High")]
        .sort_values("crash_probability", ascending=False)
        .head(2)["ticker"]
        .tolist()
    )
    low = (
        scores.loc[scores["risk_bucket"].eq("Low")]
        .sort_values("crash_probability", ascending=True)
        .head(2)["ticker"]
        .tolist()
    )
    selected = list(dict.fromkeys([*high, *low]))
    if len(selected) < 4:
        fallback = scores.sort_values("crash_probability", ascending=False)["ticker"].tolist()
        selected = list(dict.fromkeys([*selected, *fallback]))[:4]

    panel = feature_panel.loc[feature_panel["ticker"].isin(selected)].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "adj_close"]).sort_values(["ticker", "date"])
    if panel.empty:
        write_placeholder_png(path, "Representative Price Time Series", "No selected ticker prices available.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker, group in panel.groupby("ticker", sort=False):
        risk = scores.loc[scores["ticker"].eq(ticker), "risk_bucket"]
        label = f"{ticker} ({risk.iloc[0] if not risk.empty else 'n/a'})"
        ax.plot(group["date"], group["adj_close"], linewidth=1.8, label=label)
    ax.set_title("Representative Adjusted Close Price Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_probability_calibration_plot(calibration_curve: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    required = {"mean_predicted_probability", "observed_crash_rate", "n_rows"}
    if calibration_curve.empty or not required.issubset(calibration_curve.columns):
        write_placeholder_png(path, "Probability Calibration", "No calibration data available.")
        return
    curve = calibration_curve.loc[pd.to_numeric(calibration_curve["n_rows"], errors="coerce").fillna(0) > 0].copy()
    if curve.empty:
        write_placeholder_png(path, "Probability Calibration", "No populated calibration bins.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#7f8c8d", linestyle=":", label="Perfect calibration")
    ax.plot(
        curve["mean_predicted_probability"],
        curve["observed_crash_rate"],
        marker="o",
        color="#0d6b62",
        linewidth=2,
        label="Test bins",
    )
    for _, row in curve.iterrows():
        ax.text(
            row["mean_predicted_probability"],
            row["observed_crash_rate"],
            f"n={int(row['n_rows'])}",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed crash rate")
    ax.set_title("Predicted Probability Calibration")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_lda_topic_distribution(lda_ticker_topics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lda_ticker_topics.empty or "dominant_topic" not in lda_ticker_topics.columns or not lda_ticker_topics["status"].eq("ok").any():
        write_placeholder_png(path, "LDA Topic Distribution", "No LDA topics available.")
        return
    counts = (
        lda_ticker_topics.loc[lda_ticker_topics["status"].eq("ok"), "dominant_topic"]
        .value_counts()
        .sort_index()
    )
    if counts.empty:
        write_placeholder_png(path, "LDA Topic Distribution", "No dominant topic assignments available.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#0d6b62", edgecolor="white")
    ax.set_title("Dominant LDA Topic by Ticker")
    ax.set_xlabel("Dominant topic")
    ax.set_ylabel("Ticker count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_placeholder_png(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_placeholder_svg(path: Path, title: str, message: str) -> None:
    width, height = 900, 220
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="38" font-family="Arial" font-size="20" font-weight="700" fill="#1a2025">{escape(title)}</text>',
        f'<text x="24" y="86" font-family="Arial" font-size="15" fill="#5c6770">{escape(message)}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def format_chart_value(value: float, prefix: str, suffix: str) -> str:
    return f"{prefix}{value:.3f}{suffix}"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_report_outline(data_summary: pd.DataFrame, model_comparison: pd.DataFrame, outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    lookup = {
        (row["section"], row["metric"]): row["value"]
        for _, row in data_summary.iterrows()
    }
    lift = model_comparison.loc[
        model_comparison["model"].eq("full_minus_baseline") & model_comparison["split"].eq("test")
    ]
    lift_text = "Pending real-data ESG lift result."
    if not lift.empty:
        row = lift.iloc[0]
        lift_text = (
            f"Test lift: ROC-AUC {row['roc_auc']:.3f}, "
            f"Precision@Top Bucket {row['precision_at_top_bucket']:.3f}, "
            f"Crash Capture {row['crash_capture_at_top_bucket']:.3f}."
        )

    body = f"""# FDS Project Report Outline

## Title
ESG Controversy Signals for Equity Crash-Risk Monitoring

## 1. Data Summary and Visualisations
This is an equity risk-management application, not a pure return forecast. The current run contains {lookup.get(('feature_panel', 'rows'), 'N/A')} weekly feature rows, {lookup.get(('feature_panel', 'ticker_count'), 'N/A')} tickers, and {lookup.get(('model_dataset', 'rows'), 'N/A')} labeled model rows.

Use `outputs/data_summary.csv`, `outputs/cleaning_log.csv`, `outputs/sql_summary.md`, and `outputs/figures/` for the report tables and visuals.

## 2. Textual Analysis
Use `outputs/textual_analysis.csv` and `outputs/figures/text_word_cloud.svg`. If no `controversy_text.csv` or `news_text.csv` file exists, the report should state that `controversy_score` is treated as a vendor text-derived ESG signal and that direct headline sentiment is a limitation.

## 3. Machine Learning
Target: `high_crash_risk = 1` for stocks in the top 20% of future 13-week NCSKEW. Use chronological train/validation/test splits and report ROC-AUC, Precision@Top Bucket, and Crash Capture. {lift_text}

## 4. Business Analysis
Use `outputs/business_analysis.csv` and `outputs/business_portfolio_returns.csv` to discuss the economic value of excluding or reviewing High crash-risk names. Frame this as a weekly forward risk overlay for a 1 billion dollar fund and compare estimated benefit with a 4-person implementation team.

## 5. Viva Slide Outline
1. Problem and research question
2. Data, cleaning, and SQL evidence
3. ESG controversy/text signal
4. ML target, models, and validation
5. Business value and fund overlay
6. Dashboard demo and conclusion

## Appendix
Submit Python modules rather than notebooks. Cite `crashrisk/pipeline.py`, `crashrisk/features/`, `crashrisk/models/`, `crashrisk/analysis/`, and `crashrisk/api/main.py`.
"""
    (outputs_dir / "fds_report_outline.md").write_text(body, encoding="utf-8")
