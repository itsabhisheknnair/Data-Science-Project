from __future__ import annotations

import html
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from crashrisk.config import CrashRiskConfig, RawDataPaths
from crashrisk.data.loaders import (
    BENCHMARK_COLUMNS,
    CONTROVERSY_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    PRICE_COLUMNS,
    load_raw_data,
    read_tabular,
)


TEXT_FILE_STEMS = ("controversy_text", "news_text", "textual_data")
TEXT_COLUMNS = ("headline", "title", "description", "body", "text", "summary")
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
) -> dict[str, pd.DataFrame]:
    """Build FDS report-ready outputs without changing the model signal."""
    config = config or CrashRiskConfig()
    paths = RawDataPaths.from_mapping(raw_paths)
    raw = load_raw_data(paths, config=config)
    outputs_dir = Path(outputs_dir)

    data_summary, cleaning_log = build_data_summary(paths, raw, feature_panel, dataset, scores, config)
    sql_summary = build_sql_summary(raw, feature_panel, dataset, scores)
    textual_analysis = build_textual_analysis(paths, config=config)
    write_sql_summary_markdown(sql_summary, outputs_dir / "sql_summary.md")
    write_report_figures(
        feature_panel=feature_panel,
        scores=scores,
        price_scenarios=price_scenarios,
        feature_importance=feature_importance,
        model_comparison=model_comparison,
        figures_dir=outputs_dir / "figures",
    )
    write_text_word_cloud(paths, outputs_dir / "figures" / "text_word_cloud.svg")
    write_report_outline(
        data_summary=data_summary,
        model_comparison=model_comparison,
        outputs_dir=outputs_dir,
    )

    return {
        "data_summary": data_summary,
        "cleaning_log": cleaning_log,
        "sql_summary": sql_summary,
        "textual_analysis": textual_analysis,
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
        for column in required_columns:
            if column in original:
                values = original[column]
                blank_required += int(values.isna().sum())
                if values.dtype == object:
                    blank_required += int(values.astype(str).str.strip().eq("").sum())
        duplicates = int(original.duplicated().sum())
        dropped = max(0, len(original) - len(loaded))
        add_cleaning(cleaning_rows, name, "missing_required_values", blank_required, "Blank or NA values in required columns before loading.")
        add_cleaning(cleaning_rows, name, "duplicate_rows", duplicates, "Fully duplicated raw rows.")
        add_cleaning(cleaning_rows, name, "rows_removed_or_invalid", dropped, "Rows not retained by the validated loader.")

    prices_original = read_tabular(paths.prices)
    if "adj_close" in prices_original:
        prices_numeric = pd.to_numeric(prices_original["adj_close"], errors="coerce")
        add_cleaning(
            cleaning_rows,
            "prices",
            "zero_or_negative_adj_close",
            int((prices_numeric <= 0).sum()),
            "Rows with non-positive adjusted prices; these should be investigated or removed.",
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


def add_cleaning(rows: list[dict[str, object]], dataset: str, check: str, value: object, detail: str) -> None:
    rows.append({"dataset": dataset, "check": check, "value": value, "detail": detail})


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
        if "sector" in feature_panel:
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
    config = config or CrashRiskConfig()
    text_path = discover_text_path(paths)
    if text_path is None:
        return pd.DataFrame(
            [
                {
                    "status": "no_text_file",
                    "note": "No controversy_text/news_text file found. Treat controversy_score as a vendor text-derived ESG signal until raw text is supplied.",
                }
            ]
        )

    text_df = read_tabular(text_path)
    missing = [column for column in ("ticker", "date") if column not in text_df.columns]
    text_columns = [column for column in TEXT_COLUMNS if column in text_df.columns]
    if missing or not text_columns:
        return pd.DataFrame(
            [
                {
                    "status": "invalid_text_file",
                    "source_file": text_path.name,
                    "note": f"Missing required text columns: {', '.join(missing or ['one of ' + ', '.join(TEXT_COLUMNS)])}.",
                }
            ]
        )

    text_df = text_df.copy()
    text_df["ticker"] = text_df["ticker"].astype(str).str.strip().str.upper()
    text_df["date"] = pd.to_datetime(text_df["date"], errors="coerce")
    text_df["text"] = text_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    text_df = text_df.dropna(subset=["ticker", "date"])
    text_df = text_df.loc[text_df["text"].str.strip().ne("")]
    if text_df.empty:
        return pd.DataFrame(
            [{"status": "empty_text_file", "source_file": text_path.name, "note": "No usable text rows after parsing."}]
        )

    scored = text_df["text"].map(score_text)
    scored_df = pd.DataFrame(scored.tolist(), index=text_df.index)
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
        )
        .sort_values(["ticker", "week_end"])
    )
    weekly["rolling_sentiment_13w"] = weekly.groupby("ticker", sort=False)["text_sentiment_score"].transform(
        lambda series: series.rolling(13, min_periods=1).mean()
    )
    weekly.insert(0, "status", "ok")
    weekly.insert(1, "source_file", text_path.name)
    weekly = weekly.rename(columns={"week_end": "date"})
    return weekly


def discover_text_path(paths: RawDataPaths) -> Path | None:
    raw_dir = Path(paths.controversies).parent
    for stem in TEXT_FILE_STEMS:
        for suffix in (".csv", ".xlsx", ".xls"):
            candidate = raw_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def score_text(text: str) -> dict[str, float | int]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return {
            "text_sentiment_score": 0.0,
            "negative_word_count": 0,
            "positive_word_count": 0,
            "controversy_keyword_count": 0,
        }
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    controversy = sum(token in CONTROVERSY_KEYWORDS for token in tokens)
    return {
        "text_sentiment_score": (positive - negative) / len(tokens),
        "negative_word_count": int(negative),
        "positive_word_count": int(positive),
        "controversy_keyword_count": int(controversy),
    }


def write_text_word_cloud(paths: RawDataPaths, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_path = discover_text_path(paths)
    if text_path is None:
        write_placeholder_svg(path, "Text word cloud", "No news_text or controversy_text file supplied.")
        return

    text_df = read_tabular(text_path)
    text_columns = [column for column in TEXT_COLUMNS if column in text_df.columns]
    if not text_columns:
        write_placeholder_svg(path, "Text word cloud", "Text file has no headline, description, body, text, or summary column.")
        return

    text = " ".join(text_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).tolist())
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z]+", text.lower())
        if len(token) > 3 and token not in STOP_WORDS
    ]
    counts = Counter(tokens).most_common(28)
    if not counts:
        write_placeholder_svg(path, "Text word cloud", "No usable words after parsing the text file.")
        return

    max_count = counts[0][1] or 1
    width, height = 900, 360
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#1a2025">Text word cloud</text>',
    ]
    x, y = 34, 86
    for word, count in counts:
        size = 14 + int((count / max_count) * 24)
        word_width = max(70, len(word) * size * 0.6)
        if x + word_width > width - 34:
            x = 34
            y += 48
        fill = "#c0332e" if word in NEGATIVE_WORDS or word in CONTROVERSY_KEYWORDS else "#0d6b62"
        rows.append(
            f'<text x="{x:.0f}" y="{y:.0f}" font-family="Arial" font-size="{size}" font-weight="700" fill="{fill}">{escape(word)}</text>'
        )
        x += word_width + 24
    rows.append("</svg>")
    path.write_text("\n".join(rows), encoding="utf-8")


def write_report_figures(
    feature_panel: pd.DataFrame,
    scores: pd.DataFrame,
    price_scenarios: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_comparison: pd.DataFrame,
    figures_dir: Path,
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
Use `outputs/business_analysis.csv` to discuss the economic value of excluding or reviewing High crash-risk names. Frame this as a risk overlay for a 1 billion dollar fund and compare estimated benefit with a 4-person implementation team.

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
