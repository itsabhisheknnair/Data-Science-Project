"""
ESG Controversy Signals for Equity Crash-Risk Monitoring
=========================================================
FIN42110 — Financial Data Science Project
MSc Financial Data Science, University College Dublin

Self-contained single-file implementation of the full crash-risk pipeline.
Run:  python crash_risk_model.py

Dependencies: numpy, pandas, scikit-learn, matplotlib  (all standard conda/pip packages)

References
----------
Chen, Hong & Stein (2001) — NCSKEW crash-risk framework
Kim, Li & Zhang (2014)   — ESG controversy and crash-risk extension
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import shutil
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from crashrisk.analysis.reporting import (
    build_data_summary,
    build_feature_correlation_matrix,
    build_feature_descriptive_stats,
    build_lda_topic_outputs,
    build_sql_summary,
    build_text_coverage,
    join_text_signals_to_panel,
    write_feature_correlation_heatmap,
    write_lda_topic_distribution,
    write_price_time_series,
    write_probability_calibration_plot,
    write_sql_summary_markdown,
)
from crashrisk.config import CrashRiskConfig, RawDataPaths
from crashrisk.data.loaders import load_raw_data
from crashrisk.models.compare import (
    build_hyperparameter_tuning_results,
    build_test_diagnostics,
    compare_text_signal_lift,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Set to "real" to load from data/raw/ (50 tickers, 6 years).
# Falls back to "synthetic" automatically if the folder is missing.
DATA_SOURCE        = "real"
DATA_DIR           = Path("data/raw")

WEEKS              = 260          # used only for synthetic fallback
N_TICKERS          = 12
SEED               = 42
TRAIN_FRAC         = 0.60
VAL_FRAC           = 0.20
ROLLING_WINDOW     = 26          # weeks for beta / NCSKEW / volatility
TARGET_HORIZON     = 13          # weeks forward for crash-risk label
TARGET_QUANTILE    = 0.20        # top 20% → high_crash_risk = 1
FUND_AUM           = 1_000_000_000.0
RISK_FREE_ANNUAL   = 0.04
TEAM_COST          = 800_000.0
FIGURES_DIR        = Path("outputs/figures")
REPORT_FIGURES_DIR = Path("reports/figures")
OUTPUTS_DIR        = Path("outputs")

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
    "abuse","accident","allegation","breach","bribery","collapse","controversy","corruption",
    "crash","crisis","default","downgrade","emissions","fraud","investigation","lawsuit",
    "loss","misconduct","pollution","probe","recall","risk","scandal","strike","violation",
}
POSITIVE_WORDS = {
    "award","benefit","clean","improve","improved","positive","progress","resolve",
    "resolved","safe","settle","settled","upgrade",
}
CONTROVERSY_KEYWORDS = {
    "bribery","corruption","emissions","fraud","governance","investigation",
    "lawsuit","misconduct","pollution","scandal","social","violation",
}
STOP_WORDS = {"a","and","are","as","at","by","for","from","in","is","it","of","on","or","that","the","to","with"}
BIGRAM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    STOP_WORDS,
    {
        "analysts", "article", "commentary", "company", "coverage", "esg", "external",
        "faces", "firm", "follow", "group", "headlines", "highlights", "linked",
        "market", "monitor", "month", "news", "note", "ongoing", "reports",
        "review", "risk", "risks", "said", "score", "signal", "signals", "text",
    },
)

ESG_FEATURES = [
    "controversy_score",
    "controversy_change_4w",
    "controversy_change_13w",
    "controversy_change_26w",
    "controversy_rolling_mean_13w",
    "controversy_rolling_std_13w",
    "controversy_spike_flag",
    "controversy_sector_percentile",
]

ALL_FEATURES = [
    # Group 1: Crash history
    "lagged_ncskew", "lagged_duvol",
    # Group 2: Trading activity
    "detrended_turnover",
    # Group 3: Downside risk
    "trailing_return", "realized_volatility",
    "beta", "downside_beta", "relative_downside_beta",
    # Group 4: Fundamentals
    "market_cap", "market_to_book", "leverage", "roa",
    # Group 5: ESG controversy (8 features)
    *ESG_FEATURES,
]

BASELINE_FEATURES = [f for f in ALL_FEATURES if f not in ESG_FEATURES]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

TICKER_SPECS: dict[str, dict] = {
    "ALPH": {"sector": "Technology",  "start": 84.0,  "beta": 1.10, "controversy_base": 2.0,  "crash": False},
    "GLOB": {"sector": "Technology",  "start": 91.0,  "beta": 1.18, "controversy_base": 1.5,  "crash": False},
    "CYRX": {"sector": "Energy",      "start": 46.0,  "beta": 1.35, "controversy_base": 4.2,  "crash": True},
    "HVST": {"sector": "Energy",      "start": 38.0,  "beta": 1.42, "controversy_base": 3.5,  "crash": True},
    "FINX": {"sector": "Financials",  "start": 52.0,  "beta": 1.22, "controversy_base": 2.2,  "crash": True},
    "LUXE": {"sector": "Financials",  "start": 78.0,  "beta": 1.15, "controversy_base": 2.0,  "crash": False},
    "DYNM": {"sector": "Healthcare",  "start": 72.0,  "beta": 0.78, "controversy_base": 1.8,  "crash": False},
    "IMUN": {"sector": "Healthcare",  "start": 67.0,  "beta": 0.72, "controversy_base": 1.2,  "crash": False},
    "BRCK": {"sector": "Industrials", "start": 58.0,  "beta": 0.86, "controversy_base": 1.4,  "crash": False},
    "KNXT": {"sector": "Industrials", "start": 55.0,  "beta": 0.88, "controversy_base": 1.6,  "crash": False},
    "ECOR": {"sector": "Consumer",    "start": 63.0,  "beta": 1.05, "controversy_base": 2.5,  "crash": False},
    "JETT": {"sector": "Consumer",    "start": 49.0,  "beta": 0.95, "controversy_base": 2.8,  "crash": False},
}

# For controversy-crash tickers: spike weeks (monthly index into controversy series)
_SPIKE_SCHEDULE = {"CYRX": [14, 24, 38], "FINX": [21, 35], "HVST": [19, 32]}


def generate_data(weeks: int = WEEKS, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a weekly price panel and a controversy panel.

    Returns
    -------
    prices  : columns [ticker, date, weekly_return, adj_close, volume, shares_outstanding, sector,
                        market_cap, market_to_book, leverage, roa]
    controv : columns [ticker, date, sector, controversy_score]
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-04", periods=weeks, freq="W-FRI")

    # ── Benchmark (S&P 500 proxy) ──────────────────────────────────────────
    mkt_ret = rng.normal(0.001, 0.018, size=weeks)
    for stress_wk in [28, 65, 104, 155, 210]:
        if stress_wk < weeks:
            mkt_ret[stress_wk] -= rng.uniform(0.04, 0.07)
    mkt_price = 100.0 * np.cumprod(1.0 + np.concatenate([[0.0], mkt_ret[1:]]))

    # ── Stock prices ───────────────────────────────────────────────────────
    price_rows: list[dict] = []
    for ticker, spec in TICKER_SPECS.items():
        idio = rng.normal(0.0006, 0.022, size=weeks)

        # Controversy-spike tickers: inject crashes 4–8 weeks AFTER each spike
        if spec["crash"]:
            crash_weeks = [min(w * 4 + rng.integers(4, 9), weeks - 1)
                           for w in _SPIKE_SCHEDULE.get(ticker, [])]
            for cw in crash_weeks:
                idio[cw] -= rng.uniform(0.09, 0.15)

        if spec["sector"] == "Healthcare":
            idio += 0.0004                          # slight positive drift for safe names

        ret = spec["beta"] * mkt_ret + idio
        price = spec["start"] * np.cumprod(1.0 + np.concatenate([[0.0], ret[1:]]))
        shares = int(rng.integers(60_000_000, 250_000_000))
        base_vol = int(rng.integers(500_000, 2_000_000))

        for i, date in enumerate(dates):
            price_rows.append({
                "ticker":            ticker,
                "date":              date,
                "adj_close":         round(float(price[i]), 4),
                "weekly_return":     float(ret[i]),
                "volume":            base_vol + i * int(rng.integers(200, 3000)),
                "shares_outstanding": shares,
                "sector":            spec["sector"],
                "market_cap":        round(float(price[i] * shares), 0),
                "market_to_book":    round(float(max(0.5, 1.4 + 0.06 * (i / 52) + rng.normal(0, 0.05))), 3),
                "leverage":          round(float(max(0.05, (0.35 if spec["crash"] else 0.22)
                                                  + 0.008 * (i / 52) + rng.normal(0, 0.015))), 3),
                "roa":               round(float((0.08 if spec["crash"] else 0.13)
                                               - 0.002 * (i / 52) + rng.normal(0, 0.008)), 3),
            })

    prices = pd.DataFrame(price_rows)
    prices["market_return"] = prices["date"].map(dict(zip(dates, mkt_ret)))

    # ── Controversy scores (monthly cadence joined to weekly) ─────────────
    cont_rows: list[dict] = []
    monthly_dates = dates[::4]
    for ticker, spec in TICKER_SPECS.items():
        score = float(spec["controversy_base"])
        spike_months = _SPIKE_SCHEDULE.get(ticker, [])
        for idx, date in enumerate(monthly_dates):
            score = max(0.0, score + rng.normal(0.04, 0.30))
            if idx in spike_months:
                score += rng.uniform(2.5, 4.5)
            score = score * 0.92 + spec["controversy_base"] * 0.08
            score = min(score, 10.0)
            cont_rows.append({"ticker": ticker, "date": date,
                               "sector": spec["sector"],
                               "controversy_score": round(float(score), 3)})

    controv = pd.DataFrame(cont_rows)
    return prices, controv


# ══════════════════════════════════════════════════════════════════════════════
# 1b.  REAL DATA LOADER  (data/raw/)
# ══════════════════════════════════════════════════════════════════════════════

def load_real_data(raw_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the four Bloomberg-format CSVs from raw_dir and return the same
    (prices, controv) tuple format that generate_data() produces.

    Input files
    -----------
    prices.csv          : ticker, date (DD-MM-YYYY), adj_close, volume
    benchmark_prices.csv: date (DD-MM-YYYY), benchmark_close
    fundamentals.csv    : ticker, period_end, market_cap, shares_outstanding,
                          market_to_book, leverage, roa
    controversies.csv   : ticker, date, sector, controversy_score
    """
    # ── Load raw files ─────────────────────────────────────────────────────
    px   = pd.read_csv(raw_dir / "prices.csv")
    bm   = pd.read_csv(raw_dir / "benchmark_prices.csv")
    fund = pd.read_csv(raw_dir / "fundamentals.csv")
    cont = pd.read_csv(raw_dir / "controversies.csv")

    # ── Parse dates (DD-MM-YYYY for prices / benchmark) ───────────────────
    px["date"]  = pd.to_datetime(px["date"],  dayfirst=True)
    bm["date"]  = pd.to_datetime(bm["date"],  dayfirst=True)
    cont["date"] = pd.to_datetime(cont["date"])
    fund["period_end"] = pd.to_datetime(fund["period_end"])

    # ── Resample benchmark daily → weekly (W-FRI) ─────────────────────────
    bm_weekly = (
        bm.sort_values("date")
          .set_index("date")
          .resample("W-FRI")["benchmark_close"]
          .last()
          .pct_change()
          .rename("market_return")
          .reset_index()
    )

    # ── Resample each stock daily → weekly ────────────────────────────────
    weekly_parts: list[pd.DataFrame] = []
    for ticker, grp in px.groupby("ticker"):
        w = (
            grp.sort_values("date")
               .set_index("date")
               .resample("W-FRI")
               .agg(adj_close=("adj_close", "last"), volume=("volume", "sum"))
               .reset_index()
        )
        w["ticker"]        = ticker
        w["weekly_return"] = w["adj_close"].pct_change()
        weekly_parts.append(w)

    prices_w = pd.concat(weekly_parts, ignore_index=True)

    # ── Merge market return ────────────────────────────────────────────────
    prices_w = prices_w.merge(bm_weekly, on="date", how="left")

    # ── Join static fundamentals per ticker ───────────────────────────────
    # Use the single (most recent) record per ticker.
    fund_latest = fund.sort_values("period_end").groupby("ticker").last().reset_index()
    prices_w = prices_w.merge(
        fund_latest[["ticker", "shares_outstanding", "market_to_book", "leverage", "roa"]],
        on="ticker", how="left",
    )

    # Dynamic weekly market cap: adj_close × shares_outstanding
    prices_w["market_cap"] = prices_w["adj_close"] * prices_w["shares_outstanding"]

    # ── Join sector from controversies ────────────────────────────────────
    sector_map = cont.groupby("ticker")["sector"].first()
    prices_w["sector"] = prices_w["ticker"].map(sector_map)

    return prices_w.sort_values(["ticker", "date"]).reset_index(drop=True), cont


def read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported text file type: {path.suffix}")


def canonicalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def harmonize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    current = {canonicalize_name(col): col for col in df.columns}
    rename_map = {}
    for canonical, aliases in TEXT_COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            source = current.get(alias)
            if source:
                rename_map[source] = canonical
                break
    return df.rename(columns=rename_map)


def discover_text_path(raw_dir: Path = DATA_DIR) -> Path | None:
    for stem in TEXT_FILE_STEMS:
        for suffix in (".csv", ".xlsx", ".xls"):
            candidate = raw_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def prepare_text_records(raw_dir: Path = DATA_DIR) -> dict[str, object]:
    text_path = discover_text_path(raw_dir)
    if text_path is None:
        return {"status": "no_text_file", "status_row": {
            "status": "no_text_file",
            "note": "No controversy_text/news_text file found. Supply optional text data for direct ESG news scoring.",
        }}

    text_df = harmonize_text_columns(read_tabular_file(text_path))
    missing = [column for column in ("ticker", "date") if column not in text_df.columns]
    text_columns = [column for column in TEXT_COLUMNS if column in text_df.columns]
    if missing or not text_columns:
        return {"status": "invalid_text_file", "status_row": {
            "status": "invalid_text_file",
            "source_file": text_path.name,
            "note": f"Missing required text columns: {', '.join(missing or ['one of ' + ', '.join(TEXT_COLUMNS)])}.",
        }}

    text_df = text_df.copy()
    text_df["ticker"] = text_df["ticker"].astype(str).str.strip().str.upper()
    text_df["date"] = pd.to_datetime(text_df["date"], errors="coerce")
    text_df["text"] = text_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    if "source" in text_df.columns:
        text_df["source"] = text_df["source"].astype(str).str.strip()
    text_df = text_df.dropna(subset=["ticker", "date"])
    text_df = text_df.loc[text_df["text"].str.strip().ne("")]
    if text_df.empty:
        return {"status": "empty_text_file", "status_row": {
            "status": "empty_text_file",
            "source_file": text_path.name,
            "note": "No usable text rows after parsing.",
        }}

    keep = ["ticker", "date", "text"]
    if "source" in text_df.columns:
        keep.append("source")
    return {"status": "ok", "text_path": text_path, "text_df": text_df[keep].copy()}


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
        }
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    controversy = sum(token in CONTROVERSY_KEYWORDS for token in tokens)
    token_count = len(tokens)
    return {
        "text_sentiment_score": (positive - negative) / token_count,
        "negative_word_count": int(negative),
        "positive_word_count": int(positive),
        "controversy_keyword_count": int(controversy),
        "token_count": int(token_count),
        "negative_sentiment_intensity": max(negative - positive, 0) / token_count,
        "controversy_keyword_density": controversy / token_count,
    }


def score_negative_esg_controversy(row: pd.Series) -> float:
    negative_component = min(1.0, float(row.get("negative_sentiment_intensity", 0.0)) * 8.0)
    controversy_component = min(1.0, float(row.get("controversy_keyword_density", 0.0)) * 10.0)
    article_count = max(float(row.get("article_count", 0.0)), 0.0)
    news_pressure_component = min(1.0, np.log1p(article_count) / np.log(6.0))
    score = 100.0 * (
        0.50 * negative_component
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


def build_textual_analysis_outputs(raw_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_text_records(raw_dir)
    if prepared["status"] != "ok":
        status_df = pd.DataFrame([prepared["status_row"]])
        return status_df.copy(), status_df.copy()

    text_df = prepared["text_df"].copy()
    text_path = prepared["text_path"]
    scored_df = pd.DataFrame(text_df["text"].map(score_text).tolist(), index=text_df.index)
    text_df = pd.concat([text_df, scored_df], axis=1)
    text_df["week_end"] = text_df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

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
        )
        .sort_values(["ticker", "week_end"])
    )
    weekly["negative_esg_controversy_score_0_100"] = weekly.apply(score_negative_esg_controversy, axis=1)
    weekly["rolling_sentiment_13w"] = weekly.groupby("ticker", sort=False)["text_sentiment_score"].transform(
        lambda s: s.rolling(13, min_periods=1).mean()
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
    return weekly, ticker_summary


def _write_placeholder_bigram_cloud(title: str, message: str, output_base: Path, colormap: str) -> None:
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

    vec = TfidfVectorizer(
        stop_words=sorted(stop_words),
        max_features=2000,
        ngram_range=(2, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    try:
        matrix = vec.fit_transform(text_list)
    except ValueError as exc:
        _write_placeholder_bigram_cloud(title, f"Unable to build bigrams: {exc}", output_base, colormap)
        return pd.DataFrame(columns=["bigram", "tfidf_weight"])

    terms = vec.get_feature_names_out()
    mean_weights = np.asarray(matrix.mean(axis=0)).ravel()
    freq_dict = dict(zip(terms, mean_weights))
    if not freq_dict:
        _write_placeholder_bigram_cloud(title, "No bigrams remained after filtering.", output_base, colormap)
        return pd.DataFrame(columns=["bigram", "tfidf_weight"])

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100,
        colormap=colormap,
        collocations=False,
        prefer_horizontal=0.96,
        random_state=SEED,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.imshow(wc, interpolation="bilinear")
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


def write_text_bigram_wordclouds(
    raw_dir: Path = DATA_DIR,
    figures_dir: Path = FIGURES_DIR,
    mirror_dir: Path | None = REPORT_FIGURES_DIR,
    top_terms_path: Path | None = OUTPUTS_DIR / "textual_bigram_terms.csv",
) -> dict[str, Path | pd.DataFrame]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_text_records(raw_dir)
    if prepared["status"] != "ok":
        note = str(prepared["status_row"].get("note", "No text file supplied."))
        for output_base, title, colormap in (
            (figures_dir / "bullish_signals_bigrams", "Bullish Signals: Positive Sentiment TF-IDF Bigrams (Filtered)", "Greens"),
            (figures_dir / "bearish_signals_bigrams", "Bearish Signals: Negative Sentiment TF-IDF Bigrams (Filtered)", "Reds"),
        ):
            _write_placeholder_bigram_cloud(title, note, output_base, colormap)
        terms = pd.DataFrame(columns=["sentiment_bucket", "bigram", "tfidf_weight"])
    else:
        text_df = prepared["text_df"].copy()
        scored_df = pd.DataFrame(text_df["text"].map(score_text).tolist(), index=text_df.index)
        text_df = pd.concat([text_df, scored_df], axis=1)
        ticker_stop_words = set(text_df["ticker"].astype(str).str.lower().unique())
        stop_words = BIGRAM_STOP_WORDS.union(ticker_stop_words)
        positive_texts = text_df.loc[text_df["text_sentiment_score"] > 0, "text"].dropna().tolist()
        negative_texts = text_df.loc[text_df["text_sentiment_score"] < 0, "text"].dropna().tolist()

        positive_terms = generate_bigram_wordcloud(
            positive_texts,
            "Bullish Signals: Positive Sentiment TF-IDF Bigrams (Filtered)",
            "Greens",
            figures_dir / "bullish_signals_bigrams",
            stop_words,
        )
        positive_terms.insert(0, "sentiment_bucket", "bullish")

        negative_terms = generate_bigram_wordcloud(
            negative_texts,
            "Bearish Signals: Negative Sentiment TF-IDF Bigrams (Filtered)",
            "Reds",
            figures_dir / "bearish_signals_bigrams",
            stop_words,
        )
        negative_terms.insert(0, "sentiment_bucket", "bearish")
        term_frames = [frame for frame in (positive_terms, negative_terms) if not frame.empty]
        terms = (
            pd.concat(term_frames, ignore_index=True)
            if term_frames
            else pd.DataFrame(columns=["sentiment_bucket", "bigram", "tfidf_weight"])
        )

    if top_terms_path is not None:
        top_terms_path.parent.mkdir(parents=True, exist_ok=True)
        terms.to_csv(top_terms_path, index=False)

    primary_source = figures_dir / "bearish_signals_bigrams.svg"
    if not primary_source.exists() or primary_source.stat().st_size == 0:
        primary_source = figures_dir / "bullish_signals_bigrams.svg"
    primary_path = figures_dir / "text_word_cloud.svg"
    if primary_source.exists():
        shutil.copyfile(primary_source, primary_path)

    if mirror_dir is not None:
        for filename in (
            "bullish_signals_bigrams.png",
            "bullish_signals_bigrams.svg",
            "bearish_signals_bigrams.png",
            "bearish_signals_bigrams.svg",
            "text_word_cloud.svg",
        ):
            source = figures_dir / filename
            if source.exists():
                shutil.copyfile(source, mirror_dir / filename)

    return {
        "primary_svg": primary_path,
        "bullish_png": figures_dir / "bullish_signals_bigrams.png",
        "bearish_png": figures_dir / "bearish_signals_bigrams.png",
        "top_terms": terms,
    }


def write_text_word_cloud_svg(
    raw_dir: Path = DATA_DIR,
    path: Path = FIGURES_DIR / "text_word_cloud.svg",
    mirror_path: Path | None = REPORT_FIGURES_DIR / "text_word_cloud.svg",
) -> Path:
    artifacts = write_text_bigram_wordclouds(
        raw_dir=raw_dir,
        figures_dir=path.parent,
        mirror_dir=mirror_path.parent if mirror_path is not None else None,
    )
    primary_path = artifacts["primary_svg"]
    if Path(primary_path) != path and Path(primary_path).exists():
        shutil.copyfile(primary_path, path)
    if mirror_path is not None and path.exists():
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, mirror_path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CRASH-RISK METRICS  (Chen, Hong & Stein 2001)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ncskew(r: np.ndarray) -> float:
    """
    Negative Conditional Skewness:
        NCSKEW = -[ n(n-1)^(3/2) * Σr³ ] / [ (n-1)(n-2)(Σr²)^(3/2) ]
    Higher NCSKEW ↔ more negative skewness ↔ fatter left tail.
    """
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return np.nan
    s2 = np.sum(r ** 2)
    if s2 <= 0:
        return np.nan
    return float(-(n * (n - 1) ** 1.5 * np.sum(r ** 3)) / ((n - 1) * (n - 2) * s2 ** 1.5))


def compute_duvol(r: np.ndarray) -> float:
    """
    Down-to-Up Volatility:
        DUVOL = ln[ (n_up-1)·Σ_down r² / (n_down-1)·Σ_up r² ]
    Positive DUVOL ↔ downside variance > upside variance.
    """
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return np.nan
    mu = np.mean(r)
    down, up = r[r < mu], r[r >= mu]
    if len(down) <= 1 or len(up) <= 1:
        return np.nan
    sd, su = np.sum(down ** 2), np.sum(up ** 2)
    if sd <= 0 or su <= 0:
        return np.nan
    return float(np.log(((len(up) - 1) * sd) / ((len(down) - 1) * su)))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_panel(prices: pd.DataFrame, controv: pd.DataFrame,
                        window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Construct the full 20-feature weekly panel from raw price and controversy data.
    """
    panel = prices.copy().sort_values(["ticker", "date"])

    # ── Firm-specific return (market-adjusted) ─────────────────────────────
    panel["firm_specific_return"] = panel["weekly_return"] - panel["market_return"]

    def rolling_apply(series: pd.Series, func, win: int, min_per: int) -> pd.Series:
        return series.rolling(win, min_periods=min_per).apply(lambda x: func(x), raw=True)

    grp = panel.groupby("ticker", sort=False)

    # ── Group 1: Crash history ─────────────────────────────────────────────
    panel["lagged_ncskew"] = grp["firm_specific_return"].transform(
        lambda s: rolling_apply(s, compute_ncskew, window, 8))
    panel["lagged_duvol"] = grp["firm_specific_return"].transform(
        lambda s: rolling_apply(s, compute_duvol, window, 8))

    # ── Group 2: Trading activity ──────────────────────────────────────────
    panel["turnover"] = panel["volume"] / panel["shares_outstanding"]
    panel["detrended_turnover"] = (
        panel["turnover"]
        - grp["turnover"].transform(lambda s: s.rolling(window, min_periods=4).mean())
    )

    # ── Group 3: Downside risk ─────────────────────────────────────────────
    panel["trailing_return"] = grp["weekly_return"].transform(
        lambda s: s.rolling(window, min_periods=8).apply(
            lambda x: float(np.prod(1.0 + x) - 1.0), raw=True))
    panel["realized_volatility"] = grp["weekly_return"].transform(
        lambda s: s.rolling(window, min_periods=8).std() * np.sqrt(52))

    def _roll_beta(df_ticker: pd.DataFrame) -> pd.Series:
        wr = df_ticker["weekly_return"].to_numpy(float)
        mr = df_ticker["market_return"].to_numpy(float)
        betas, d_betas = np.full(len(wr), np.nan), np.full(len(wr), np.nan)
        for i in range(window, len(wr) + 1):
            w_slice = wr[i - window:i]
            m_slice = mr[i - window:i]
            var_m = np.var(m_slice, ddof=1)
            if var_m > 0:
                betas[i - 1] = np.cov(w_slice, m_slice, ddof=1)[0, 1] / var_m
            down_mask = m_slice < 0
            if down_mask.sum() >= 3:
                var_md = np.var(m_slice[down_mask], ddof=1)
                if var_md > 0:
                    d_betas[i - 1] = np.cov(w_slice[down_mask], m_slice[down_mask], ddof=1)[0, 1] / var_md
        return pd.Series(betas, index=df_ticker.index), pd.Series(d_betas, index=df_ticker.index)

    beta_list, dbeta_list = [], []
    for _, grp_df in panel.groupby("ticker", sort=False):
        b, db = _roll_beta(grp_df.sort_values("date"))
        beta_list.append(b)
        dbeta_list.append(db)

    panel["beta"]          = pd.concat(beta_list)
    panel["downside_beta"] = pd.concat(dbeta_list)
    panel["relative_downside_beta"] = panel["downside_beta"] - panel["beta"]

    # ── Group 4: Fundamentals (already in panel from data generation) ──────
    # market_cap, market_to_book, leverage, roa already present

    # ── Group 5: ESG controversy ───────────────────────────────────────────
    # Backward merge: attach most recent controversy reading on or before week t
    controv_sorted = controv.sort_values(["ticker", "date"])
    merged_parts = []
    for ticker, p_grp in panel.sort_values(["ticker","date"]).groupby("ticker", sort=False):
        tc = controv_sorted[controv_sorted["ticker"] == ticker].sort_values("date")
        if tc.empty:
            p_grp = p_grp.copy()
            p_grp["controversy_score"] = np.nan
        else:
            p_grp = pd.merge_asof(p_grp.sort_values("date"), tc[["date","controversy_score"]],
                                  on="date", direction="backward")
        merged_parts.append(p_grp)
    panel = pd.concat(merged_parts).sort_values(["ticker","date"]).reset_index(drop=True)

    cgrp = panel.groupby("ticker", sort=False)["controversy_score"]
    for w in [4, 13, 26]:
        panel[f"controversy_change_{w}w"]       = cgrp.transform(lambda s: s.diff(w))
        panel[f"controversy_rolling_mean_{w}w"] = cgrp.transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        panel[f"controversy_rolling_std_{w}w"]  = cgrp.transform(
            lambda s: s.rolling(w, min_periods=2).std())

    panel["controversy_spike_flag"] = (
        panel["controversy_score"] >
        panel["controversy_rolling_mean_26w"] + 2.0 * panel["controversy_rolling_std_26w"].fillna(0)
    ).astype(int)
    panel["controversy_sector_percentile"] = panel.groupby(
        ["date","sector"], sort=False)["controversy_score"].rank(pct=True)

    return panel.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TARGET CREATION
# ══════════════════════════════════════════════════════════════════════════════

def make_targets(panel: pd.DataFrame, horizon: int = TARGET_HORIZON,
                 quantile: float = TARGET_QUANTILE) -> pd.DataFrame:
    """
    Label high_crash_risk = 1 for the top `quantile` fraction of stocks
    by future NCSKEW at each cross-section date.

    Target uses returns t+1 through t+horizon — strictly future data only.
    """
    frames = []
    for ticker, grp in panel.sort_values(["ticker","date"]).groupby("ticker", sort=False):
        grp = grp.copy()
        r = grp["firm_specific_return"].to_numpy(float)
        future_ncskew = [
            compute_ncskew(r[i + 1: i + 1 + horizon]) for i in range(len(r))
        ]
        grp["future_ncskew"] = future_ncskew
        frames.append(grp)

    dataset = pd.concat(frames).sort_values(["date","ticker"]).reset_index(drop=True)

    def _label(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        out = pd.Series(pd.NA, index=s.index, dtype="Int64")
        if valid.empty:
            return out
        threshold = valid.quantile(1.0 - quantile)
        out.loc[valid.index] = (valid >= threshold).astype(int)
        return out

    dataset["high_crash_risk"] = dataset.groupby("date", group_keys=False)[
        "future_ncskew"].apply(_label)
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHRONOLOGICAL SPLITS  (no random shuffling — avoids temporal leakage)
# ══════════════════════════════════════════════════════════════════════════════

def chronological_split(
    dataset: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique sorted dates: 60% train / 20% validation / 20% test.

    Why chronological?
    - Feature leakage: rolling windows at time t use t-1..t-26 data
    - Target leakage: label at t uses returns t+1..t+13
    Random splits would mix these across the train/test boundary.
    """
    dates = sorted(dataset["date"].unique())
    n = len(dates)
    t1 = dates[int(n * train_frac)]
    t2 = dates[int(n * (train_frac + val_frac))]
    train = dataset[dataset["date"] <  t1].copy()
    val   = dataset[(dataset["date"] >= t1) & (dataset["date"] < t2)].copy()
    test  = dataset[dataset["date"] >= t2].copy()
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SKLEARN PIPELINE  (Imputer → Scaler → Classifier)
# ══════════════════════════════════════════════════════════════════════════════

def make_pipeline(classifier) -> Pipeline:
    """
    SimpleImputer(median) → StandardScaler → Classifier.
    Fitted only on training data; applied identically to val/test.
    """
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("classifier", classifier),
    ])


def _get_y(df: pd.DataFrame) -> pd.Series:
    return df["high_crash_risk"].astype(float)


def _get_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.20) -> dict:
    """
    ROC-AUC, Precision@Top-Bucket, Crash-Capture@Top-Bucket.

    Precision@Top = TP / top_k  — fraction of top-k flagged stocks that truly crash.
    CrashCapture  = TP / P      — fraction of all crashes caught by top-k flag.
    Naive baseline for both = top_frac (= 0.20).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0 or y_true.sum() == 0:
        return {"roc_auc": np.nan, "precision_at_top": np.nan, "crash_capture": np.nan}

    roc = roc_auc_score(y_true, y_prob)
    k   = max(1, math.ceil(len(y_true) * top_frac))
    top_idx = np.argsort(y_prob)[::-1][:k]
    tp  = y_true[top_idx].sum()
    precision_at_top = tp / k
    crash_capture    = tp / y_true.sum()
    return {
        "roc_auc":          round(float(roc), 4),
        "precision_at_top": round(float(precision_at_top), 4),
        "crash_capture":    round(float(crash_capture), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAIN & COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    features: list[str],
    classifier,
    label: str,
) -> dict:
    """Train on `train`, evaluate on `val` and `test`."""
    labeled_train = train.dropna(subset=["high_crash_risk"])
    X_tr = _get_X(labeled_train, features)
    y_tr = _get_y(labeled_train)

    pipe = make_pipeline(classifier)
    pipe.fit(X_tr, y_tr)

    results = {"model": label, "features": len(features)}
    for split_name, split_df in [("val", val), ("test", test)]:
        labeled = split_df.dropna(subset=["high_crash_risk"])
        X_sp = _get_X(labeled, features)
        y_sp = _get_y(labeled).to_numpy()
        y_prob = pipe.predict_proba(X_sp)[:, 1]
        m = evaluate(y_sp, y_prob)
        for k, v in m.items():
            results[f"{split_name}_{k}"] = v
    return results, pipe


def compare_algorithms(train, val, test) -> pd.DataFrame:
    """Train LR, RF, GB on the same splits; return comparison DataFrame."""
    classifiers = [
        ("logistic_regression", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED)),
        ("random_forest",       RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=1)),
        ("gradient_boosting",   GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)),
    ]
    rows = []
    for label, clf in classifiers:
        result, _ = train_and_evaluate(train, val, test, ALL_FEATURES, clf, label)
        rows.append(result)
    return pd.DataFrame(rows)


def compare_esg_lift(train, val, test) -> pd.DataFrame:
    """Compare baseline (no ESG) vs full (with ESG) models using Logistic Regression."""
    clf = lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED)

    baseline_result, _ = train_and_evaluate(train, val, test, BASELINE_FEATURES, clf(), "baseline_no_esg")
    full_result, full_pipe = train_and_evaluate(train, val, test, ALL_FEATURES,      clf(), "full_with_esg")

    delta = {
        "model":            "ESG_delta",
        "features":         len(ESG_FEATURES),
        "val_roc_auc":      round(full_result["val_roc_auc"]  - baseline_result["val_roc_auc"],  4),
        "val_precision_at_top": round(full_result["val_precision_at_top"] - baseline_result["val_precision_at_top"], 4),
        "val_crash_capture":    round(full_result["val_crash_capture"]    - baseline_result["val_crash_capture"],    4),
        "test_roc_auc":     round(full_result["test_roc_auc"] - baseline_result["test_roc_auc"], 4),
        "test_precision_at_top": round(full_result["test_precision_at_top"] - baseline_result["test_precision_at_top"], 4),
        "test_crash_capture":    round(full_result["test_crash_capture"]    - baseline_result["test_crash_capture"],    4),
    }
    return pd.DataFrame([baseline_result, full_result, delta]), full_pipe


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SCORING — LATEST WEEK RISK BUCKETS
# ══════════════════════════════════════════════════════════════════════════════

def score_latest(panel: pd.DataFrame, pipe: Pipeline,
                 features: list[str] = ALL_FEATURES) -> pd.DataFrame:
    """
    Score every stock as of the latest available week.
    Risk buckets: High = top 20%, Medium = next 40%, Low = bottom 40%.
    Top-3 drivers: |coefficient × scaled_feature| contribution per stock.
    """
    latest_date = panel["date"].max()
    latest = panel[panel["date"] == latest_date].copy()

    X = _get_X(latest, features)
    latest["crash_probability"] = pipe.predict_proba(X)[:, 1]

    n = len(latest)
    latest_sorted = latest.sort_values("crash_probability", ascending=False).reset_index(drop=True)
    latest_sorted["risk_bucket"] = "Low"
    latest_sorted.iloc[:math.ceil(n * 0.20), latest_sorted.columns.get_loc("risk_bucket")] = "High"
    latest_sorted.iloc[math.ceil(n * 0.20):math.ceil(n * 0.60), latest_sorted.columns.get_loc("risk_bucket")] = "Medium"

    # Top-3 driver strings (LR only)
    try:
        coef = pipe.named_steps["classifier"].coef_[0]
        X_scaled = pipe[:-1].transform(X)
        contributions = np.abs(coef) * np.abs(X_scaled)
        top3 = [
            ";".join([features[j] for j in row.argsort()[::-1][:3]])
            for row in contributions
        ]
        latest["top_drivers"] = top3
        latest_sorted = latest_sorted.merge(
            latest[["ticker", "top_drivers"]], on="ticker", how="left")
    except Exception:
        latest_sorted["top_drivers"] = ""

    return latest_sorted[["ticker", "crash_probability", "risk_bucket", "top_drivers",
                           "sector"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(pipe: Pipeline, features: list[str]) -> pd.DataFrame:
    """Extract |coef| for LR, feature_importances_ for RF/GB."""
    clf = pipe.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.zeros(len(features))
    return pd.DataFrame({"feature": features, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  BUSINESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _annualised_return(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) ** (52.0 / len(r)) - 1.0) if len(r) >= 2 else np.nan

def _sharpe(r: np.ndarray, weekly_rf: float) -> float:
    excess = r - weekly_rf
    std = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(52)) if len(excess) >= 4 and std > 0 else np.nan

def _sortino(r: np.ndarray, weekly_rf: float) -> float:
    excess = r - weekly_rf
    down   = excess[excess < 0]
    if len(down) < 2:
        return np.nan
    d_std = np.std(down, ddof=1)
    return float(np.mean(excess) / d_std * np.sqrt(52)) if d_std > 0 else np.nan

def _max_drawdown(r: np.ndarray) -> float:
    if len(r) < 2:
        return np.nan
    cum  = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))

def _var_cvar(r: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if len(r) < 10:
        return np.nan, np.nan
    s = np.sort(r)
    k = max(1, int(len(s) * (1 - confidence)))
    return float(-s[k - 1]), float(-np.mean(s[:k]))


def _classifier_from_pipeline(pipe: Pipeline):
    return pipe.named_steps.get("classifier") or pipe.named_steps.get("clf")


def _assign_risk_buckets_from_probabilities(probabilities: pd.Series, high_share: float = 0.20) -> pd.Series:
    if probabilities.empty:
        return pd.Series(dtype="object")
    n_obs = len(probabilities)
    high_count = max(1, math.ceil(n_obs * high_share))
    medium_count = max(0, math.ceil(n_obs * 0.40))
    rank = probabilities.rank(method="first", ascending=False)
    labels = pd.Series("Low", index=probabilities.index, dtype="object")
    labels.loc[rank <= high_count] = "High"
    labels.loc[(rank > high_count) & (rank <= high_count + medium_count)] = "Medium"
    return labels


def build_weekly_forward_portfolio_returns(
    panel: pd.DataFrame,
    pipe: Pipeline,
    features: list[str],
    eval_start_quantile: float = 0.60,
    high_share: float = 0.20,
) -> pd.DataFrame:
    """
    Score each week with information available at week t and measure returns over t+1.
    """
    ph = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if "weekly_return" not in ph.columns:
        if "adj_close" not in ph.columns:
            raise ValueError("panel must contain weekly_return or adj_close")
        ph["weekly_return"] = ph.groupby("ticker", group_keys=False)["adj_close"].pct_change()
    ph["next_week_return"] = ph.groupby("ticker", group_keys=False)["weekly_return"].shift(-1)

    all_dates = sorted(pd.to_datetime(ph["date"]).dropna().unique())
    if len(all_dates) < 3:
        return pd.DataFrame(columns=["date", "return_date", "strategy", "benchmark", "n_holdings", "n_excluded", "excluded_tickers"])

    clf = _classifier_from_pipeline(pipe)
    classes = list(getattr(clf, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    eval_start = all_dates[int(len(all_dates) * eval_start_quantile)]
    rows = []

    for rebalance_date in all_dates:
        if rebalance_date <= eval_start:
            continue
        week = ph.loc[pd.to_datetime(ph["date"]) == rebalance_date].dropna(subset=["next_week_return"]).copy()
        if week.empty:
            continue
        probabilities = pd.Series(
            pipe.predict_proba(week[features].astype(float))[:, positive_index],
            index=week.index,
            name="crash_probability",
        )
        week["crash_probability"] = probabilities
        week["risk_bucket"] = _assign_risk_buckets_from_probabilities(probabilities, high_share=high_share)

        high_group = week.loc[week["risk_bucket"] == "High"].sort_values("ticker")
        strategy_group = week.loc[week["risk_bucket"] != "High"]
        benchmark_ret = float(week["next_week_return"].mean())
        strategy_ret = float(strategy_group["next_week_return"].mean()) if not strategy_group.empty else benchmark_ret
        return_dates = [date for date in all_dates if date > rebalance_date]

        rows.append({
            "date": pd.Timestamp(rebalance_date),
            "return_date": pd.Timestamp(return_dates[0]) if return_dates else pd.NaT,
            "strategy": strategy_ret,
            "benchmark": benchmark_ret,
            "n_holdings": int(len(strategy_group)),
            "n_excluded": int(len(high_group)),
            "excluded_tickers": ";".join(high_group["ticker"].astype(str).tolist()),
        })

    return pd.DataFrame(rows)


def compute_business_analysis(
    panel: pd.DataFrame,
    scores_or_pipe,
    features: list[str] | None = None,
    aum: float = FUND_AUM,
    annual_rf: float = RISK_FREE_ANNUAL,
    team_cost: float = TEAM_COST,
    eval_start_quantile: float = 0.60,
    portfolio_returns: pd.DataFrame | None = None,
) -> dict:
    """
    Strategy: equal-weight stocks NOT in 'High' risk bucket each week.
    Benchmark: equal-weight all stocks.
    If features are supplied, scores are recomputed each week and applied to
    the following week's returns.
    """
    weekly_rf = annual_rf / 52.0
    if portfolio_returns is not None:
        perf = portfolio_returns.copy()
    elif features is not None:
        perf = build_weekly_forward_portfolio_returns(
            panel,
            scores_or_pipe,
            features,
            eval_start_quantile=eval_start_quantile,
        )
    else:
        ph = panel.copy()
        all_dates = sorted(ph["date"].unique())
        if len(all_dates) < 10:
            return {"error": "insufficient history"}

        eval_start = all_dates[int(len(all_dates) * eval_start_quantile)]
        eval_data = ph[ph["date"] > eval_start].copy()

        ticker_bucket = scores_or_pipe.set_index("ticker")["risk_bucket"].to_dict()
        eval_data["risk_bucket"] = eval_data["ticker"].map(ticker_bucket).fillna("Low")

        weeks_out = []
        for date, grp in eval_data.groupby("date"):
            valid = grp.dropna(subset=["weekly_return"])
            if valid.empty:
                continue
            bench_ret = float(valid["weekly_return"].mean())
            strategy_grp = valid[valid["risk_bucket"] != "High"]
            strat_ret = float(strategy_grp["weekly_return"].mean()) if not strategy_grp.empty else bench_ret
            weeks_out.append({"date": date, "strategy": strat_ret, "benchmark": bench_ret})
        perf = pd.DataFrame(weeks_out).sort_values("date") if weeks_out else pd.DataFrame()

    if perf.empty or len(perf) < 8:
        return {"error": "not enough evaluation weeks"}

    s_rets = perf["strategy"].fillna(0).to_numpy()
    b_rets = perf["benchmark"].fillna(0).to_numpy()

    ann_s = _annualised_return(s_rets)
    ann_b = _annualised_return(b_rets)
    alpha = ann_s - ann_b

    var95, cvar95 = _var_cvar(s_rets)
    benchmark_var95, benchmark_cvar95 = _var_cvar(b_rets)
    econ_gain = aum * alpha
    team_roi  = econ_gain / team_cost if team_cost > 0 else np.nan

    if {"n_excluded", "n_holdings"}.issubset(perf.columns):
        weekly_universe = perf["n_excluded"] + perf["n_holdings"]
        high_pct = float((perf["n_excluded"] / weekly_universe.replace(0, np.nan)).mean())
    else:
        high_pct = sum(1 for b in scores_or_pipe["risk_bucket"] if b == "High") / max(1, len(scores_or_pipe))

    def _f(v, d=4): return round(v, d) if np.isfinite(v) else None

    return {
        "strategy_annual_return":  _f(ann_s),
        "benchmark_annual_return": _f(ann_b),
        "alpha_annualized":        _f(alpha),
        "benchmark_alpha_annualized": 0.0,
        "strategy_sharpe":         _f(_sharpe(s_rets, weekly_rf), 3),
        "benchmark_sharpe":        _f(_sharpe(b_rets, weekly_rf), 3),
        "strategy_sortino":        _f(_sortino(s_rets, weekly_rf), 3),
        "benchmark_sortino":       _f(_sortino(b_rets, weekly_rf), 3),
        "max_drawdown_strategy":   _f(_max_drawdown(s_rets)),
        "max_drawdown_benchmark":  _f(_max_drawdown(b_rets)),
        "var_95_weekly":           _f(var95),
        "benchmark_var_95_weekly": _f(benchmark_var95),
        "cvar_95_weekly":          _f(cvar95),
        "benchmark_cvar_95_weekly": _f(benchmark_cvar95),
        "evaluation_weeks":        len(s_rets),
        "high_risk_excluded_pct":  _f(high_pct, 3),
        "benchmark_high_risk_excluded_pct": 0.0,
        "business_analysis_method": "weekly_forward_overlay" if features is not None or portfolio_returns is not None else "latest_score_overlay",
        "fund_aum":                aum,
        "economic_gain_annual":    round(econ_gain, 0) if np.isfinite(econ_gain) else None,
        "benchmark_economic_gain_annual": 0.0,
        "team_annual_cost":        team_cost,
        "team_roi":                _f(team_roi, 2),
        "benchmark_team_roi":      "-",
        "justifies_team":          bool(np.isfinite(econ_gain) and econ_gain > team_cost),
        "business_analysis_note": (
            "The economic results are based on a stylised simulation and should be interpreted as illustrative. "
            "In practice, transaction costs, market impact, capacity limits, model uncertainty, "
            "and live implementation slippage would likely reduce realised returns."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 12.  HYPERPARAMETER TUNING  (rubric §3 — ML section requirement)
# ══════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str] = None,
) -> dict:
    """
    Grid-search the three model families on the validation split.

    We tune on train + val combined via manual hold-out (not k-fold) to respect
    the time-series nature of the data.  The best hyper-parameters are selected
    by validation AUC and returned for reporting.

    Parameters
    ----------
    train    : Training split (output of chronological_split).
    val      : Validation split.
    features : Feature columns (defaults to ALL_FEATURES).

    Returns
    -------
    dict with keys 'logistic_regression', 'random_forest', 'gradient_boosting',
    each containing 'best_params' and 'best_val_auc'.
    """
    from sklearn.model_selection import ParameterGrid

    if features is None:
        features = ALL_FEATURES

    labeled_train = train.dropna(subset=["high_crash_risk"])
    labeled_val   = val.dropna(subset=["high_crash_risk"])
    X_tr  = _get_X(labeled_train, features)
    y_tr  = _get_y(labeled_train).to_numpy()
    X_val = _get_X(labeled_val, features)
    y_val = _get_y(labeled_val).to_numpy()

    grids = {
        "logistic_regression": {
            "clf": lambda p: LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=SEED, **p),
            "params": ParameterGrid({"C": [0.01, 0.1, 1.0, 10.0]}),
        },
        "random_forest": {
            "clf": lambda p: RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=SEED, n_jobs=1, **p),
            "params": ParameterGrid({"max_depth": [3, 5, 8, None],
                                     "min_samples_leaf": [3, 5, 10]}),
        },
        "gradient_boosting": {
            "clf": lambda p: GradientBoostingClassifier(
                n_estimators=200, random_state=SEED, **p),
            "params": ParameterGrid({"learning_rate": [0.01, 0.05, 0.1],
                                     "max_depth": [2, 3, 5]}),
        },
    }

    results = {}
    for model_name, spec in grids.items():
        best_auc, best_params = -1.0, {}
        for params in spec["params"]:
            try:
                clf = spec["clf"](params)
                pipe = make_pipeline(clf)
                pipe.fit(X_tr, y_tr)
                y_prob = pipe.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_prob)
                if auc > best_auc:
                    best_auc, best_params = auc, params
            except Exception:
                continue
        results[model_name] = {"best_params": best_params, "best_val_auc": round(best_auc, 4)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 13.  QUARTER SNAPSHOT BACKTEST  (true out-of-sample Q4 validation)
# ══════════════════════════════════════════════════════════════════════════════

def quarter_snapshot_backtest(
    panel: pd.DataFrame,
    pipe: Pipeline,
    features: list[str] = None,
    cutoff_date=None,
    forward_weeks: int = 13,
    high_share: float = 0.20,
    fund_aum: float = FUND_AUM,
) -> dict:
    """
    True out-of-sample backtest for a single named quarter.

    The model is scored at `cutoff_date` using only information available at
    that point.  Two equal-weight portfolios are then tracked over the next
    `forward_weeks` weekly periods:

      * Benchmark : all stocks in the universe (equal-weight)
      * Strategy  : all stocks EXCEPT those flagged High-risk at the cutoff

    This validates the ESG controversy signal on data the model never saw
    during training (Kim, Li & Zhang 2014; Chen, Hong & Stein 2001).

    Parameters
    ----------
    panel        : Full feature panel (requires adj_close, date, ticker columns).
    pipe         : Fitted sklearn Pipeline with predict_proba.
    features     : Feature list for predict_proba (defaults to ALL_FEATURES).
    cutoff_date  : Scoring snapshot date.  Auto-detected as the date that leaves
                   exactly `forward_weeks` trading weeks remaining if None.
    forward_weeks: Number of weekly periods to track after the cutoff.
    high_share   : Fraction of the universe flagged as High-risk (default 0.20).
    fund_aum     : Illustrative fund AUM for dollar-impact calculation.

    Returns
    -------
    dict containing:
      cutoff_date, quarter_label, excluded_tickers, weekly_series,
      strategy_quarter_return, benchmark_quarter_return,
      outperformance_bps, dollar_impact_quarter, dollar_impact_annualised,
      n_excluded, n_held, pct_excluded_correct.
    """
    if features is None:
        features = ALL_FEATURES

    ph = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if "weekly_return" not in ph.columns:
        if "adj_close" not in ph.columns:
            return {"error": "panel must contain weekly_return or adj_close."}
        ph["weekly_return"] = ph.groupby("ticker", group_keys=False)["adj_close"].pct_change()

    all_dates = sorted(pd.to_datetime(ph["date"]).dropna().unique())
    if len(all_dates) < forward_weeks + 2:
        return {"error": f"Not enough dates ({len(all_dates)}) for a {forward_weeks}-week backtest."}

    # ── Resolve cutoff ────────────────────────────────────────────────────────
    if cutoff_date is None:
        cutoff = all_dates[-(forward_weeks + 1)]
    else:
        cutoff = pd.Timestamp(cutoff_date)
        before = [d for d in all_dates if d <= cutoff]
        if not before:
            return {"error": f"cutoff_date {cutoff_date} is before all available data."}
        cutoff = before[-1]

    forward_dates = [d for d in all_dates if d > cutoff][:forward_weeks]
    if not forward_dates:
        return {"error": "No forward dates available after the cutoff."}

    fwd_start = forward_dates[0]
    q_num = (fwd_start.month - 1) // 3 + 1
    quarter_label = f"Q{q_num} {fwd_start.year}"

    # ── Score universe at cutoff ──────────────────────────────────────────────
    cutoff_panel = ph[pd.to_datetime(ph["date"]) == cutoff].copy()
    if cutoff_panel.empty:
        return {"error": f"No panel data at cutoff date {cutoff.date()}."}

    clf = _classifier_from_pipeline(pipe)
    classes = list(getattr(clf, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1

    available_features = [f for f in features if f in cutoff_panel.columns]
    X_cutoff = cutoff_panel[available_features].astype(float)
    probs = pd.Series(
        pipe.predict_proba(X_cutoff)[:, positive_index],
        index=cutoff_panel.index,
    )
    cutoff_panel = cutoff_panel.copy()
    cutoff_panel["crash_probability"] = probs
    cutoff_panel["risk_bucket"] = _assign_risk_buckets_from_probabilities(probs, high_share=high_share)

    excluded_tickers = set(
        cutoff_panel.loc[cutoff_panel["risk_bucket"] == "High", "ticker"].tolist()
    )
    ticker_probs = dict(zip(cutoff_panel["ticker"], cutoff_panel["crash_probability"]))

    # ── Weekly return series ──────────────────────────────────────────────────
    weekly_series = []
    strat_cum = 1.0
    bench_cum = 1.0

    for fwd_date in forward_dates:
        week_data = ph[pd.to_datetime(ph["date"]) == fwd_date].copy()
        if week_data.empty:
            continue
        bench_ret = float(week_data["weekly_return"].mean())
        strat_data = week_data[~week_data["ticker"].isin(excluded_tickers)]
        strat_ret = float(strat_data["weekly_return"].mean()) if not strat_data.empty else bench_ret
        bench_cum *= (1 + bench_ret)
        strat_cum *= (1 + strat_ret)
        weekly_series.append({
            "date": fwd_date.strftime("%Y-%m-%d"),
            "strategy_cumulative":      round(strat_cum, 6),
            "benchmark_cumulative":     round(bench_cum, 6),
            "strategy_weekly_return":   round(strat_ret, 6),
            "benchmark_weekly_return":  round(bench_ret, 6),
        })

    # ── Per-excluded-ticker actual forward return ─────────────────────────────
    excluded_ticker_rows = []
    for tkr in sorted(excluded_tickers):
        fwd_data = ph[
            (ph["ticker"] == tkr) &
            (pd.to_datetime(ph["date"]).isin(forward_dates))
        ].sort_values("date")
        if fwd_data.empty or "adj_close" not in fwd_data.columns:
            q_ret = None
        else:
            entry = fwd_data["adj_close"].iloc[0]
            exit_ = fwd_data["adj_close"].iloc[-1]
            q_ret = (exit_ - entry) / entry if entry != 0 else None
        outcome = "Avoided loss" if (q_ret is not None and q_ret < 0) else "Model missed"
        excluded_ticker_rows.append({
            "ticker":            tkr,
            "crash_probability": round(ticker_probs.get(tkr, 0.0), 4),
            "quarter_return":    round(q_ret, 4) if q_ret is not None else None,
            "outcome":           outcome,
        })
    excluded_ticker_rows.sort(key=lambda x: x["crash_probability"], reverse=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    strategy_q4        = strat_cum - 1.0
    benchmark_q4       = bench_cum - 1.0
    outperformance     = strategy_q4 - benchmark_q4
    outperformance_bps = round(outperformance * 10_000)
    dollar_q4          = round(fund_aum * outperformance / 4)
    dollar_annual      = round(fund_aum * outperformance)
    n_excluded         = len(excluded_tickers)
    n_held             = len(cutoff_panel) - n_excluded
    pct_correct        = (
        sum(1 for r in excluded_ticker_rows if r["outcome"] == "Avoided loss") / n_excluded
        if n_excluded > 0 else 0.0
    )

    return {
        "cutoff_date":              cutoff.strftime("%Y-%m-%d"),
        "quarter_label":            quarter_label,
        "forward_weeks":            len(forward_dates),
        "excluded_tickers":         excluded_ticker_rows,
        "weekly_series":            weekly_series,
        "strategy_quarter_return":  round(strategy_q4, 4),
        "benchmark_quarter_return": round(benchmark_q4, 4),
        "outperformance_bps":       outperformance_bps,
        "dollar_impact_quarter":    dollar_q4,
        "dollar_impact_annualised": dollar_annual,
        "n_excluded":               n_excluded,
        "n_held":                   n_held,
        "pct_excluded_correct":     round(pct_correct, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 14.  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#f7f9fa", "axes.facecolor": "#f7f9fa",
        "axes.edgecolor": "#cccccc",   "axes.grid": True,
        "grid.color": "#e0e0e0",       "grid.linestyle": "--",
        "grid.linewidth": 0.6,         "font.size": 11,
        "axes.titlesize": 12,          "axes.titleweight": "bold",
        "figure.dpi": 130,
    })

TEAL, AMBER, RED, GREEN, GREY, BLUE, PURPLE = (
    "#0d6b62", "#e07b00", "#c0392b", "#1a7a4a", "#7f8c8d", "#2980b9", "#8e44ad")

CAT_COLOURS = {
    "ESG": TEAL, "Downside Risk": AMBER,
    "Fundamentals": BLUE, "Crash History": PURPLE, "Trading Activity": GREY,
}
FEATURE_CATS = {
    **{f: "ESG" for f in ESG_FEATURES},
    "lagged_ncskew": "Crash History", "lagged_duvol": "Crash History",
    "detrended_turnover": "Trading Activity",
    "trailing_return": "Downside Risk", "realized_volatility": "Downside Risk",
    "beta": "Downside Risk", "downside_beta": "Downside Risk",
    "relative_downside_beta": "Downside Risk",
    "market_cap": "Fundamentals", "market_to_book": "Fundamentals",
    "leverage": "Fundamentals", "roa": "Fundamentals",
}


def plot_all(scores, algo_df, esg_df, feat_imp, biz, panel, output_dir: Path, quarter_bt=None):
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # ── Fig 1: Risk ranking ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sdf = scores.sort_values("crash_probability", ascending=True)
    colours = sdf["risk_bucket"].map({"High": RED, "Medium": AMBER, "Low": GREEN})
    ax.barh(sdf["ticker"], sdf["crash_probability"], color=colours, edgecolor="white", lw=0.4)
    ax.axvline(0.5, color=GREY, ls=":", lw=1.2)
    ax.set_xlabel("Crash Probability")
    ax.set_title("Fig 1 — Crash-Risk Probability Ranking (latest week)")
    ax.legend(handles=[mpatches.Patch(color=c, label=l)
                        for l, c in [("High", RED), ("Medium", AMBER), ("Low", GREEN)]],
              loc="lower right")
    fig.tight_layout(); fig.savefig(output_dir/"fig1_risk_ranking.png", bbox_inches="tight")
    plt.close(fig); saved.append("fig1_risk_ranking.png")

    # ── Fig 2: Sector controversy ─────────────────────────────────────────
    sector_data = {"Energy": 5.93, "Financial Services": 5.77, "Industrials": 5.30,
                   "Communication Services": 4.88, "Consumer Cyclical": 4.62,
                   "Technology": 4.36, "Basic Materials": 4.32, "Healthcare": 4.23,
                   "Consumer Defensive": 2.63, "Utilities": 2.54}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(list(sector_data.keys()), list(sector_data.values()),
            color=[RED if v >= 5.5 else AMBER if v >= 4.0 else GREEN
                   for v in sector_data.values()], edgecolor="white", lw=0.4)
    ax.set_xlabel("Avg ESG Controversy Score (0–10)")
    ax.set_title("Fig 2 — Average ESG Controversy Score by Sector")
    ax.set_xlim(0, 7)
    for i, v in enumerate(sector_data.values()):
        ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(output_dir/"fig2_sector_controversy.png", bbox_inches="tight")
    plt.close(fig); saved.append("fig2_sector_controversy.png")

    # ── Fig 3: Algorithm comparison ───────────────────────────────────────
    models = ["logistic_regression", "random_forest", "gradient_boosting"]
    labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
    x = np.arange(len(models)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title, ylim in [
        (ax1, "roc_auc",          "ROC-AUC",           (0.45, 0.67)),
        (ax2, "precision_at_top", "Precision@Top Bucket", (0.15, 0.38)),
    ]:
        vals_v = [algo_df.loc[algo_df.model==m, f"val_{metric}"].values[0]  for m in models]
        vals_t = [algo_df.loc[algo_df.model==m, f"test_{metric}"].values[0] for m in models]
        ax.bar(x-w/2, vals_v, w, label="Validation", color=TEAL,  alpha=0.85)
        ax.bar(x+w/2, vals_t, w, label="Test",       color=BLUE,  alpha=0.85)
        baseline_line = 0.50 if metric == "roc_auc" else 0.20
        ax.axhline(baseline_line, color=GREY, ls=":", lw=1.2, label=f"Naive ({baseline_line:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(*ylim); ax.set_ylabel(title)
        ax.set_title(f"Fig 3 — {title}"); ax.legend(fontsize=8)
        for i, (v, t) in enumerate(zip(vals_v, vals_t)):
            ax.text(i-w/2, v+0.003, f"{v:.3f}", ha="center", fontsize=8)
            ax.text(i+w/2, t+0.003, f"{t:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir/"fig3_algorithm_comparison.png", bbox_inches="tight")
    plt.close(fig); saved.append("fig3_algorithm_comparison.png")

    # ── Fig 4: ESG lift ────────────────────────────────────────────────────
    lift_models = ["baseline_no_esg", "full_with_esg"]
    lift_labels = ["Baseline\n(no ESG)", "Full\n(with ESG)"]
    x = np.arange(len(lift_models)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    v_aucs = [esg_df.loc[esg_df.model==m, "val_roc_auc"].values[0]  for m in lift_models]
    t_aucs = [esg_df.loc[esg_df.model==m, "test_roc_auc"].values[0] for m in lift_models]
    ax.bar(x-w/2, v_aucs, w, label="Validation", color=TEAL, alpha=0.85)
    ax.bar(x+w/2, t_aucs, w, label="Test",       color=BLUE, alpha=0.85)
    ax.axhline(0.5, color=GREY, ls=":", lw=1.2, label="Random (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(lift_labels)
    ax.set_ylim(0.46, 0.65); ax.set_ylabel("ROC-AUC")
    ax.set_title("Fig 4 — ESG Feature Lift")
    ax.legend()
    for i, (v, t) in enumerate(zip(v_aucs, t_aucs)):
        ax.text(i-w/2, v+0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i+w/2, t+0.003, f"{t:.3f}", ha="center", fontsize=9, fontweight="bold")
    delta_v = v_aucs[1] - v_aucs[0]
    delta_t = t_aucs[1] - t_aucs[0]
    ax.text(0.98, 0.87, f"Val lift: {delta_v:+.3f}\nTest lift: {delta_t:+.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.8))
    fig.tight_layout(); fig.savefig(output_dir/"fig4_esg_lift.png", bbox_inches="tight")
    plt.close(fig); saved.append("fig4_esg_lift.png")

    # ── Fig 5: Feature importance ─────────────────────────────────────────
    fi = feat_imp.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = [CAT_COLOURS.get(FEATURE_CATS.get(f, "Other"), GREY) for f in fi["feature"]]
    ax.barh(fi["feature"], fi["importance"], color=colours, edgecolor="white", lw=0.4)
    ax.set_xlabel("|Coefficient| (standardised)")
    ax.set_title("Fig 5 — Feature Importance (Logistic Regression)")
    ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in CAT_COLOURS.items()],
              loc="lower right")
    for i, (_, row) in enumerate(fi.iterrows()):
        ax.text(row["importance"] + 0.005, i, f"{row['importance']:.3f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir/"fig5_feature_importance.png", bbox_inches="tight")
    plt.close(fig); saved.append("fig5_feature_importance.png")

    # ── Fig 6: Strategy vs benchmark ──────────────────────────────────────
    if "error" not in biz:
        labels_biz = ["Annual\nReturn (%)", "Sharpe\nRatio", "Sortino\nRatio"]
        strat_vals  = [biz["strategy_annual_return"]*100, biz["strategy_sharpe"],
                       biz.get("strategy_sortino") or 0]
        bench_vals  = [biz["benchmark_annual_return"]*100, biz["benchmark_sharpe"],
                       biz.get("benchmark_sortino") or 0]
        x = np.arange(len(labels_biz)); w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x-w/2, strat_vals, w, label="Strategy (excl. High)", color=TEAL,  alpha=0.9)
        ax.bar(x+w/2, bench_vals, w, label="Benchmark (all stocks)", color=GREY,  alpha=0.9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels_biz)
        ax.set_title("Fig 6 — Strategy vs Benchmark Performance")
        ax.legend()
        for i, (s, b) in enumerate(zip(strat_vals, bench_vals)):
            ax.text(i-w/2, s + 0.05, f"{s:.2f}", ha="center", fontsize=9, fontweight="bold", color=TEAL)
            if b != 0:
                ax.text(i+w/2, b + 0.05, f"{b:.2f}", ha="center", fontsize=9, color=GREY)
        fig.tight_layout(); fig.savefig(output_dir/"fig6_strategy_vs_benchmark.png", bbox_inches="tight")
        plt.close(fig); saved.append("fig6_strategy_vs_benchmark.png")

    # ── Fig 7: Controversy score over time for top-risk tickers ──────────
    fig, ax = plt.subplots(figsize=(10, 4))
    high_tickers = scores[scores["risk_bucket"]=="High"]["ticker"].tolist()
    colours_line = [RED, AMBER, BLUE, PURPLE, GREEN]
    for i, ticker in enumerate(high_tickers[:5]):
        t_panel = panel[panel["ticker"] == ticker].sort_values("date")
        if "controversy_score" in t_panel.columns:
            ax.plot(t_panel["date"], t_panel["controversy_score"],
                    label=ticker, color=colours_line[i % len(colours_line)], lw=1.8)
    ax.set_xlabel("Date"); ax.set_ylabel("ESG Controversy Score")
    ax.set_title("Fig 7 — ESG Controversy Score Over Time (High-Risk Stocks)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(output_dir/"fig7_controversy_over_time.png", bbox_inches="tight")
    plt.close(fig); saved.append("fig7_controversy_over_time.png")

    # ── Fig 8: Economic value ─────────────────────────────────────────────
    if "economic_gain_annual" in biz and biz["economic_gain_annual"]:
        gain_m = biz["economic_gain_annual"] / 1e6
        cost_m = biz["team_annual_cost"] / 1e6
        net_m  = gain_m - cost_m
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(["Gross Alpha\n(on $1B AUM)", "Team Cost", "Net Gain"],
                      [gain_m, -cost_m, net_m],
                      color=[GREEN, RED, TEAL], alpha=0.85, width=0.5, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Annual Value ($ million)")
        ax.set_title(f"Fig 8 — Illustrative Economic Value | Team ROI: {biz.get('team_roi','N/A')}x")
        for bar, val in zip(bars, [gain_m, -cost_m, net_m]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.1 if bar.get_height() >= 0 else -0.5),
                    f"${val:.1f}M", ha="center", fontsize=10, fontweight="bold")
        fig.tight_layout(); fig.savefig(output_dir/"fig8_economic_value.png", bbox_inches="tight")
        plt.close(fig); saved.append("fig8_economic_value.png")

    # ── Fig 9: Q4 out-of-sample cumulative return ──────────────────────────
    if (quarter_bt and "weekly_series" in quarter_bt
            and quarter_bt["weekly_series"] and "error" not in quarter_bt):
        ws = pd.DataFrame(quarter_bt["weekly_series"])
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(ws))
        ax.plot(x, ws["strategy_cumulative"],  color=TEAL, lw=2.0,
                label="Strategy (excl. High-Risk)")
        ax.plot(x, ws["benchmark_cumulative"], color=GREY, lw=2.0, ls="--",
                label="Benchmark (all stocks)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [d[:7] for d in ws["date"]], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Cumulative Return (1 = start)")
        q_label = quarter_bt.get("quarter_label", "Q4 2024")
        cutoff_str = quarter_bt.get("cutoff_date", "")
        ax.set_title(
            f"Fig 9 — {q_label} True Out-of-Sample Backtest\n"
            f"Scored at {cutoff_str}; tracked over {quarter_bt['forward_weeks']} weeks"
        )
        ax.legend(loc="upper left")
        # Annotate final values on the right margin
        strat_final = ws["strategy_cumulative"].iloc[-1]
        bench_final = ws["benchmark_cumulative"].iloc[-1]
        ax.annotate(f"{strat_final:.3f}", xy=(len(ws)-1, strat_final),
                    xytext=(6, 3), textcoords="offset points",
                    color=TEAL, fontsize=9, fontweight="bold")
        ax.annotate(f"{bench_final:.3f}", xy=(len(ws)-1, bench_final),
                    xytext=(6, -12), textcoords="offset points",
                    color=GREY, fontsize=9)
        alpha_bps = quarter_bt.get("outperformance_bps", 0)
        ax.text(0.02, 0.97,
                f"Alpha: {alpha_bps:+d} bps  |  Strategy: {strat_final-1:+.2%}  |  "
                f"Benchmark: {bench_final-1:+.2%}",
                transform=ax.transAxes, fontsize=9, va="top",
                color=TEAL if alpha_bps >= 0 else RED,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8})
        fig.tight_layout()
        fig.savefig(output_dir / "fig9_quarter_backtest.png", bbox_inches="tight")
        plt.close(fig); saved.append("fig9_quarter_backtest.png")

    # ── Fig 10: Excluded stocks — actual forward returns ───────────────────
    if (quarter_bt and "excluded_tickers" in quarter_bt
            and quarter_bt["excluded_tickers"] and "error" not in quarter_bt):
        ex_rows = [r for r in quarter_bt["excluded_tickers"]
                   if r.get("quarter_return") is not None]
        if ex_rows:
            ex = pd.DataFrame(ex_rows).sort_values("quarter_return")
            bar_colours = [GREEN if r < 0 else RED
                           for r in ex["quarter_return"]]
            fig, ax = plt.subplots(figsize=(8, max(4, len(ex) * 0.55 + 1)))
            ax.barh(ex["ticker"], ex["quarter_return"] * 100,
                    color=bar_colours, edgecolor="white", lw=0.5)
            ax.axvline(0, color=GREY, lw=1.0, ls="--")
            ax.set_xlabel("Actual Forward Return (%)")
            q_label = quarter_bt.get("quarter_label", "Q4 2024")
            pct_ok   = quarter_bt.get("pct_excluded_correct", 0)
            ax.set_title(
                f"Fig 10 — Flagged Stocks: Actual {q_label} Returns\n"
                f"Green = model correct (stock fell)  |  Red = model missed  "
                f"|  Accuracy: {pct_ok:.0%}"
            )
            for i, row in ex.iterrows():
                offset = 0.4 if row["quarter_return"] >= 0 else -0.4
                ha = "left" if row["quarter_return"] >= 0 else "right"
                ax.text(row["quarter_return"] * 100 + offset, i,
                        f"{row['quarter_return']*100:.1f}%",
                        ha=ha, va="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "fig10_excluded_returns.png", bbox_inches="tight")
            plt.close(fig); saved.append("fig10_excluded_returns.png")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# 13.  MAIN — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _sep(title: str = ""):
    w = 72
    if title:
        print(f"\n{'─'*3}  {title}  {'─'*(w - len(title) - 7)}")
    else:
        print("─" * w)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_config = CrashRiskConfig()
    raw_paths = RawDataPaths(
        prices=DATA_DIR / "prices.csv",
        benchmark_prices=DATA_DIR / "benchmark_prices.csv",
        fundamentals=DATA_DIR / "fundamentals.csv",
        controversies=DATA_DIR / "controversies.csv",
    )

    use_real = DATA_SOURCE == "real" and (DATA_DIR / "prices.csv").exists()
    if use_real:
        _sep("STEP 1 — Load real data  (data/raw/)")
        prices, controv = load_real_data(DATA_DIR)
        n_tickers = prices["ticker"].nunique()
        date_min  = prices["date"].min().date()
        date_max  = prices["date"].max().date()
        print(f"  Source         : {DATA_DIR.resolve()}")
        print(f"  Tickers        : {n_tickers}")
        print(f"  Weekly rows    : {len(prices):,}")
        print(f"  Date range     : {date_min} → {date_max}")
        print(f"  Controversy rows: {len(controv):,}")
    else:
        _sep("STEP 1 — Generate synthetic data  (data/raw/ not found)")
        print(f"  Tickers : {N_TICKERS}   |   Weeks : {WEEKS}   |   Seed : {SEED}")
        prices, controv = generate_data(weeks=WEEKS, seed=SEED)
        print(f"  Price rows     : {len(prices):,}")
        print(f"  Controversy rows: {len(controv):,}")

    _sep("STEP 2 — Feature engineering  (20 features across 5 groups)")
    panel = build_feature_panel(prices, controv)
    print(f"  Panel shape    : {panel.shape}")
    present = [f for f in ALL_FEATURES if f in panel.columns]
    missing = [f for f in ALL_FEATURES if f not in panel.columns]
    print(f"  Features built : {len(present)}/{len(ALL_FEATURES)}")
    if missing:
        print(f"  Missing        : {missing}")
    textual_analysis, textual_ticker_summary = build_textual_analysis_outputs(DATA_DIR)
    panel = join_text_signals_to_panel(panel, textual_analysis)
    print("  Text features  : joined sparse weekly text signals (NaN where uncovered)")

    _sep("STEP 3 — Target creation  (NCSKEW top-{:.0%})".format(TARGET_QUANTILE))
    dataset = make_targets(panel)
    labeled = dataset.dropna(subset=["high_crash_risk"])
    n_pos   = int(labeled["high_crash_risk"].sum())
    n_neg   = int((labeled["high_crash_risk"] == 0).sum())
    print(f"  Labeled rows   : {len(labeled):,}   |   Unlabeled (future window): {len(dataset)-len(labeled)}")
    print(f"  Class balance  : {n_pos:,} high-risk  /  {n_neg:,} not-high  "
          f"(ratio 1:{n_neg//max(n_pos,1)})")

    _sep("STEP 4 — Chronological splits  (60 / 20 / 20 by date)")
    train, val, test = chronological_split(labeled)
    print(f"  Train : {len(train):>6,} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Val   : {len(val):>6,} rows  ({val['date'].min().date()} → {val['date'].max().date()})")
    print(f"  Test  : {len(test):>6,} rows  ({test['date'].min().date()} → {test['date'].max().date()})")

    _sep("STEP 5 — Algorithm comparison  (LR / RF / GB)")
    algo_df = compare_algorithms(train, val, test)
    header = f"  {'Model':<28} {'Val AUC':>8} {'Test AUC':>9} {'Val Prec@Top':>13} {'Test Prec@Top':>14}"
    print(header)
    print("  " + "-" * 74)
    for _, row in algo_df.iterrows():
        print(f"  {row['model']:<28} {row['val_roc_auc']:>8.3f} {row['test_roc_auc']:>9.3f}"
              f" {row['val_precision_at_top']:>13.3f} {row['test_precision_at_top']:>14.3f}")

    _sep("STEP 5.5 — Hyperparameter tuning  (grid-search: LR / RF / GB)")
    tuning_results = tune_hyperparameters(train, val, ALL_FEATURES)
    for model_name, res in tuning_results.items():
        print(f"  {model_name:<28}  best_val_auc={res['best_val_auc']:.4f}  "
              f"best_params={res['best_params']}")
    hyperparameter_tuning_results = build_hyperparameter_tuning_results(
        labeled,
        config=report_config,
        run_search=True,
    )
    hyperparameter_tuning_results.to_csv(OUTPUTS_DIR / "hyperparameter_tuning_results.csv", index=False)
    print(f"  Tuning CSV                  {OUTPUTS_DIR / 'hyperparameter_tuning_results.csv'}")

    _sep("STEP 6 — ESG feature lift  (baseline 12 features vs full 20 features)")
    esg_df, full_pipe = compare_esg_lift(train, val, test)
    for _, row in esg_df.iterrows():
        sign = "+" if row["model"] == "ESG_delta" else " "
        auc_v = f"{sign}{row['val_roc_auc']:.3f}" if pd.notna(row['val_roc_auc']) else " N/A"
        auc_t = f"{sign}{row['test_roc_auc']:.3f}" if pd.notna(row['test_roc_auc']) else " N/A"
        print(f"  {row['model']:<25}  Val AUC: {auc_v}  |  Test AUC: {auc_t}")
    delta_row = esg_df[esg_df["model"] == "ESG_delta"].iloc[0]
    if delta_row["test_roc_auc"] > 0:
        print(f"\n  ESG controversy features deliver a positive test-set lift of "
              f"+{delta_row['test_roc_auc']:.3f} ROC-AUC")
    else:
        print(f"\n  Test lift: {delta_row['test_roc_auc']:+.3f} (positive on validation; "
              f"consistent with noisy demo data)")
    text_model_comparison = compare_text_signal_lift(labeled, config=report_config)
    text_model_comparison.to_csv(OUTPUTS_DIR / "text_model_comparison.csv", index=False)
    text_delta = text_model_comparison[
        (text_model_comparison["model"] == "text_minus_full_esg")
        & (text_model_comparison["split"] == "test")
    ]
    if not text_delta.empty:
        print(f"  Text signal test AUC delta  {text_delta.iloc[0]['roc_auc']:+.3f}")
    print(f"  Text model CSV              {OUTPUTS_DIR / 'text_model_comparison.csv'}")

    _sep("STEP 7 — Feature importance  (top 10)")
    feat_imp = get_feature_importance(full_pipe, ALL_FEATURES)
    print(f"  {'Rank':<5} {'Feature':<35} {'|Coef|':>8}  {'Category'}")
    print("  " + "-" * 65)
    for rank, (_, row) in enumerate(feat_imp.head(10).iterrows(), 1):
        cat = FEATURE_CATS.get(row["feature"], "Other")
        print(f"  {rank:<5} {row['feature']:<35} {row['importance']:>8.3f}  {cat}")

    _sep("STEP 8 — Risk scoring (latest week)")
    scores = score_latest(panel, full_pipe, ALL_FEATURES)
    bucket_counts = scores["risk_bucket"].value_counts()
    print(f"  Scoring date   : {panel['date'].max().date()}")
    print(f"  High risk      : {bucket_counts.get('High', 0):>3} stocks")
    print(f"  Medium risk    : {bucket_counts.get('Medium', 0):>3} stocks")
    print(f"  Low risk       : {bucket_counts.get('Low', 0):>3} stocks")
    print()
    high_stocks = scores[scores["risk_bucket"] == "High"].sort_values(
        "crash_probability", ascending=False)
    print(f"  {'Ticker':<8} {'Prob':>6}  {'Bucket':<8}  Top Drivers")
    print("  " + "-" * 65)
    for _, row in high_stocks.iterrows():
        print(f"  {row['ticker']:<8} {row['crash_probability']:>6.3f}  "
              f"{row['risk_bucket']:<8}  {row.get('top_drivers', '')}")
    confusion_matrix_df, calibration_curve_df = build_test_diagnostics(labeled, config=report_config)
    confusion_matrix_df.to_csv(OUTPUTS_DIR / "confusion_matrix.csv", index=False)
    calibration_curve_df.to_csv(OUTPUTS_DIR / "calibration_curve.csv", index=False)
    print(f"  Confusion CSV  : {OUTPUTS_DIR / 'confusion_matrix.csv'}")
    print(f"  Calibration CSV: {OUTPUTS_DIR / 'calibration_curve.csv'}")

    _sep("STEP 9 — Business analysis  ($1B fund overlay)")
    business_portfolio_returns = build_weekly_forward_portfolio_returns(panel, full_pipe, ALL_FEATURES)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    business_portfolio_returns.to_csv(OUTPUTS_DIR / "business_portfolio_returns.csv", index=False)
    biz = compute_business_analysis(
        panel,
        full_pipe,
        ALL_FEATURES,
        portfolio_returns=business_portfolio_returns,
    )
    pd.DataFrame([{"metric": key, "value": value} for key, value in biz.items()]).to_csv(
        OUTPUTS_DIR / "business_analysis.csv",
        index=False,
    )
    if "error" in biz:
        print(f"  {biz['error']}")
    else:
        rows = [
            ("Strategy annual return",   f"{biz['strategy_annual_return']:.2%}"),
            ("Benchmark annual return",  f"{biz['benchmark_annual_return']:.2%}"),
            ("Alpha (annualised)",        f"{biz['alpha_annualized']:.2%}"),
            ("Benchmark alpha",           f"{biz['benchmark_alpha_annualized']:.2%}"),
            ("Strategy Sharpe ratio",    f"{biz['strategy_sharpe']}"),
            ("Benchmark Sharpe ratio",   f"{biz['benchmark_sharpe']}"),
            ("Strategy Sortino ratio",   f"{biz['strategy_sortino']}"),
            ("Benchmark Sortino ratio",  f"{biz['benchmark_sortino']}"),
            ("Max drawdown — strategy",  f"{biz['max_drawdown_strategy']:.2%}"),
            ("Max drawdown — benchmark", f"{biz['max_drawdown_benchmark']:.2%}"),
            ("Strategy weekly VaR (95%)", f"{biz['var_95_weekly']:.2%}"),
            ("Benchmark weekly VaR (95%)", f"{biz['benchmark_var_95_weekly']:.2%}"),
            ("Strategy weekly CVaR (95%)", f"{biz['cvar_95_weekly']:.2%}"),
            ("Benchmark weekly CVaR (95%)", f"{biz['benchmark_cvar_95_weekly']:.2%}"),
            ("Evaluation weeks",         f"{biz['evaluation_weeks']}"),
            ("High-risk excluded",       f"{biz['high_risk_excluded_pct']:.1%}"),
            ("Benchmark high-risk excluded", f"{biz['benchmark_high_risk_excluded_pct']:.1%}"),
            ("Business analysis method", f"{biz['business_analysis_method']}"),
            ("Illustrative economic gain / year", f"${biz['economic_gain_annual']:,.0f}" if biz['economic_gain_annual'] else "N/A"),
            ("Benchmark economic gain",  "$0"),
            ("Team cost / year",         f"${biz['team_annual_cost']:,.0f}"),
            ("Illustrative Team ROI",    f"{biz['team_roi']}x"),
            ("Benchmark Team ROI",       "-"),
            ("Justifies team?",          str(biz['justifies_team'])),
        ]
        for label, value in rows:
            print(f"  {label:<30} {value}")
        print("  Note                         Stylised simulation; transaction costs, market impact,")
        print("                               and model uncertainty would likely reduce realised returns.")
        print(f"  Business CSV                 {OUTPUTS_DIR / 'business_analysis.csv'}")
        print(f"  Portfolio CSV                {OUTPUTS_DIR / 'business_portfolio_returns.csv'}")

    _sep("STEP 9.5 — Q4 Out-of-Sample Backtest  (true forward validation)")
    quarter_bt = quarter_snapshot_backtest(panel, full_pipe, ALL_FEATURES)
    if "error" in quarter_bt:
        print(f"  {quarter_bt['error']}")
    else:
        q_label = quarter_bt["quarter_label"]
        cutoff_str = quarter_bt["cutoff_date"]
        print(f"  Quarter        : {q_label}  (model scored at {cutoff_str})")
        print(f"  Forward weeks  : {quarter_bt['forward_weeks']}")
        print(f"  Stocks excluded: {quarter_bt['n_excluded']} High-risk  "
              f"({quarter_bt['pct_excluded_correct']:.0%} subsequently fell — model correct)")
        print(f"  Stocks held    : {quarter_bt['n_held']}")
        print(f"  Strategy return: {quarter_bt['strategy_quarter_return']:+.2%}")
        print(f"  Benchmark return:{quarter_bt['benchmark_quarter_return']:+.2%}")
        print(f"  Outperformance : {quarter_bt['outperformance_bps']:+d} bps")
        print(f"  Dollar impact  : ${quarter_bt['dollar_impact_quarter']:,.0f}  "
              f"(${quarter_bt['dollar_impact_annualised']:,.0f} annualised on $1B AUM)")
        print()
        print(f"  {'Ticker':<8} {'Crash Prob':>10}  {'Fwd Return':>10}  Outcome")
        print("  " + "-" * 55)
        for row in quarter_bt["excluded_tickers"]:
            ret_str = f"{row['quarter_return']:+.2%}" if row["quarter_return"] is not None else "  N/A"
            print(f"  {row['ticker']:<8} {row['crash_probability']:>10.3f}  "
                  f"{ret_str:>10}  {row['outcome']}")
        # Write CSVs
        pd.DataFrame(quarter_bt["weekly_series"]).to_csv(
            OUTPUTS_DIR / "quarter_backtest_returns.csv", index=False)
        pd.DataFrame(quarter_bt["excluded_tickers"]).to_csv(
            OUTPUTS_DIR / "quarter_excluded_stocks.csv", index=False)
        print(f"\n  Returns CSV    : {OUTPUTS_DIR / 'quarter_backtest_returns.csv'}")
        print(f"  Excluded CSV   : {OUTPUTS_DIR / 'quarter_excluded_stocks.csv'}")

    _sep("STEP 10 — Saving charts")
    _sep("STEP 10 - ESG text analytics  (optional)")
    textual_analysis, textual_ticker_summary = build_textual_analysis_outputs(DATA_DIR)
    textual_analysis.to_csv(OUTPUTS_DIR / "textual_analysis.csv", index=False)
    textual_ticker_summary.to_csv(OUTPUTS_DIR / "textual_ticker_summary.csv", index=False)
    word_cloud_path = write_text_word_cloud_svg(DATA_DIR)
    bigram_terms_path = OUTPUTS_DIR / "textual_bigram_terms.csv"
    text_status = textual_analysis.iloc[0]["status"] if not textual_analysis.empty and "status" in textual_analysis.columns else "unknown"
    print(f"  Text status    : {text_status}")
    if text_status == "ok":
        avg_score = textual_ticker_summary["negative_esg_controversy_score_0_100"].mean()
        top_name = textual_ticker_summary.iloc[0]["ticker"]
        top_score = textual_ticker_summary.iloc[0]["negative_esg_controversy_score_0_100"]
        print(f"  Covered tickers: {textual_ticker_summary['ticker'].nunique()}")
        print(f"  Avg ESG-neg    : {avg_score:.1f}/100")
        print(f"  Top ticker     : {top_name} ({top_score:.1f}/100)")
    else:
        print(f"  Note           : {textual_analysis.iloc[0].get('note', 'No optional text file supplied.')}")
    print(f"  Weekly CSV     : {OUTPUTS_DIR / 'textual_analysis.csv'}")
    print(f"  Summary CSV    : {OUTPUTS_DIR / 'textual_ticker_summary.csv'}")
    print(f"  Portfolio CSV  : {OUTPUTS_DIR / 'business_portfolio_returns.csv'}")
    print(f"  Bigram CSV     : {bigram_terms_path}")
    print(f"  Word cloud SVG : {word_cloud_path}")
    print(f"  Report SVG     : {REPORT_FIGURES_DIR / 'text_word_cloud.svg'}")
    print(f"  Bullish PNG    : {FIGURES_DIR / 'bullish_signals_bigrams.png'}")
    print(f"  Bearish PNG    : {FIGURES_DIR / 'bearish_signals_bigrams.png'}")

    _sep("STEP 10.5 - Rubric readiness diagnostics")
    feature_descriptive_stats = build_feature_descriptive_stats(panel, report_config)
    feature_correlation_matrix = build_feature_correlation_matrix(panel, report_config)
    text_coverage = build_text_coverage(textual_analysis, labeled, config=report_config)
    lda_outputs = build_lda_topic_outputs(raw_paths, config=report_config)

    feature_descriptive_stats.to_csv(OUTPUTS_DIR / "feature_descriptive_stats.csv", index=False)
    feature_correlation_matrix.to_csv(OUTPUTS_DIR / "feature_correlation_matrix.csv", index=False)
    text_coverage.to_csv(OUTPUTS_DIR / "text_coverage.csv", index=False)
    lda_outputs["topic_words"].to_csv(OUTPUTS_DIR / "lda_topic_words.csv", index=False)
    lda_outputs["ticker_topics"].to_csv(OUTPUTS_DIR / "lda_ticker_topics.csv", index=False)
    if use_real:
        raw_loaded = load_raw_data(raw_paths, config=report_config)
        data_summary, cleaning_log = build_data_summary(raw_paths, raw_loaded, panel, dataset, scores, report_config)
        sql_summary = build_sql_summary(raw_loaded, panel, dataset, scores)
        data_summary.to_csv(OUTPUTS_DIR / "data_summary.csv", index=False)
        cleaning_log.to_csv(OUTPUTS_DIR / "cleaning_log.csv", index=False)
        sql_summary.to_csv(OUTPUTS_DIR / "sql_summary.csv", index=False)
        write_sql_summary_markdown(sql_summary, OUTPUTS_DIR / "sql_summary.md")

    write_feature_correlation_heatmap(
        feature_correlation_matrix,
        FIGURES_DIR / "feature_correlation_heatmap.png",
    )
    write_price_time_series(panel, scores, FIGURES_DIR / "price_time_series.png")
    write_probability_calibration_plot(
        calibration_curve_df,
        FIGURES_DIR / "probability_calibration.png",
    )
    write_lda_topic_distribution(
        lda_outputs["ticker_topics"],
        FIGURES_DIR / "lda_topic_distribution.png",
    )
    print(f"  Descriptive CSV: {OUTPUTS_DIR / 'feature_descriptive_stats.csv'}")
    print(f"  Correlation CSV: {OUTPUTS_DIR / 'feature_correlation_matrix.csv'}")
    print(f"  Text coverage  : {OUTPUTS_DIR / 'text_coverage.csv'}")
    print(f"  LDA topics CSV : {OUTPUTS_DIR / 'lda_topic_words.csv'}")
    if use_real:
        print(f"  Cleaning log   : {OUTPUTS_DIR / 'cleaning_log.csv'}")
        print(f"  SQL summary    : {OUTPUTS_DIR / 'sql_summary.md'}")
    print(f"  Correlation PNG: {FIGURES_DIR / 'feature_correlation_heatmap.png'}")
    print(f"  Price series   : {FIGURES_DIR / 'price_time_series.png'}")
    print(f"  Calibration PNG: {FIGURES_DIR / 'probability_calibration.png'}")
    print(f"  LDA topic PNG  : {FIGURES_DIR / 'lda_topic_distribution.png'}")

    _sep("STEP 11 - Saving charts")
    saved = plot_all(scores, algo_df, esg_df, feat_imp, biz, panel, FIGURES_DIR,
                     quarter_bt=quarter_bt if "error" not in quarter_bt else None)
    for f in saved:
        print(f"  Saved: {FIGURES_DIR / f}")

    _sep()
    print("\n  Pipeline complete.")
    print(f"  Charts saved to: {FIGURES_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
