"""
Generate PNG charts from outputs/ CSVs and build the final docx report.
Run: python reports/generate_report_charts.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent          # project root
OUT  = BASE / "outputs"
IMG  = BASE / "reports" / "figures"
IMG.mkdir(parents=True, exist_ok=True)

TEAL   = "#0d6b62"
AMBER  = "#e07b00"
RED    = "#c0392b"
GREEN  = "#1a7a4a"
GREY   = "#7f8c8d"
LBLUE  = "#2980b9"
PURPLE = "#8e44ad"
BG     = "#f7f9fa"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  10,
    "figure.dpi":       150,
})


# ── 1. Crash-risk probability ranking ─────────────────────────────────────────
def chart_risk_ranking():
    df = pd.read_csv(OUT / "stock_scores.csv")
    df = df.sort_values("crash_probability", ascending=True)
    colours = df["risk_bucket"].map({"High": RED, "Medium": AMBER, "Low": GREEN}).fillna(GREY)

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(df["ticker"], df["crash_probability"], color=colours, edgecolor="white", linewidth=0.4)
    ax.axvline(0.20, color=GREY,  linestyle=":", linewidth=1.2, label="Low/Medium boundary (20%)")
    ax.axvline(0.50, color=AMBER, linestyle=":", linewidth=1.2, label="Medium/High boundary (50%)")
    ax.set_xlabel("Crash Probability")
    ax.set_title("Fig 1  |  Crash-Risk Probability Ranking (all stocks, latest week)")
    legend_patches = [
        mpatches.Patch(color=RED,   label="High risk"),
        mpatches.Patch(color=AMBER, label="Medium risk"),
        mpatches.Patch(color=GREEN, label="Low risk"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(IMG / "fig1_risk_ranking.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig1_risk_ranking.png")


# ── 2. Sector controversy bar chart ──────────────────────────────────────────
def chart_sector_controversy():
    sector_data = {
        "Energy":                  5.93,
        "Financial Services":      5.77,
        "Industrials":             5.30,
        "Communication Services":  4.88,
        "Consumer Cyclical":       4.62,
        "Technology":              4.36,
        "Basic Materials":         4.32,
        "Healthcare":              4.23,
        "Consumer Defensive":      2.63,
        "Utilities":               2.54,
    }
    sectors = list(sector_data.keys())
    scores  = list(sector_data.values())
    colours = [RED if s >= 5.5 else AMBER if s >= 4.0 else GREEN for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sectors, scores, color=colours, edgecolor="white", linewidth=0.4)
    ax.axvline(5.0, color=GREY, linestyle=":", linewidth=1.2, label="Score = 5")
    ax.set_xlabel("Average ESG Controversy Score (0–10)")
    ax.set_title("Fig 2  |  Average ESG Controversy Score by Sector")
    ax.set_xlim(0, 7)
    for i, v in enumerate(scores):
        ax.text(v + 0.08, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "fig2_sector_controversy.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig2_sector_controversy.png")


# ── 3. Algorithm comparison ───────────────────────────────────────────────────
def chart_algorithm_comparison():
    df = pd.read_csv(OUT / "algorithm_comparison.csv")
    models = ["logistic_regression", "random_forest", "gradient_boosting"]
    labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
    colours_val  = [TEAL, TEAL, TEAL]
    colours_test = [LBLUE, LBLUE, LBLUE]

    val_auc  = [df.loc[(df.model==m)&(df.split=="validation"), "roc_auc"].values[0] for m in models]
    test_auc = [df.loc[(df.model==m)&(df.split=="test"),       "roc_auc"].values[0] for m in models]
    val_prec  = [df.loc[(df.model==m)&(df.split=="validation"), "precision_at_top_bucket"].values[0] for m in models]
    test_prec = [df.loc[(df.model==m)&(df.split=="test"),       "precision_at_top_bucket"].values[0] for m in models]

    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC-AUC
    ax1.bar(x - w/2, val_auc,  w, label="Validation", color=TEAL,  alpha=0.85)
    ax1.bar(x + w/2, test_auc, w, label="Test",       color=LBLUE, alpha=0.85)
    ax1.axhline(0.5, color=GREY, linestyle=":", linewidth=1.2, label="Random baseline")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(0.45, 0.65)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("Fig 3a  |  ROC-AUC by Algorithm")
    ax1.legend()
    for i, (v, t) in enumerate(zip(val_auc, test_auc)):
        ax1.text(i-w/2, v+0.002, f"{v:.3f}", ha="center", fontsize=8)
        ax1.text(i+w/2, t+0.002, f"{t:.3f}", ha="center", fontsize=8)

    # Precision@Top
    ax2.bar(x - w/2, val_prec,  w, label="Validation", color=TEAL,  alpha=0.85)
    ax2.bar(x + w/2, test_prec, w, label="Test",       color=LBLUE, alpha=0.85)
    ax2.axhline(0.20, color=GREY, linestyle=":", linewidth=1.2, label="Naive baseline (20%)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0.15, 0.35)
    ax2.set_ylabel("Precision @ Top Bucket")
    ax2.set_title("Fig 3b  |  Precision@Top by Algorithm")
    ax2.legend()
    for i, (v, t) in enumerate(zip(val_prec, test_prec)):
        ax2.text(i-w/2, v+0.002, f"{v:.3f}", ha="center", fontsize=8)
        ax2.text(i+w/2, t+0.002, f"{t:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(IMG / "fig3_algorithm_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3_algorithm_comparison.png")


# ── 4. ESG lift comparison ────────────────────────────────────────────────────
def chart_esg_lift():
    data = {
        "Baseline\n(no ESG)": {"val": 0.558, "test": 0.569},
        "Full\n(with ESG)":   {"val": 0.600, "test": 0.598},
    }
    labels = list(data.keys())
    val_scores  = [data[l]["val"]  for l in labels]
    test_scores = [data[l]["test"] for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2, val_scores,  w, label="Validation", color=TEAL,  alpha=0.85)
    ax.bar(x + w/2, test_scores, w, label="Test",       color=LBLUE, alpha=0.85)
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1.2, label="Random baseline")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.46, 0.62)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Fig 4  |  ESG Feature Lift — Baseline vs Full Model")
    ax.legend()

    # annotate delta arrows
    for i, (v, t) in enumerate(zip(val_scores, test_scores)):
        ax.text(i-w/2, v+0.002, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i+w/2, t+0.002, f"{t:.3f}", ha="center", fontsize=9, fontweight="bold")

    # lift annotations
    ax.annotate("", xy=(0.5-w/2, 0.603), xytext=(0.5-w/2-0.01, 0.560),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.text(0.56, 0.583, "+4.2 pp\nval lift", color=RED, fontsize=9, fontweight="bold")
    ax.annotate("", xy=(0.5+w/2, 0.601), xytext=(0.5+w/2+0.01, 0.571),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.text(0.92, 0.585, "+2.9 pp\ntest lift", color=RED, fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(IMG / "fig4_esg_lift.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig4_esg_lift.png")


# ── 5. Feature importance ─────────────────────────────────────────────────────
def chart_feature_importance():
    df = pd.read_csv(OUT / "feature_importance.csv").sort_values("importance", ascending=True)
    category_map = {
        "controversy_score":              "ESG",
        "controversy_rolling_mean_13w":   "ESG",
        "controversy_change_13w":         "ESG",
        "controversy_sector_percentile":  "ESG",
        "controversy_rolling_std_13w":    "ESG",
        "controversy_change_4w":          "ESG",
        "controversy_spike_flag":         "ESG",
        "controversy_change_26w":         "ESG",
        "downside_beta":                  "Downside Risk",
        "relative_downside_beta":         "Downside Risk",
        "trailing_return":                "Downside Risk",
        "beta":                           "Downside Risk",
        "realized_volatility":            "Downside Risk",
        "market_cap":                     "Fundamentals",
        "leverage":                       "Fundamentals",
        "roa":                            "Fundamentals",
        "market_to_book":                 "Fundamentals",
        "lagged_ncskew":                  "Crash History",
        "lagged_duvol":                   "Crash History",
        "detrended_turnover":             "Trading Activity",
    }
    cat_colours = {
        "ESG":              TEAL,
        "Downside Risk":    AMBER,
        "Fundamentals":     LBLUE,
        "Crash History":    PURPLE,
        "Trading Activity": GREY,
    }
    df["category"] = df["feature"].map(category_map).fillna("Other")
    df["colour"]   = df["category"].map(cat_colours)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["feature"], df["importance"], color=df["colour"], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("|Coefficient| (standardised features)")
    ax.set_title("Fig 5  |  Feature Importance — Logistic Regression Coefficients")

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colours.items()]
    ax.legend(handles=legend_patches, loc="lower right")

    for i, row in enumerate(df.itertuples()):
        ax.text(row.importance + 0.01, i, f"{row.importance:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, df["importance"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(IMG / "fig5_feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig5_feature_importance.png")


# ── 6. Business analysis — strategy vs benchmark ──────────────────────────────
def chart_business_metrics():
    metrics = {
        "Annual Return (%)":    (23.29, 17.32),
        "Sharpe Ratio":         (1.182,  0.872),
        "Sortino Ratio":        (1.910,  1.371),
        "Max Drawdown (%)":     (-11.98, -12.25),
    }
    labels   = list(metrics.keys())
    strategy = [v[0] for v in metrics.values()]
    bench    = [v[1] for v in metrics.values()]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, strategy, w, label="Strategy (no High risk)", color=TEAL,  alpha=0.9)
    ax.bar(x + w/2, bench,    w, label="Benchmark (all stocks)",  color=GREY,  alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Fig 6  |  Strategy vs Benchmark — Key Performance Metrics")
    ax.legend()

    for i, (s, b) in enumerate(zip(strategy, bench)):
        offset = 0.15 if s >= 0 else -0.35
        ax.text(i-w/2, s + (0.05 if s >= 0 else -0.35), f"{s}", ha="center", fontsize=9, fontweight="bold", color=TEAL)
        ax.text(i+w/2, b + (0.05 if b >= 0 else -0.35), f"{b}", ha="center", fontsize=9, color=GREY)

    fig.tight_layout()
    fig.savefig(IMG / "fig6_business_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig6_business_metrics.png")


# ── 7. Price scenarios (fan chart) ────────────────────────────────────────────
def chart_price_scenarios():
    df = pd.read_csv(OUT / "price_scenarios.csv")
    # Pick top-5 by risk: highest crash_probability if present, else first 5 tickers
    if "crash_probability" in df.columns:
        top_tickers = (
            df[["ticker","crash_probability"]].drop_duplicates()
            .nlargest(5, "crash_probability")["ticker"].tolist()
        )
    else:
        top_tickers = df["ticker"].unique()[:5].tolist()

    weeks = sorted(df["week"].unique()) if "week" in df.columns else list(range(14))

    fig, ax = plt.subplots(figsize=(10, 5))

    colour_cycle = [RED, AMBER, LBLUE, PURPLE, TEAL]
    for idx, ticker in enumerate(top_tickers):
        sub = df[df["ticker"] == ticker]
        if sub.empty:
            continue
        col = colour_cycle[idx % len(colour_cycle)]

        # Try to get bear/base/bull columns
        if {"bear_price","base_price","bull_price"}.issubset(sub.columns):
            w = sub["week"]
            ax.fill_between(w, sub["bear_price"], sub["bull_price"], alpha=0.12, color=col)
            ax.plot(w, sub["base_price"], color=col, linewidth=2, label=ticker)
            ax.plot(w, sub["bear_price"], color=col, linewidth=0.8, linestyle="--")
            ax.plot(w, sub["bull_price"], color=col, linewidth=0.8, linestyle="--")
        elif "price" in sub.columns:
            ax.plot(sub["week"], sub["price"], color=col, linewidth=2, label=ticker)

    ax.set_xlabel("Weeks Ahead")
    ax.set_ylabel("Price (indexed / absolute)")
    ax.set_title("Fig 7  |  13-Week Price Scenario Range — Top-5 High-Risk Stocks")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "fig7_price_scenarios.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig7_price_scenarios.png")


# ── 8. Economic value waterfall ───────────────────────────────────────────────
def chart_economic_value():
    categories = ["Illustrative Gross\nAlpha ($1B AUM)", "Est. Transaction\nCosts", "Illustrative Net\nGain"]
    values     = [59.7, -5.0, 54.7]   # $M
    colours    = [GREEN, RED, TEAL]
    running    = [0, 59.7, 59.7]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cat, val, base) in enumerate(zip(categories, values, running)):
        colour = GREEN if val >= 0 else RED
        bottom = base if val < 0 else base
        ax.bar(i, abs(val), bottom=(base if val >= 0 else base + val),
               color=colour, width=0.5, alpha=0.85, edgecolor="white")
        sign = "+" if val >= 0 else ""
        ax.text(i, base + val + (0.5 if val >= 0 else -1.5),
                f"${sign}{val}M", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0.8, color=GREY, linewidth=0.8, linestyle=":")
    ax.text(2.35, 0.8, f"Team cost: $0.8M", color=GREY, fontsize=9, va="bottom")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Annual Value ($M)")
    ax.set_title("Fig 8  |  Illustrative Economic Value — $1B Fund Overlay")
    ax.set_ylim(-2, 70)
    fig.tight_layout()
    fig.savefig(IMG / "fig8_economic_value.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig8_economic_value.png")


# ── Build markdown with embedded image links ──────────────────────────────────
def build_markdown_with_figures():
    source_md = Path(__file__).parent / "fds_project_report_full.md"
    dest_md   = Path(__file__).parent / "fds_project_report_with_figures.md"
    text = source_md.read_text(encoding="utf-8")

    figure_inserts = {
        "### 3.6 Algorithm Comparison Results": (
            "\n\n![Fig 3 — Algorithm Comparison](figures/fig3_algorithm_comparison.png)\n\n"
        ),
        "### 3.7 ESG Feature Lift Analysis": (
            "\n\n![Fig 4 — ESG Feature Lift](figures/fig4_esg_lift.png)\n\n"
        ),
        "### 3.8 Feature Importance": (
            "\n\n![Fig 5 — Feature Importance](figures/fig5_feature_importance.png)\n\n"
        ),
        "### 4.3 Results": (
            "\n\n![Fig 6 — Strategy vs Benchmark](figures/fig6_business_metrics.png)\n\n"
        ),
        "### 4.4 Economic Value": (
            "\n\n![Fig 8 — Economic Value](figures/fig8_economic_value.png)\n\n"
        ),
    }

    # Insert after the matching heading line
    for heading, img_md in figure_inserts.items():
        # Find the heading and insert image after its line
        text = text.replace(
            heading + "\n",
            heading + img_md,
            1,
        )

    # Insert Fig 1 after the SQL section, before Section 2
    text = text.replace(
        "## 2. Textual Analysis",
        "![Fig 1 — Crash-Risk Ranking](figures/fig1_risk_ranking.png)\n\n"
        "![Fig 2 — Sector Controversy](figures/fig2_sector_controversy.png)\n\n"
        "---\n\n## 2. Textual Analysis",
        1,
    )

    # Insert Fig 7 before caveats
    text = text.replace(
        "### 4.5 Caveats",
        "![Fig 7 — Price Scenarios](figures/fig7_price_scenarios.png)\n\n"
        "### 4.5 Caveats",
        1,
    )

    dest_md.write_text(text, encoding="utf-8")
    return dest_md


# ── Convert to docx via pandoc ────────────────────────────────────────────────
def build_docx(md_path: Path):
    docx_path = md_path.parent / "fds_project_report_final.docx"
    if shutil.which("pandoc") is None:
        print("\n  ! pandoc not found; skipped docx conversion.")
        return None
    result = subprocess.run(
        [
            "pandoc", str(md_path),
            "-o", str(docx_path),
            "--from", "markdown",
            "--to", "docx",
            "--resource-path", str(md_path.parent),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("pandoc stderr:", result.stderr)
        raise RuntimeError("pandoc conversion failed")
    print(f"\n  ✓ {docx_path}")
    return docx_path


if __name__ == "__main__":
    print("Generating charts...")
    chart_risk_ranking()
    chart_sector_controversy()
    chart_algorithm_comparison()
    chart_esg_lift()
    chart_feature_importance()
    chart_business_metrics()
    chart_price_scenarios()
    chart_economic_value()

    print("\nBuilding markdown with figure links...")
    md_with_figs = build_markdown_with_figures()

    print("Converting to docx...")
    build_docx(md_with_figs)

    print("\nDone! Report saved to reports/fds_project_report_full.docx")
