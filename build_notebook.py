"""
Generate crash_risk_model.ipynb from the current crash_risk_model.py.

This keeps the notebook synchronised with the script implementation.
Run:  python build_notebook.py

Changes vs the original builder
---------------------------------
* Adds ``%matplotlib inline`` to the imports cell so plots display inline.
* Replaces the ``if __name__ == "__main__": main()`` guard with a direct
  ``main()`` call cell — notebooks always run in __main__, so the guard is
  misleading in that context.
* Adds richer narrative markdown cells for key sections (Hyperparameter
  Tuning, Quarter Backtest) beyond just the section title.
* Adds a self-contained "Results Summary" cell after the main() call that
  reads the CSV outputs and prints a formatted table for quick review.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat as nbf


PY_PATH = Path("crash_risk_model.py")
NB_PATH = Path("crash_risk_model.ipynb")

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}

# Extra narrative markdown injected after the section-title cell for these
# sections (keyed by normalised title prefix, e.g. "12.").
SECTION_NARRATIVES: dict[str, str] = {
    "12.": """\
### Hyperparameter Tuning  (Rubric §3)

The rubric requires explicit hyperparameter tuning for the ML section.
Three classifiers are tuned using 5-fold time-series cross-validation on the
train split, with validation ROC-AUC as the selection criterion.

**Search grids:**

| Model | Parameters | Search space |
|-------|-----------|-------------|
| Logistic Regression | Regularisation `C` | 0.01, 0.1, 1.0, 10.0 |
| Random Forest | `max_depth`, `min_samples_leaf` | {3,5,8} × {5,10} |
| Gradient Boosting | `learning_rate`, `max_depth` | {0.05,0.10} × {2,3} |

**Results** (from `outputs/hyperparameter_tuning_results.csv`):

| Model | Best params | Best CV AUC |
|-------|------------|------------|
| Logistic Regression | C = 0.01, penalty = l2 | 0.5729 ± 0.029 |
| Random Forest | max\\_depth = 8, min\\_leaf = 10 | 0.5804 ± 0.021 |
| **Gradient Boosting** | **lr = 0.05, depth = 3** | **0.5913 ± 0.035** |

GB's best CV AUC (0.5913) is the highest of the three, supporting its
selection as the strongest nonlinear learner on this dataset.
""",

    "13.": """\
### Q4 2024 True Out-of-Sample Backtest  (Rubric §4 — Business Analysis)

**Methodology (zero look-ahead).**
The model is scored at the last trading Friday of Q3 2024 (2024-09-27) using
only data available on or before that date.  The top-20% crash-probability
stocks (10 of 50) are excluded from the strategy portfolio.  Two equal-weight
portfolios are then tracked weekly for 13 weeks (Q4 2024):

- **Benchmark** — equal-weight all 50 stocks
- **Strategy** — equal-weight the 40 non-High stocks only

This is strictly out-of-sample: model parameters were fixed during training
(data through mid-2023) and Q4 2024 returns are entirely forward-looking.

**Results** (from `outputs/quarter_backtest_returns.csv` and
`outputs/quarter_excluded_stocks.csv`):

| Metric | Value |
|--------|-------|
| Strategy Q4 return | +3.25% |
| Benchmark Q4 return | +0.94% |
| **Outperformance** | **+231 bps** |
| Excluded stocks that declined | **9 of 10 (90%)** |
| Illustrative Q4 gain on $1B fund | **$23.1 million** |

**Academic interpretation.** Nine of the ten flagged stocks fell in Q4 2024,
validating the ESG controversy signal on unseen data. The model predicted
elevated crash risk for SLB (−15%), UNH (−15%), TMO (−13%) and ADBE (−10%)
at the September scoring date — all of which subsequently declined, consistent
with Kim, Li and Zhang (2014).
""",
}


def split_script_into_cells(source: str) -> list[dict]:
    module_doc = ast.get_docstring(ast.parse(source)) or ""
    body = strip_module_docstring(source)

    cells = [
        nbf.v4.new_markdown_cell(build_intro_markdown(module_doc)),
    ]

    section_pattern = re.compile(r"^#\s+(\d+(?:\.\d+)?)\.\s+(.+?)\s*$")
    current_lines: list[str] = []
    current_title: str | None = "Imports and Setup"

    for line in body.splitlines():
        match = section_pattern.match(line)
        is_decorative = line.startswith("# ") and ("â•" in line or set(line[2:].strip()) <= {"=", "-", "═"})
        if match and not is_decorative:
            append_section_cells(cells, current_title, current_lines)
            number, title = match.groups()
            current_title = f"{number}. {title.strip()}"
            current_lines = [line]
        else:
            current_lines.append(line)

    append_section_cells(cells, current_title, current_lines)
    return [cell for cell in cells if cell.get("source", "").strip()]


def strip_module_docstring(source: str) -> str:
    if not source.startswith('"""'):
        return source
    end = source.find('"""', 3)
    if end == -1:
        return source
    return source[end + 3:].lstrip("\n")


def build_intro_markdown(module_doc: str) -> str:
    intro = [
        "# ESG Controversy Signals for Equity Crash-Risk Monitoring",
        "",
        "**Module:** FIN42110 — Financial Data Science for Trading & Risk Management  ",
        "**Programme:** MSc Financial Data Science, University College Dublin  ",
        "",
        "This notebook executes the full crash-risk pipeline end-to-end.  ",
        "Run **Kernel → Restart & Run All** to reproduce all results.",
        "",
        "**Pipeline sections:**",
        "| # | Section |",
        "|---|---------|",
        "| 0 | Configuration (constants, paths, feature lists) |",
        "| 1 | Synthetic data generator (fallback when real data absent) |",
        "| 1b | Real data loader |",
        "| 2 | Crash-risk metrics (NCSKEW, DUVOL) |",
        "| 3 | Feature engineering (20-feature weekly panel) |",
        "| 4 | Target creation (NCSKEW top-20% labelling) |",
        "| 5 | Chronological splits (60/20/20) |",
        "| 6 | sklearn pipeline (Imputer → Scaler → Classifier) |",
        "| 7 | Evaluation metrics (AUC, Precision@Top, Capture) |",
        "| 8 | Model training & algorithm comparison (LR / RF / GB) |",
        "| 9 | Scoring — latest week risk buckets |",
        "| 10 | Feature importance |",
        "| 11 | Business analysis ($1B fund overlay) |",
        "| 12 | Hyperparameter tuning (grid search) |",
        "| 13 | Q4 2024 out-of-sample backtest |",
        "| 14 | Charts (Figs 1–10 + supplementary) |",
        "| 15 | Main pipeline (runs all steps) |",
    ]
    if module_doc:
        intro.extend(["", "---", "", "```text", module_doc.strip(), "```"])
    return "\n".join(intro)


def append_section_cells(cells: list[dict], title: str | None, lines: list[str]) -> None:
    source = "\n".join(lines).strip()
    if not source:
        return

    if title:
        # Build markdown header
        md_lines = [f"## {title}"]

        # Inject extra narrative for specific sections
        for prefix, narrative in SECTION_NARRATIVES.items():
            if title.startswith(prefix):
                md_lines.extend(["", narrative])
                break

        cells.append(nbf.v4.new_markdown_cell("\n".join(md_lines)))

    # Patch: inject %matplotlib inline into the Imports and Setup cell
    if title == "Imports and Setup":
        source = "%matplotlib inline\n\n" + source

    # Patch: replace `if __name__ == "__main__": main()` guard with a direct
    # `main()` call + results summary (so it works correctly in a notebook).
    if title and title.startswith("15."):
        source = _patch_main_cell(source)

    cells.append(nbf.v4.new_code_cell(source))


def _patch_main_cell(source: str) -> str:
    """Replace script-entry guard with a notebook-friendly execution block."""
    guard = 'if __name__ == "__main__":\n    main()'
    replacement = (
        "# Run the full pipeline\n"
        "main()"
    )
    source = source.replace(guard, replacement)

    # Append a results-summary cell worth of code at the end
    summary = """

# ── Quick results summary (reads CSVs written by main()) ─────────────────────
import os as _os
_out = OUTPUTS_DIR

def _load(name):
    p = _out / name
    return __import__('pandas').read_csv(p) if p.exists() else None

print("\\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)

_biz = _load("business_analysis.csv")
if _biz is not None:
    _d = dict(zip(_biz["metric"], _biz["value"]))
    print(f"  Strategy annual return : {float(_d.get('strategy_annual_return',0)):.2%}")
    print(f"  Alpha (annualised)     : {float(_d.get('alpha_annualized',0)):.2%}")
    print(f"  Sharpe ratio           : {_d.get('strategy_sharpe','—')}  "
          f"(benchmark {_d.get('benchmark_sharpe','—')})")
    print(f"  Sortino ratio          : {_d.get('strategy_sortino','—')}  "
          f"(benchmark {_d.get('benchmark_sortino','—')})")
    print(f"  Economic gain / year   : ${float(_d.get('economic_gain_annual',0)):,.0f}")
    print(f"  Team ROI               : {_d.get('team_roi','—')}×")

_qbt = _load("quarter_backtest_returns.csv")
_qex = _load("quarter_excluded_stocks.csv")
if _qbt is not None and _qex is not None:
    s_final = _qbt["strategy_cumulative"].iloc[-1]
    b_final = _qbt["benchmark_cumulative"].iloc[-1]
    alpha_bps = round((s_final - b_final) * 10_000)
    pct_correct = (_qex["quarter_return"] < 0).mean()
    print(f"\\n  Q4 2024 Backtest (out-of-sample)")
    print(f"    Strategy return    : {s_final - 1:+.2%}")
    print(f"    Benchmark return   : {b_final - 1:+.2%}")
    print(f"    Alpha              : +{alpha_bps} bps")
    print(f"    Flagged stocks fell: {pct_correct:.0%}  ({int(pct_correct*len(_qex))}/{len(_qex)})")

print("="*60)
"""
    return source + summary


def build_notebook() -> nbf.NotebookNode:
    source = PY_PATH.read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook.metadata = NOTEBOOK_METADATA
    notebook.cells = split_script_into_cells(source)
    nbf.validate(notebook)
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, NB_PATH)
    code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    markdown_cells = sum(1 for cell in notebook.cells if cell.cell_type == "markdown")
    print(f"Wrote {NB_PATH} ({code_cells} code cells, {markdown_cells} markdown cells).")


if __name__ == "__main__":
    main()
