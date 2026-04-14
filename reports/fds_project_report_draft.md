# ESG Controversy Signals for Equity Crash-Risk Monitoring

## 1. Data Summary and Visualisations

This project is an equity risk-management application, not a pure return-forecasting product. It uses Bloomberg-style stock prices, market benchmark prices, firm fundamentals, and ESG controversy data to predict future crash-risk classification.

Report artifacts to use:

- `outputs/data_summary.csv` for row counts, ticker counts, date ranges, sector coverage, and model-row counts.
- `outputs/cleaning_log.csv` for validation and cleaning details, including invalid rows, duplicate checks, date alignment, and the 45-day fundamentals lag.
- `outputs/sql_summary.md` for SQL queries and results that describe the dataset.
- `outputs/figures/` for dashboard-ready visualisations.

Recommended report figures:

- Crash-risk probability ranking.
- Average controversy score by sector.
- Average controversy score over time.
- Feature importance.
- ESG controversy model lift.
- 13-week price scenario range.

## 2. Textual Analysis

The current ESG signal is `controversy_score`, which should be described as a vendor ESG controversy signal. If Bloomberg/news headline or controversy-description text is available, place it in the raw data folder as `news_text.csv` or `controversy_text.csv`.

Expected optional text columns:

- Required: `ticker`, `date`
- Text field: at least one of `headline`, `title`, `description`, `body`, `text`, or `summary`
- Optional: `source`

The pipeline writes `outputs/textual_analysis.csv`. When text is available, it contains article counts, simple sentiment, negative word counts, controversy keyword counts, and 13-week rolling sentiment. If no text file is supplied, the output explicitly states the limitation. The pipeline also writes `outputs/figures/text_word_cloud.svg`; without raw text it becomes a placeholder stating that direct text was not supplied.

Report language to use if no raw text is available:

> The controversy score is treated as a vendor text-derived ESG controversy signal. Direct news-headline sentiment analysis is a limitation and would be the next extension once Bloomberg headline/description text is available.

## 3. Machine Learning

The modeling target is `high_crash_risk = 1`, defined as the stock being in the top 20% future crash-risk bucket based on 13-week future NCSKEW.

Modeling approach:

- Primary model: Logistic Regression with balanced class weights for interpretability.
- Benchmark comparison: baseline without ESG controversy features versus full model with ESG controversy features.
- Algorithm comparison: Logistic Regression, Random Forest, and Gradient Boosting on the same chronological splits.
- Evaluation metrics: ROC-AUC, Precision@Top Bucket, and Crash Capture.

Report artifacts to use:

- `outputs/model_dataset.parquet`
- `outputs/algorithm_comparison.csv`
- `outputs/esg_model_comparison.csv`
- `outputs/feature_importance.csv`

Key interpretation:

> A positive full-minus-baseline result means ESG controversy features improve out-of-sample crash-risk ranking or top-bucket capture relative to the non-ESG benchmark.

## 4. Business Analysis

The business use case is a long-only risk overlay. Each week, a fund can review or exclude stocks classified as High crash risk, then compare performance with the full equal-weight universe.

Report artifacts to use:

- `outputs/business_analysis.csv`
- `outputs/figures/price_scenario_range.svg`

Metrics to report:

- Strategy annual return versus benchmark annual return.
- Annualized alpha.
- Sharpe and Sortino ratios.
- Strategy versus benchmark max drawdown.
- VaR and CVaR.
- Percentage of high-risk names excluded.
- Estimated annual economic gain for a 1 billion dollar AUM fund.
- Whether the estimated gain justifies a 4-person implementation team.

Recommended wording:

> The business analysis should be interpreted as an illustrative fund overlay until the model is rerun on full real Bloomberg history.

## Appendix: Code Files

Submit the Python package files rather than notebooks. The most important files to cite in the appendix are:

- `crashrisk/pipeline.py` for the end-to-end run.
- `crashrisk/features/` for returns, turnover, downside-risk, crash-risk, and ESG controversy features.
- `crashrisk/models/` for chronological splits, training, scoring, scenario ranges, and ESG/model comparison.
- `crashrisk/analysis/` for business analysis and FDS report artifacts.
- `crashrisk/api/main.py` for the live Render-style upload API.
