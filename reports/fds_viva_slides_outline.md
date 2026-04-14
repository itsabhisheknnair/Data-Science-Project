# Viva Slides Outline

## Slide 1: Problem and Research Question

- Title: ESG Controversy Signals for Equity Crash-Risk Monitoring.
- Main question: Do ESG controversy signals help predict future stock crash risk?
- Positioning: equity risk-management application for funds, not a point price-forecasting product.
- Output: crash-risk probability, Low/Medium/High bucket, top drivers, and scenario range.

## Slide 2: Data and Cleaning

- Inputs: prices, benchmark prices, fundamentals, ESG controversies.
- Use `outputs/data_summary.csv` for ticker counts, row counts, sector coverage, and date range.
- Use `outputs/cleaning_log.csv` for missing values, invalid prices, duplicate checks, and the 45-day fundamentals reporting lag.
- Use `outputs/sql_summary.md` to show SQL evidence from the dataset.

## Slide 3: ESG and Textual Signal

- Core ESG signal: Bloomberg-style `controversy_score`.
- Engineered controversy features: score level, 4/13/26-week changes, rolling mean, rolling standard deviation, spike flag, and sector percentile.
- Optional text extension: `news_text` or `controversy_text` with sentiment and keyword features.
- Use `outputs/textual_analysis.csv` and `outputs/figures/text_word_cloud.svg` to state whether direct text was available or whether the vendor score is used as the text-derived proxy.

## Slide 4: Machine Learning Design

- Target: top 20% future 13-week NCSKEW bucket.
- Primary model: interpretable Logistic Regression with balanced class weights.
- Comparisons: Random Forest and Gradient Boosting.
- Validation: chronological train/validation/test split to avoid time leakage.
- Metrics: ROC-AUC, Precision@Top Bucket, and Crash Capture.

## Slide 5: Does ESG Add Signal?

- Use `outputs/esg_model_comparison.csv`.
- Compare baseline model without ESG controversy features versus full model with ESG controversy features.
- Explain the full-minus-baseline row:
  - Positive ROC-AUC lift means better ranking of future risky names.
  - Positive precision lift means better accuracy among top-risk flags.
  - Positive crash-capture lift means more future high-risk names caught in the top bucket.

## Slide 6: Business Value and Demo

- Use `outputs/business_analysis.csv`.
- Explain the risk overlay: review or exclude High crash-risk stocks.
- Report alpha, Sharpe, Sortino, drawdown, VaR/CVaR, and estimated gain on 1 billion dollar AUM.
- Close with the live dashboard: upload raw Bloomberg-style files, run live score, review risk table, scenario chart, explainability, validation, and business analysis.
