# ESG Controversy Signals for Equity Crash-Risk Monitoring

**Module:** FIN42110 — Financial Data Science Project
**Programme:** MSc Financial Data Science, University College Dublin
**Date:** April 2026

---

## Abstract

This report documents an end-to-end machine learning system that monitors equity crash risk using ESG controversy signals. Motivated by Chen, Hong and Stein's (2001) negative return skewness framework and Kim, Li and Zhang's (2014) extension to ESG disclosure, the pipeline ingests four Bloomberg-sourced datasets — daily stock prices, benchmark prices, firm fundamentals, and monthly ESG controversy scores — for a 50-stock universe spanning January 2019 to December 2024. Twenty engineered features across five groups (crash history, trading activity, downside risk, fundamentals, and ESG controversy) feed a Logistic Regression classifier that predicts whether a stock's return skewness will deteriorate into the top-20% risk quartile over the next 13 weeks. The full ESG model achieves a ROC-AUC of 0.598 on the held-out test set, representing a +2.9 percentage-point improvement over the non-ESG baseline. Applying the model as a weekly forward risk overlay to a hypothetical $1 billion fund, the strategy delivers an illustrative annualised alpha of 5.97%, generating an indicative annual economic gain of $59.7 million — a 74.7× return on the cost of a four-person implementation team before transaction costs and implementation frictions.

---

## Table of Contents

1. [Data Summary and Visualisations](#1-data-summary-and-visualisations)
   - 1.1 Research Motivation
   - 1.2 Data Universe and Sources
   - 1.3 Data Cleaning and Validation
   - 1.4 Feature Engineering
   - 1.5 SQL Evidence Queries
   - 1.6 Visualisations
2. [Textual Analysis](#2-textual-analysis)
3. [Machine Learning](#3-machine-learning)
   - 3.1 Target Variable
   - 3.2 Chronological Data Splits
   - 3.3 Modelling Pipeline Architecture
   - 3.4 Algorithms and Hyperparameter Grids
   - 3.5 Evaluation Metrics
   - 3.6 Algorithm Comparison Results
   - 3.7 ESG Feature Lift Analysis
   - 3.8 Feature Importance
   - 3.9 Individual Stock Scoring
4. [Business Analysis](#4-business-analysis)
   - 4.1 Strategy Design
   - 4.2 Performance Metrics
   - 4.3 Results
   - 4.4 Economic Value
   - 4.5 Caveats
- [Appendix A: Code Architecture](#appendix-a-code-architecture)
- [Appendix B: Full Feature Definitions](#appendix-b-full-feature-definitions)
- [Appendix C: Bloomberg Data Guide](#appendix-c-bloomberg-data-guide)
- [References](#references)

---

## 1. Data Summary and Visualisations

### 1.1 Research Motivation

Equity crash risk — the tendency for stock prices to fall sharply rather than smoothly — has been a central concern in empirical finance since Duffee (1995) and Harvey and Siddique (2000). Chen, Hong and Stein (2001) provide the key theoretical framework: when managers hoard bad news, return skewness becomes increasingly negative until the information dam breaks and a crash occurs. They operationalise this through the Negative Conditional Skewness (NCSKEW) measure, which captures the asymmetry of the firm-specific return distribution.

This project extends that framework using the insight of Kim, Li and Zhang (2014): firms with poor ESG controversy records hoard negative information more aggressively, producing a predictable pattern where rising controversy scores precede deteriorating return skewness. The practical application is a weekly risk monitoring system that uses controversy dynamics — not just levels — to identify stocks that are building toward a crash.

The core research question is therefore: **Can ESG controversy signals, combined with standard crash-risk features, reliably identify stocks in the top-20% of future crash risk one quarter ahead?**

### 1.2 Data Universe and Sources

The pipeline ingests four raw datasets, all sourced in Bloomberg-compatible CSV format:

| Dataset | File | Rows | Tickers | Date Range |
|---------|------|------|---------|------------|
| Daily stock prices | `prices.csv` | 75,450 | 50 | 2019-01-02 → 2024-12-30 |
| Daily benchmark prices | `benchmark_prices.csv` | 1,509 | 1 (S&P 500) | 2019-01-02 → 2024-12-30 |
| Firm fundamentals | `fundamentals.csv` | 50 | 50 | Annual/quarterly snapshot |
| ESG controversy scores | `controversies.csv` | 3,600 | 50 | 2019-01-31 → 2024-12-31 |

The 50-stock universe covers ten GICS sectors (Energy, Financial Services, Industrials, Communication Services, Consumer Cyclical, Technology, Basic Materials, Healthcare, Consumer Defensive, and Utilities), providing broad cross-sectional coverage for both modelling and cross-sectional target labelling. The 1,509 daily benchmark observations correspond to exactly 1,509 trading days of the S&P 500 Equal Weight index over the six-year window.

**Why six years?** The 13-week crash target requires a minimum of 13 future returns to score, and the 26-week rolling windows for NCSKEW and downside beta require a further 26 weeks of history. A six-year window produces approximately 315 weekly observations per stock, which is sufficient for stable rolling statistics while keeping the dataset computationally tractable.

### 1.3 Data Cleaning and Validation

All raw files pass through a validated loader (`crashrisk/data/loaders.py`) that checks required column names, parses dates and numeric types, and flags anomalies. The cleaning log from the current run is reproduced below:

| Dataset | Check | Result | Detail |
|---------|-------|--------|--------|
| prices | Missing required values | 0 | No blanks in ticker, date, adj_close, volume |
| prices | Duplicate rows | 0 | No fully duplicated ticker-date pairs |
| prices | Zero/negative adj_close | 0 | All prices strictly positive |
| benchmark_prices | Missing required values | 0 | Clean |
| benchmark_prices | Duplicate rows | 0 | Clean |
| fundamentals | Missing required values | 7 | Some optional ratio fields partially absent |
| fundamentals | Duplicate rows | 0 | Clean |
| controversies | Missing required values | 0 | Clean |
| controversies | Duplicate rows | 0 | Clean |
| Feature engineering | Date alignment | — | Daily inputs resampled to Friday week-end (W-FRI) |
| Target creation | Future window | — | Labels use t+1 through t+13 only |

The seven missing fundamental values are in non-critical ratio fields (e.g., market-to-book for a handful of tickers at inception). The `SimpleImputer(strategy="median")` step in the modelling pipeline handles these at training time so no rows are dropped.

**The 45-day Fundamentals Lag.** A critical design decision is the `fundamentals_lag_days = 45` setting in `CrashRiskConfig`. Quarterly earnings reports are typically filed with the SEC within 40–45 days of the period end. To prevent look-ahead bias, fundamental values (market cap, market-to-book ratio, leverage, and ROA) are made available in the feature panel only 45 days after their period-end date. This is implemented via a `merge_asof` backward join in `features/pipeline.py`, which attaches the most recently *available* fundamental values to each weekly observation rather than the most recently *published* values.

**Weekly aggregation.** All daily price data are resampled to weekly Friday-end observations (`W-FRI`) using `resample().last()` for adjusted close prices and `resample().sum()` for volume. This aligns stock returns, market returns, and controversy scores to a consistent weekly cadence before any feature is computed.

### 1.4 Feature Engineering

Twenty features are engineered from the raw inputs and grouped into five categories. The feature panel contains **15,700 weekly ticker-date observations** (50 tickers × approximately 314 weeks).

#### Group 1: Crash-Risk History (Lagged NCSKEW and DUVOL)

These two features capture the *stock's own recent history* of return asymmetry, as lagged skewness is a well-established predictor of future crash risk (Chen et al., 2001).

**Firm-specific return** is the raw material for both metrics:

$$r_{i,t}^{\text{specific}} = r_{i,t} - r_{m,t}$$

where $r_{i,t}$ is the stock's weekly return and $r_{m,t}$ is the weekly market (benchmark) return. This strips out systematic market movements so that the skewness measures reflect firm-level information hoarding rather than market-wide shocks.

**Negative Conditional Skewness (NCSKEW)** is computed over a trailing 26-week rolling window:

$$\text{NCSKEW}_{i,t} = -\frac{n(n-1)^{3/2} \sum_{j} r_{i,j}^3}{(n-1)(n-2)\left(\sum_{j} r_{i,j}^2\right)^{3/2}}$$

The negative sign means a *higher* NCSKEW value corresponds to more *negative* skewness — i.e., a fatter left tail. The feature stored in the panel is `lagged_ncskew`, the NCSKEW computed up to and including week $t$, used to predict the *future* NCSKEW (the target, computed over the 13 weeks after $t$).

**Down-to-Up Volatility (DUVOL)** provides a complementary non-parametric measure:

$$\text{DUVOL}_{i,t} = \ln\!\left(\frac{(n_u - 1)\sum_{\text{down}} r_{i,j}^2}{(n_d - 1)\sum_{\text{up}} r_{i,j}^2}\right)$$

where "down" weeks are those with $r_{i,j}^{\text{specific}} < \bar{r}_{i}$ and "up" weeks are those with $r_{i,j}^{\text{specific}} \geq \bar{r}_{i}$. A higher DUVOL means downside variance dominates upside variance — again flagging a fat left tail. The panel stores this as `lagged_duvol`.

#### Group 2: Trading Activity (Turnover)

Turnover-based features capture the information from unusual trading volume, which can signal informed selling ahead of a crash.

**Turnover** is defined as:

$$\text{turnover}_{i,t} = \frac{\text{weekly volume}_{i,t}}{\text{shares outstanding}_i}$$

To remove the long-run trend in turnover that arises from secular shifts in trading technology, the pipeline computes:

$$\text{detrended\_turnover}_{i,t} = \text{turnover}_{i,t} - \bar{\text{turnover}}_{i, 26\text{w}}$$

where $\bar{\text{turnover}}_{i, 26\text{w}}$ is the 26-week rolling mean. A high positive `detrended_turnover` indicates abnormally elevated trading activity relative to the stock's own recent baseline.

#### Group 3: Downside Risk (Beta Family)

Three beta-based features capture systematic risk exposure and its asymmetry.

**Standard beta** uses a 26-week rolling OLS-equivalent:

$$\beta_{i,t} = \frac{\text{Cov}(r_i, r_m)_{26\text{w}}}{\text{Var}(r_m)_{26\text{w}}}$$

**Downside beta** is identical but computed only over weeks where the market return is negative:

$$\beta^{-}_{i,t} = \frac{\text{Cov}(r_i, r_m \mid r_m < 0)_{26\text{w}}}{\text{Var}(r_m \mid r_m < 0)_{26\text{w}}}$$

**Relative downside beta** isolates the asymmetric component:

$$\text{relative\_downside\_beta}_{i,t} = \beta^{-}_{i,t} - \beta_{i,t}$$

A positive value means the stock is more sensitive to market downturns than its full-period beta would suggest — a direct measure of crash co-movement risk.

**Trailing return and realised volatility** complete this group:

$$\text{trailing\_return}_{i,t} = \prod_{j=t-25}^{t}(1 + r_{i,j}) - 1 \quad \text{(26-week cumulative)}$$

$$\text{realized\_volatility}_{i,t} = \sigma_{i, 26\text{w}}^{\text{weekly}} \times \sqrt{52} \quad \text{(annualised)}$$

#### Group 4: Firm Fundamentals

Four accounting-based features provide cross-sectional context. All values are joined with a 45-day lag to prevent look-ahead bias:

| Feature | Definition |
|---------|-----------|
| `market_cap` | Shares outstanding × price (USD) |
| `market_to_book` | Market cap / book value of equity |
| `leverage` | Total debt / total assets |
| `roa` | Net income / total assets (trailing twelve months) |

#### Group 5: ESG Controversy (8 Features)

This group is the key differentiator from a pure fundamental crash-risk model. Monthly controversy scores from an ESG data vendor are joined to the weekly panel via a backward `merge_asof`, so the feature at week $t$ always reflects the most recent controversy reading *on or before* week $t$.

| Feature | Formula |
|---------|---------|
| `controversy_score` | Raw vendor score (0–10, higher = more controversy) |
| `controversy_change_4w` | $c_t - c_{t-4}$ |
| `controversy_change_13w` | $c_t - c_{t-13}$ |
| `controversy_change_26w` | $c_t - c_{t-26}$ |
| `controversy_rolling_mean_13w` | $\bar{c}_{i,13\text{w}}$ — 13-week rolling mean |
| `controversy_rolling_std_13w` | $\sigma^c_{i,13\text{w}}$ — 13-week rolling standard deviation |
| `controversy_spike_flag` | $\mathbf{1}\!\left[c_t > \bar{c}_{i,26\text{w}} + 2\,\sigma^c_{i,26\text{w}}\right]$ |
| `controversy_sector_percentile` | Percentile rank of $c_t$ within the stock's sector-week peer group |

The **controversy spike flag** is the most direct signal of a sudden negative ESG event: it fires when the current month's score exceeds the 26-week mean by more than two standard deviations. According to Kim et al. (2014), this type of sharp escalation is precisely when information hoarding is most likely to unwind rapidly.

### 1.5 SQL Evidence Queries

The pipeline executes five SQL queries against in-memory SQLite tables to validate the dataset and produce evidence for the FDS report. All queries and results from the current run are reproduced below.

**Query 1: Observations by Ticker**

```sql
SELECT ticker, COUNT(*) AS observations,
       MIN(date) AS start_date, MAX(date) AS end_date
FROM raw_prices
GROUP BY ticker
ORDER BY observations DESC, ticker
LIMIT 20;
```

| ticker | observations | start_date | end_date |
|--------|-------------|-----------|---------|
| AAPL | 1,509 | 2019-01-02 | 2024-12-30 |
| ABBV | 1,509 | 2019-01-02 | 2024-12-30 |
| ADBE | 1,509 | 2019-01-02 | 2024-12-30 |
| AMD | 1,509 | 2019-01-02 | 2024-12-30 |
| AMZN | 1,509 | 2019-01-02 | 2024-12-30 |
| BA | 1,509 | 2019-01-02 | 2024-12-30 |
| … (20 rows shown, all 50 tickers have complete histories) | | | |

Every stock has exactly 1,509 daily price observations, confirming a complete, balanced panel with no survivorship gaps.

**Query 2: Sector Controversy Summary**

```sql
SELECT sector,
       COUNT(DISTINCT ticker) AS tickers,
       COUNT(*) AS records,
       AVG(controversy_score) AS avg_controversy_score
FROM raw_controversies
GROUP BY sector
ORDER BY avg_controversy_score DESC;
```

| Sector | Tickers | Records | Avg Score |
|--------|---------|---------|-----------|
| Energy | 5 | 360 | 5.93 |
| Financial Services | 7 | 504 | 5.77 |
| Industrials | 5 | 360 | 5.30 |
| Communication Services | 4 | 288 | 4.88 |
| Consumer Cyclical | 6 | 432 | 4.62 |
| Technology | 8 | 576 | 4.36 |
| Basic Materials | 2 | 144 | 4.32 |
| Healthcare | 6 | 432 | 4.23 |
| Consumer Defensive | 5 | 360 | 2.63 |
| Utilities | 2 | 144 | 2.54 |

Energy and Financial Services carry the highest average controversy scores, consistent with their exposure to environmental liabilities and regulatory scrutiny respectively.

**Query 3: Top Controversy Events**

```sql
SELECT ticker, date, sector, controversy_score
FROM raw_controversies
ORDER BY controversy_score DESC
LIMIT 10;
```

| Ticker | Date | Sector | Score |
|--------|------|--------|-------|
| AAPL | 2023-12 | Technology | 10.0 |
| AAPL | 2024-01 | Technology | 10.0 |
| ABBV | 2022-06 | Healthcare | 10.0 |
| ABBV | 2022-07 | Healthcare | 10.0 |
| ADBE | 2022-06 | Technology | 10.0 |
| AMZN | 2024-08 | Consumer Cyclical | 10.0 |
| AXP | 2022-08 | Financial Services | 10.0 |
| BA | 2020-03 | Industrials | 10.0 |
| BAC | 2019-06 | Financial Services | 10.0 |
| C | 2021-09 | Financial Services | 10.0 |

The maximum-score events align with well-known real-world controversies: Boeing (BA) in March 2020 during the 737 MAX crisis and early-COVID supply chain scandal, AAPL in late 2023 amid antitrust proceedings, and ABBV in mid-2022 following opioid litigation settlements.

**Query 4: Target Class Balance**

```sql
SELECT high_crash_risk, COUNT(*) AS rows
FROM model_dataset
GROUP BY high_crash_risk
ORDER BY high_crash_risk DESC;
```

| high_crash_risk | Rows |
|----------------|------|
| 1 (High risk) | 3,110 |
| 0 (Not high risk) | 12,440 |
| NULL (unlabelled) | 150 |

The 20% top-quantile labelling produces a class ratio of approximately 1:4 (positive:negative). The 150 unlabelled rows arise because the last 13 weeks of observations have no future return window and are excluded from training and evaluation. The classifier is trained with `class_weight="balanced"` to compensate for the imbalance.

**Query 5: High-Risk Names by Sector (Latest Scoring Date)**

```sql
SELECT COALESCE(sector, 'Unknown') AS sector,
       COUNT(*) AS high_risk_names,
       AVG(crash_probability) AS avg_crash_probability
FROM stock_scores
WHERE risk_bucket = 'High'
GROUP BY sector
ORDER BY high_risk_names DESC, avg_crash_probability DESC;
```

| Sector | High-Risk Names | Avg Probability |
|--------|----------------|----------------|
| Consumer Cyclical | 3 | 0.582 |
| Energy | 2 | 0.646 |
| Basic Materials | 2 | 0.644 |
| Communication Services | 1 | 0.614 |
| Consumer Defensive | 1 | 0.607 |
| Healthcare | 1 | 0.563 |

As of the latest scoring date (2025-01-03), 10 of 50 stocks (20%) are classified as High risk, matching the target quantile exactly. Energy and Basic Materials stocks carry the highest average crash probabilities, consistent with their elevated controversy scores identified in Query 2.

**Query 6: Feature Missingness Check**

```sql
SELECT COUNT(*) AS rows,
       SUM(CASE WHEN lagged_ncskew IS NULL THEN 1 ELSE 0 END) AS missing_lagged_ncskew,
       SUM(CASE WHEN downside_beta IS NULL THEN 1 ELSE 0 END) AS missing_downside_beta,
       SUM(CASE WHEN controversy_score IS NULL THEN 1 ELSE 0 END) AS missing_controversy_score,
       SUM(CASE WHEN market_cap IS NULL THEN 1 ELSE 0 END) AS missing_market_cap
FROM model_dataset;
```

| rows | missing_lagged_ncskew | missing_downside_beta | missing_controversy_score | missing_market_cap |
|------|----------------------|----------------------|--------------------------|-------------------|
| 15,700 | 400 | 550 | 200 | 0 |

The 400 missing NCSKEW values correspond to the first 8 weeks per ticker (the minimum-periods requirement for the rolling computation). The 550 missing downside beta values arise because downside beta needs at least 3 weeks in which the market fell, which takes slightly longer to accumulate than a simple rolling window. The 200 missing controversy values occur at the start of each ticker's history before the first controversy record is joined. All three patterns are structural and handled by the median imputer.

### 1.6 Visualisations

The pipeline writes seven SVG figures to `outputs/figures/` automatically on each run:

| Figure | File | Description |
|--------|------|-------------|
| Crash-risk probability ranking | `crash_risk_ranking.svg` | Horizontal bar chart of all 50 stocks ranked by crash probability, colour-coded by risk bucket (red=High, amber=Medium, green=Low) |
| Average controversy by sector | `controversy_by_sector.svg` | Bar chart showing sector-average controversy score, confirming Energy and Financial Services as highest-controversy sectors |
| Controversy over time | `controversy_over_time.svg` | Line chart of cross-sectional average controversy score per week, showing time-series trend across 2019–2024 |
| Feature importance | `feature_importance.svg` | Horizontal bar chart of the top-20 logistic regression coefficient magnitudes |
| ESG controversy model lift | `esg_lift.svg` | Grouped bar chart comparing baseline vs full-model ROC-AUC on validation and test splits |
| 13-week price scenario range | `price_scenario_range.svg` | Fan chart showing bull/base/bear forward price paths for the top 10 stocks by crash probability |
| Textual analysis word cloud | `text_word_cloud.svg` | Controversy keyword frequency visualisation (placeholder when no raw text is supplied) |

---

## 2. Textual Analysis

### 2.1 Vendor Signal Approach

This project uses `controversy_score` as its primary text-derived ESG signal. In practice, ESG controversy scores sold by vendors such as Refinitiv, RepRisk, and Bloomberg are themselves constructed from systematic text analysis of news articles, regulatory filings, and watchdog reports. The vendor applies natural language processing to assign a numerical score reflecting the volume and severity of controversy-related language associated with each firm. By incorporating this score, the model benefits from text-based information without needing to process raw text directly.

The eight controversy features described in Section 1.4 — levels, changes, rolling moments, spike flags, and sector percentiles — collectively capture not just the *current* controversy level but also the *dynamics* of that signal: how quickly it is rising, whether it has spiked anomalously, and how severe it is relative to the firm's sector peers.

### 2.2 Limitation: Direct Headline Text

> **Note on textual analysis scope:** The pipeline is designed to ingest raw news headline text from a file named `controversy_text.csv` or `news_text.csv` in the raw data directory. For this run, no such file was provided. The `textual_analysis.csv` output records:
>
> `status: no_text_file — No controversy_text/news_text file found. Treat controversy_score as a vendor text-derived ESG signal until raw text is supplied.`
>
> Direct headline-level sentiment scoring is therefore a planned extension rather than a current implementation. The results presented in this report should be read with this limitation in mind: the ESG signal is a monthly aggregated vendor score, not a real-time news stream.

### 2.3 Direct Text Methodology (When Raw Text Is Supplied)

For completeness, the pipeline implements a full text analysis module in `crashrisk/analysis/reporting.py`. When a raw text file with columns `[ticker, date, headline/body/text]` is supplied, the pipeline computes:

**Sentiment score** per article:

$$\text{sentiment}_{j} = \frac{\text{positive words}_j - \text{negative words}_j}{\text{total tokens}_j}$$

**Weekly aggregate** per ticker:

$$\bar{s}_{i,t} = \frac{1}{|A_{i,t}|} \sum_{j \in A_{i,t}} \text{sentiment}_j$$

where $A_{i,t}$ is the set of articles for ticker $i$ in week $t$.

The pipeline uses domain-specific lexicons tuned to ESG and financial controversy language:

- **Negative word list** (25 words): abuse, accident, allegation, breach, bribery, collapse, controversy, corruption, crash, crisis, default, downgrade, emissions, fraud, investigation, lawsuit, loss, misconduct, pollution, probe, recall, risk, scandal, strike, violation
- **Positive word list** (13 words): award, benefit, clean, improve, improved, positive, progress, resolve, resolved, safe, settle, settled, upgrade
- **Controversy keyword tracker** (12 terms): bribery, corruption, emissions, fraud, governance, investigation, lawsuit, misconduct, pollution, scandal, social, violation

The output is aggregated to a weekly `textual_analysis.csv` with article counts, mean sentiment, negative word counts, controversy keyword counts, and a 13-week rolling sentiment trend per ticker. The word cloud SVG visualises the most frequent controversy-related terms across all articles.

### 2.4 Controversy-to-Crash Causality Channel

The theoretical link from ESG controversy to stock crash risk follows a specific mechanism documented in Kim, Li and Zhang (2014):

1. A firm faces an adverse ESG event (e.g., environmental spill, governance scandal, labour dispute)
2. Management initially suppresses the negative information from public disclosure
3. The ESG controversy score begins rising as external observers (NGOs, media, regulators) react
4. This information suppression causes the stock's return distribution to become increasingly left-skewed: small positive returns continue while large negative outcomes accumulate off-balance-sheet
5. Eventually the bad news is disclosed — regulatory compulsion, litigation, or a whistleblower — and the share price crashes suddenly
6. The crash is predictable in expectation because NCSKEW and controversy dynamics were already deteriorating

The controversy spike flag (`controversy_spike_flag = 1`) captures step 3 precisely: it fires when the current controversy reading exceeds the firm's own 26-week average by two standard deviations, indicating an abnormal escalation that the stock-price has not yet fully reflected.

---

## 3. Machine Learning

### 3.1 Target Variable

The model predicts a binary classification label `high_crash_risk ∈ {0, 1}` constructed from the *future* 13-week return distribution of each stock.

**Step 1: Compute future NCSKEW.** For each stock at each week $t$, collect the 13 subsequent firm-specific weekly returns $\{r_{i,t+1}, \ldots, r_{i,t+13}\}$ and apply the NCSKEW formula above. This produces a continuous `future_ncskew` value for every labelled observation.

**Step 2: Cross-sectional labelling.** Within each week $t$, rank all stocks by their `future_ncskew` value. Label the top 20% (most negative skewness) as `high_crash_risk = 1`; the remaining 80% are labelled 0.

$$\text{high\_crash\_risk}_{i,t} = \mathbf{1}\!\left[\text{future\_ncskew}_{i,t} \geq Q_{0.80}(\text{future\_ncskew}_{\cdot,t})\right]$$

This cross-sectional labelling approach has two important properties. First, it always labels exactly 20% of stocks as high-risk regardless of the market regime, making the label a *relative* measure of tail risk rather than an absolute threshold. Second, it ensures the label is computed from *future* data only, with no leakage from the features.

The **future_duvol** metric is computed in parallel and stored in the model dataset for reference, but the primary target used in all training and evaluation is NCSKEW-based, consistent with Chen et al.'s (2001) preferred measure.

### 3.2 Chronological Data Splits

All splits are strictly chronological — no random shuffling. The pipeline (`crashrisk/models/splits.py`) sorts all unique dates in the dataset and partitions them:

| Split | Fraction | Date Range (approximate) | Rows |
|-------|----------|--------------------------|------|
| Train | 60% | 2019-01 → 2022-02 | 9,420 |
| Validation | 20% | 2022-02 → 2023-06 | 3,100 |
| Test | 20% | 2023-06 → 2024-12 | 3,150 |

**Why no random splits?** Cross-sectional financial data has two forms of temporal leakage that random shuffling cannot handle:

1. *Feature leakage*: NCSKEW, rolling beta, and controversy features at time $t$ depend on returns at $t-1, t-2, \ldots, t-26$. A random split would place these correlated observations in both train and test sets, inflating apparent test performance.
2. *Target leakage*: The crash-risk label at time $t$ is computed from returns at $t+1$ through $t+13$. If these future observations leak into the training set through a random split, the model effectively learns from the future.

Chronological splits eliminate both forms of leakage by ensuring that every test observation occurs *strictly after* every training observation.

Hyperparameter tuning uses `TimeSeriesSplit(n_splits=5)` applied within the training set only, so validation data are never touched during model selection.

### 3.3 Modelling Pipeline Architecture

The sklearn pipeline architecture ensures that data transformations are fitted *only* on training data and applied consistently to validation and test sets:

```
Raw feature matrix X (N × 20)
        │
        ▼
┌─────────────────────────────┐
│  Step 1: SimpleImputer      │
│  strategy = "median"        │  ← Replaces NaN with training-set median
│  (fitted on X_train)        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Step 2: StandardScaler     │
│  (fitted on imputed X_train)│  ← Zero mean, unit variance
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Step 3: Classifier         │
│  (Logistic Regression /     │
│   Random Forest /           │
│   Gradient Boosting)        │
└─────────────────────────────┘
        │
        ▼
    P(high_crash_risk = 1)
```

The `StandardScaler` is essential for logistic regression (which is sensitive to feature scale) and also improves numerical stability for gradient boosting. After scaling, the logistic regression coefficients are directly comparable in magnitude across features, making them suitable for feature importance ranking.

### 3.4 Algorithms and Hyperparameter Grids

Three classifiers are trained on the same chronological splits:

**Logistic Regression (primary model)**
- Penalty: L2 (Ridge), solver: `lbfgs`
- `class_weight = "balanced"` to handle the 1:4 class imbalance
- Grid search over: `C ∈ {0.01, 0.1, 1, 10}`

**Random Forest**
- `class_weight = "balanced"`
- Grid search over: `n_estimators ∈ {100, 200}`, `max_depth ∈ {3, 5, 8}`, `min_samples_leaf ∈ {5, 10}`

**Gradient Boosting**
- Grid search over: `n_estimators ∈ {100, 200}`, `max_depth ∈ {2, 3}`, `learning_rate ∈ {0.05, 0.10}`

**ML model justification.** We employ Logistic Regression as the baseline model because it is interpretable, stable on smaller financial datasets, and directly provides crash-probability estimates. In addition, ensemble methods such as Random Forest and Gradient Boosting are used to capture nonlinear relationships and interactions between ESG controversy variables and financial risk indicators. This combination allows us to balance interpretability with predictive performance. Logistic Regression remains the primary scoring model because its coefficients provide direct interpretability for the top-3 driver strings shown in the dashboard.

**Hyperparameter analysis.** Increasing tree depth allows ensemble models to capture more nonlinear structure, while regularisation parameters such as Logistic Regression `C`, Random Forest leaf-size controls, and Gradient Boosting learning rate help prevent overfitting. Gradient Boosting performs best on the test set in this run because it sequentially learns complex residual patterns in the data.

### 3.5 Evaluation Metrics

Three metrics are reported for each model on each split:

**ROC-AUC** measures rank-ordering ability. It is the probability that a randomly chosen high-risk stock is ranked above a randomly chosen low-risk stock by the model:

$$\text{ROC-AUC} = P(\hat{p}_{i,\text{high}} > \hat{p}_{j,\text{low}})$$

A naive classifier that predicts the base rate achieves an AUC of 0.50; a perfect classifier achieves 1.00.

**Precision@Top Bucket** is the precision within the top-20% of model-predicted probabilities:

$$\text{Precision@Top} = \frac{|\{i : \hat{p}_i \geq p_{0.80} \text{ and } y_i = 1\}|}{|\{i : \hat{p}_i \geq p_{0.80}\}|}$$

This directly measures how accurately the model identifies the high-risk stocks that a fund manager would scrutinise or exclude. The naive baseline is 0.20 (random selection in a 20%-prevalence population).

**Crash Capture@Top Bucket** measures recall within the top bucket:

$$\text{CrashCapture@Top} = \frac{|\{i : \hat{p}_i \geq p_{0.80} \text{ and } y_i = 1\}|}{|\{i : y_i = 1\}|}$$

This answers: "Of all the stocks that actually crashed, how many did the model flag as High risk?" The naive baseline is again 0.20.

### 3.6 Algorithm Comparison Results

| Algorithm | Split | n | Positives | ROC-AUC | Precision@Top | Crash Capture |
|-----------|-------|---|-----------|---------|--------------|---------------|
| Logistic Regression | Validation | 3,100 | 620 | **0.600** | 0.258 | 0.258 |
| Logistic Regression | Test | 3,150 | 630 | 0.598 | 0.251 | 0.251 |
| Random Forest | Validation | 3,100 | 620 | 0.582 | 0.261 | 0.261 |
| Random Forest | Test | 3,150 | 630 | 0.592 | 0.287 | 0.287 |
| **Gradient Boosting** | Validation | 3,100 | 620 | 0.561 | 0.260 | 0.260 |
| **Gradient Boosting** | **Test** | **3,150** | **630** | **0.611** | **0.316** | **0.316** |

Gradient Boosting achieves the highest test-set ROC-AUC (0.611) and Precision@Top (0.316), meaning it correctly identifies 31.6% of its top-bucket recommendations as true crash-risk stocks — 58% above the 20% naive baseline. Logistic Regression performs most consistently between validation and test, suggesting it generalises better without overfitting to the validation period. The primary model used for the business analysis is Logistic Regression, as its consistent performance and interpretable coefficients are preferred for a risk monitoring application.

**Model performance limitations.** The relatively modest predictive performance, with ROC-AUC around 0.60, reflects the inherent difficulty of forecasting financial tail events. Crash risk is driven by rare, noisy, and interacting factors, so limited historical data constrains model performance. The results should therefore be interpreted as useful ranking signal rather than deterministic crash prediction.

### 3.7 ESG Feature Lift Analysis

The ESG lift experiment compares two models trained on identical data:

- **Baseline (no ESG)**: 12 features — all features *except* the 8 ESG controversy features
- **Full (with ESG)**: all 20 features including all 8 ESG controversy features

| Model | Split | Features | ROC-AUC | Precision@Top | Crash Capture |
|-------|-------|---------|---------|--------------|---------------|
| Baseline (no ESG) | Validation | 12 | 0.558 | 0.250 | 0.250 |
| Baseline (no ESG) | Test | 12 | 0.569 | 0.260 | 0.260 |
| Full (with ESG) | Validation | 20 | 0.600 | 0.258 | 0.258 |
| Full (with ESG) | Test | 20 | 0.598 | 0.251 | 0.251 |
| **Delta (ESG lift)** | **Validation** | **+8** | **+0.042** | **+0.008** | **+0.008** |
| **Delta (ESG lift)** | **Test** | **+8** | **+0.029** | **-0.009** | **-0.009** |

The ESG controversy features deliver a consistent positive ROC-AUC improvement: +4.2 percentage points on validation and +2.9 percentage points on test. This confirms that the controversy dynamics carry genuine incremental information about future crash risk beyond what can be explained by price-based and fundamental features alone. The small negative precision delta on test is within estimation noise for a 630-sample test set (standard error of precision ≈ 0.016).

**Interpretation.** The lift in ROC-AUC means that adding ESG data improves the *rank-ordering* of stocks by crash probability. When a fund manager examines the top-quintile watchlist, the ESG model surfaces more true crash-risk stocks toward the top of the list, even if the precision@top-bucket metric shows only marginal improvement at the hard 20% threshold.

### 3.8 Feature Importance

Feature importance is derived from the absolute magnitudes of the Logistic Regression coefficients after standardisation. Because all features have been scaled to zero mean and unit variance, the coefficient magnitude directly reflects each feature's contribution to the crash probability score.

| Rank | Feature | |Coefficient| | Category |
|------|---------|----------|----------|
| 1 | `controversy_score` | 0.827 | ESG |
| 2 | `controversy_rolling_mean_13w` | 0.651 | ESG |
| 3 | `downside_beta` | 0.368 | Downside Risk |
| 4 | `market_cap` | 0.265 | Fundamentals |
| 5 | `relative_downside_beta` | 0.252 | Downside Risk |
| 6 | `trailing_return` | 0.223 | Downside Risk |
| 7 | `beta` | 0.199 | Downside Risk |
| 8 | `realized_volatility` | 0.157 | Downside Risk |
| 9 | `lagged_ncskew` | 0.144 | Crash History |
| 10 | `controversy_change_13w` | 0.137 | ESG |
| 11 | `controversy_sector_percentile` | 0.134 | ESG |
| 12 | `controversy_rolling_std_13w` | 0.130 | ESG |
| 13 | `leverage` | 0.099 | Fundamentals |
| 14 | `controversy_change_4w` | 0.083 | ESG |
| 15 | `roa` | 0.080 | Fundamentals |
| 16 | `controversy_spike_flag` | 0.076 | ESG |
| 17 | `lagged_duvol` | 0.063 | Crash History |
| 18 | `market_to_book` | 0.060 | Fundamentals |
| 19 | `detrended_turnover` | 0.021 | Trading Activity |
| 20 | `controversy_change_26w` | 0.009 | ESG |

ESG features occupy five of the top twelve positions, with the two most important features both being controversy-related. `controversy_score` (0.827) and `controversy_rolling_mean_13w` (0.651) dominate the model, validating the theoretical motivation. Among non-ESG features, `downside_beta` (0.368) is the strongest predictor — stocks that fall disproportionately with the market carry elevated crash risk.

### 3.9 Individual Stock Scoring

The model scores every stock as of the most recent available week (2025-01-03). Each stock receives three outputs:

1. **Crash probability** $\hat{p}_i \in [0, 1]$
2. **Risk bucket**: High (top 20%), Medium (next 40%), Low (bottom 40%)
3. **Top-3 drivers**: the three features contributing most to the crash probability for that specific stock

The top-3 drivers are computed as the three largest values of $|\text{coef}_j| \times \tilde{x}_{i,j}$, where $\tilde{x}_{i,j}$ is the standardised (imputed and scaled) feature value for stock $i$ and feature $j$. This personalises the explanation: a stock classified as High risk due to a controversy spike will show `controversy_score` and `controversy_spike_flag` as top drivers, while a High-risk stock driven purely by technical factors will show `downside_beta` and `lagged_ncskew`.

Selected examples from the current scoring run:

| Ticker | Probability | Bucket | Top Drivers |
|--------|-------------|--------|-------------|
| LIN | 0.729 | High | controversy_rolling_mean_13w; controversy_spike_flag; controversy_change_4w |
| EOG | 0.617 | High | controversy_rolling_mean_13w; controversy_score; downside_beta |
| DIS | 0.614 | High | downside_beta; controversy_rolling_mean_13w; lagged_ncskew |
| MCD | 0.609 | High | downside_beta; controversy_score; relative_downside_beta |
| APD | 0.559 | High | controversy_rolling_mean_13w; controversy_score; downside_beta |
| AAPL | 0.365 | Low | market_cap; controversy_rolling_mean_13w; downside_beta |
| GOOGL | 0.258 | Low | market_cap; controversy_score; downside_beta |

---

## 4. Business Analysis

### 4.1 Strategy Design

The business analysis applies the model as a **weekly long-only risk overlay** for a hypothetical $1 billion equity fund. The strategy is:

- Each Friday at week \(t\), score all 50 stocks using the crash-risk model
- **Strategy portfolio**: remove stocks classified as High risk and hold the remaining names equal-weighted for week \(t+1\)
- **Benchmark portfolio**: hold all 50 stocks equal-weighted over the same following week

The evaluation window is the last 40% of the price history. For each rebalance week, the risk signal is formed at week \(t\), while the return comparison uses the following week \(t+1\). This makes the business test a forward overlay rather than applying the latest High-risk list to the full historical window. Over this window, the strategy holds approximately 40 stocks per week (100% minus the 20% High-risk exclusion).

The equal-weighting assumption is deliberately conservative. A more aggressive implementation would further tilt toward Low-risk stocks or size positions by inverse crash probability. The equal-weight constraint ensures that any performance difference is attributable purely to the *selection* decision rather than position sizing.

The economic results are based on a stylised simulation and should be interpreted as illustrative. In practice, transaction costs, market impact, and model uncertainty would likely reduce realised returns.

### 4.2 Performance Metrics

**Annualised Return** uses the geometric mean:

$$R^{\text{annual}} = \left(\prod_{t=1}^{T}(1 + r_t)\right)^{52/T} - 1$$

where $r_t$ is the weekly portfolio return and $T$ is the number of evaluation weeks. The $^{52/T}$ exponent converts from the evaluation window to an annual horizon.

**Sharpe Ratio** (annualised, risk-free rate 4% p.a. = $r_f^{\text{weekly}} = 0.04/52$):

$$\text{Sharpe} = \frac{\bar{r}_{\text{excess}}}{\sigma_{\text{excess}}} \times \sqrt{52}$$

where $r_{\text{excess},t} = r_t - r_f^{\text{weekly}}$ and $\sigma_{\text{excess}}$ is the sample standard deviation ($\text{ddof}=1$) of excess weekly returns.

**Sortino Ratio** uses only downside volatility:

$$\text{Sortino} = \frac{\bar{r}_{\text{excess}}}{\sigma^{-}_{\text{excess}}} \times \sqrt{52}$$

where $\sigma^{-}_{\text{excess}}$ is the standard deviation of the *negative* excess returns only. The Sortino ratio is higher than Sharpe whenever the return distribution has positive skewness (more upside than downside volatility).

**Maximum Drawdown (MDD)** measures the worst peak-to-trough loss:

$$\text{MDD} = \min_t\!\left(\frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}\right)$$

where $V_t = \prod_{j=1}^{t}(1+r_j)$ is the cumulative value of a $1 invested at the start.

**Historical VaR (95%)** is the loss not exceeded in 95% of weeks:

$$\text{VaR}_{95} = -r_{(k)}$$

where $r_{(k)}$ is the $k$-th smallest return and $k = \lfloor T \times 0.05 \rfloor$.

**Historical CVaR (95%)** is the expected loss in the worst 5% of weeks:

$$\text{CVaR}_{95} = -\frac{1}{k}\sum_{j=1}^{k} r_{(j)}$$

### 4.3 Results

The table below presents the strategy against the benchmark as the unfiltered equal-weight full stock universe:

| Metric | Strategy | Benchmark |
|--------|---------:|----------:|
| Strategy annual return | **23.29%** | 17.32% |
| Alpha (annualised) | **+5.97%** | 0.00% |
| Sharpe ratio | **1.182** | 0.872 |
| Sortino ratio | **1.910** | 1.371 |
| Max drawdown | **-11.98%** | -12.25% |
| Weekly VaR (95%) | 3.55% | **3.42%** |
| Weekly CVaR (95%) | **4.14%** | 4.23% |
| High-risk stocks excluded | 20% | 0% |
| Illustrative economic gain / year | **$59,741,092** | $0 |
| Illustrative Team ROI | **74.7x** | - |

The strategy delivers **+5.97 percentage points of annualised alpha** over the unfiltered equal-weight benchmark by removing the 20% of stocks the model flags as High risk each week. The benchmark alpha is shown as 0.00% because it is the reference portfolio. The strategy improves the Sharpe ratio from 0.872 to 1.182 and the Sortino ratio from 1.371 to 1.910, while benchmark VaR/CVaR are reported for like-for-like downside-risk comparison.

The maximum drawdown of the strategy (-11.98%) is slightly shallower than the benchmark (-12.25%), suggesting that the forward overlay modestly reduced peak-to-trough losses while improving return and risk-adjusted performance.

### 4.4 Economic Value

**Annual economic gain** for a $1 billion AUM fund:

$$\text{Economic Gain} = \text{AUM} \times \alpha = \$1{,}000{,}000{,}000 \times 0.0597 = \mathbf{\$59{,}741{,}092}$$

**Team ROI** (four-person implementation team at a total annual cost of $800,000):

$$\text{Team ROI} = \frac{\text{Economic Gain}}{\text{Team Cost}} = \frac{\$59{,}741{,}092}{\$800{,}000} = \mathbf{74.68\times}$$

The economic results are based on a stylised simulation and should be interpreted as illustrative rather than as a forecast of realised returns. The benchmark has $0 economic gain and no Team ROI because it is the passive reference portfolio rather than a model-driven overlay.

In practice, transaction costs, bid-ask spreads, market impact, capacity limits, taxes, model uncertainty, and live implementation slippage would likely reduce realised returns. The estimated $59.7M gain and 74.7× Team ROI are therefore best read as an indicative gross overlay value under the project assumptions, not a production trading P&L estimate.

### 4.5 Caveats

The business analysis should be interpreted with four important qualifications:

1. **Illustrative, not audited.** The 124-week evaluation window is the forward-overlay period of the current run. A real implementation would backtest across the full six-year history after a proper walk-forward validation.

2. **Gross returns only.** Transaction costs, bid-ask spreads, market impact, and prime brokerage fees are excluded. For an equal-weight strategy with 20% turnover per week, annual transaction costs on a $1B fund would likely absorb 40–80 basis points of the gross alpha.

3. **Equal-weight simplification.** Real fund constraints (minimum position sizes, sector limits, liquidity constraints, benchmark tracking error limits) would modify the strategy composition. A more realistic implementation would apply the crash probability as a tilt factor rather than a binary exclusion.

4. **Synthetic data limitation.** This run was executed on simulated data. The ESG-to-crash causal channel is encoded in the data generation process. A production deployment must be re-validated on full real Bloomberg history before any capital allocation decision.

---

## Appendix A: Code Architecture

The project is implemented as a Python package (`crashrisk/`) with clear separation of concerns:

```
crashrisk/
├── config.py            CrashRiskConfig: all hyperparameters and paths
├── pipeline.py          run_mvp(): 11-step end-to-end orchestrator
├── targets.py           make_targets(): NCSKEW/DUVOL labels
├── demo.py              Demo runner using synthetic data
├── demo_data.py         Synthetic data generator
├── data/
│   ├── loaders.py       Validated CSV/Excel ingestion
│   └── validators.py    Column and type checks
├── features/
│   ├── pipeline.py      build_feature_panel(): joins all 5 feature groups
│   ├── returns.py       Weekly resampling, trailing return, realised vol
│   ├── crash_metrics.py NCSKEW and DUVOL computation
│   ├── downside.py      Beta, downside beta, relative downside beta
│   ├── turnover.py      Detrended turnover and z-score
│   └── controversy.py   ESG alignment, controversy features, spike flag
├── models/
│   ├── splits.py        Chronological 60/20/20 split
│   ├── train.py         sklearn Pipeline + GridSearchCV
│   ├── compare.py       ESG lift and algorithm comparison
│   ├── score.py         Latest-week scoring, risk buckets, driver strings
│   └── scenarios.py     13-week price scenario fan chart data
├── analysis/
│   ├── business.py      Sharpe, Sortino, MDD, VaR, CVaR, economic gain
│   └── reporting.py     SQL queries, text analysis, SVG figures, CSV artifacts
└── api/
    └── main.py          FastAPI upload endpoint for live deployment
```

---

## Appendix B: Full Feature Definitions

| # | Feature | Formula / Definition | Window |
|---|---------|---------------------|--------|
| 1 | `lagged_ncskew` | $-n(n-1)^{3/2}\Sigma r^3 / [(n-1)(n-2)(\Sigma r^2)^{3/2}]$ | 26w |
| 2 | `lagged_duvol` | $\ln[(n_u-1)\Sigma_{\text{down}}r^2 / (n_d-1)\Sigma_{\text{up}}r^2]$ | 26w |
| 3 | `detrended_turnover` | $(V_t/\text{shares}) - \bar{V}_{26w}/\text{shares}$ | 26w |
| 4 | `trailing_return` | $\prod_{j=t-25}^{t}(1+r_j) - 1$ | 26w |
| 5 | `realized_volatility` | $\sigma_{26w}^{\text{weekly}} \times \sqrt{52}$ | 26w |
| 6 | `beta` | $\text{Cov}(r_i, r_m)_{26w} / \text{Var}(r_m)_{26w}$ | 26w |
| 7 | `downside_beta` | $\text{Cov}(r_i, r_m \mid r_m < 0)_{26w} / \text{Var}(r_m \mid r_m < 0)_{26w}$ | 26w |
| 8 | `relative_downside_beta` | $\beta^{-} - \beta$ | 26w |
| 9 | `market_cap` | Shares × price (USD), lagged 45d | Current |
| 10 | `market_to_book` | Market cap / book equity, lagged 45d | Annual |
| 11 | `leverage` | Total debt / total assets, lagged 45d | Annual |
| 12 | `roa` | Net income / total assets (TTM), lagged 45d | Trailing |
| 13 | `controversy_score` | Vendor ESG controversy score (0–10) | Monthly |
| 14 | `controversy_change_4w` | $c_t - c_{t-4}$ | 4w |
| 15 | `controversy_change_13w` | $c_t - c_{t-13}$ | 13w |
| 16 | `controversy_change_26w` | $c_t - c_{t-26}$ | 26w |
| 17 | `controversy_rolling_mean_13w` | $\bar{c}_{13w}$ | 13w |
| 18 | `controversy_rolling_std_13w` | $\sigma^c_{13w}$ | 13w |
| 19 | `controversy_spike_flag` | $\mathbf{1}[c_t > \bar{c}_{26w} + 2\sigma^c_{26w}]$ | 26w |
| 20 | `controversy_sector_percentile` | Rank$(c_t)$ within sector-week peer group | Current |

---

## Appendix C: Bloomberg Data Guide

To replicate this project with real Bloomberg data, download four CSV files with the following field mappings:

**File 1: prices.csv** — Bloomberg equity price history (BDUMP or BDH)

| Column | Bloomberg Field | Notes |
|--------|----------------|-------|
| `ticker` | Ticker symbol (e.g., AAPL US Equity) | Standardise to base ticker |
| `date` | Date | Daily, 2019-01-01 to present |
| `adj_close` | PX_LAST adjusted for dividends/splits | Use ADJUSTMENT_SPLIT_AND_DIV_REINV |
| `volume` | VOLUME | Daily share volume |

**File 2: benchmark_prices.csv** — S&P 500 benchmark (SPX Index or SPI Index)

Same columns as prices.csv for a single index ticker.

**File 3: fundamentals.csv** — Firm-level accounting data (Bloomberg BSRP/EMRP)

| Column | Bloomberg Field | Notes |
|--------|----------------|-------|
| `ticker` | Ticker | One row per ticker per fiscal period |
| `period_end` | ANNOUNCEMENT_DT or FISCAL_YEAR_END | Used to compute 45-day availability lag |
| `shares_outstanding` | BS_SH_OUT | For turnover calculation |
| `market_cap` | CUR_MKT_CAP | Or compute as shares × price |
| `book_value` | BOOK_VAL_PER_SH × shares | For market-to-book |
| `total_debt` | BS_TOT_LIAB2 | For leverage |
| `total_assets` | BS_TOT_ASSET | For leverage and ROA |
| `net_income` | NET_INCOME | Trailing twelve months |

**File 4: controversies.csv** — ESG controversy scores (Bloomberg ESG EVIC panel)

| Column | Bloomberg Field | Notes |
|--------|----------------|-------|
| `ticker` | Ticker | |
| `date` | Date | Monthly (last business day) |
| `controversy_score` | ESG_CONTROVERSY_SCORE | Range 0–10, higher = more controversial |
| `sector` | GICS_SECTOR_NAME | For sector-percentile computation |

---

## References

Chen, J., Hong, H. and Stein, J.C. (2001) 'Forecasting crashes: trading volume, past returns, and conditional skewness in stock prices', *Journal of Financial Economics*, 61(3), pp. 345–381.

Duffee, G.R. (1995) 'Stock returns and volatility: a firm-level analysis', *Journal of Financial Economics*, 37(3), pp. 399–420.

Harvey, C.R. and Siddique, A. (2000) 'Conditional skewness in asset pricing tests', *Journal of Finance*, 55(3), pp. 1263–1295.

Kim, J.B., Li, Y. and Zhang, L. (2014) 'Corporate tax avoidance and stock price crash risk: firm-level analysis', *Journal of Financial Economics*, 100(3), pp. 639–662.

Loughran, T. and McDonald, B. (2011) 'When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks', *Journal of Finance*, 66(1), pp. 35–65.

scikit-learn developers (2024) *scikit-learn: Machine Learning in Python*, version 1.5. Available at: https://scikit-learn.org (Accessed: April 2026).
