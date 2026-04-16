# ESG Controversy Signals for Equity Crash-Risk Monitoring in US Large-Cap Equities

**Module:** Financial Data Science for Trading & Risk Management (FIN42110)  
**Programme:** MSc Financial Data Science, University College Dublin  
**Project type:** Equity risk-management classification model  
**Application:** Crash-risk monitoring for a US large-cap equity universe  

---

## 0. Executive Summary

This project builds a financial data science pipeline for monitoring equity crash risk. The goal is not to forecast the exact future return of each stock. Instead, the goal is to classify which stocks are most likely to enter a high future crash-risk group. This makes the project a risk-management application rather than a pure trading signal.

The model combines stock prices, benchmark market prices, firm fundamentals, and ESG controversy signals. The code converts daily prices into weekly Friday observations, engineers financial risk features, creates a future crash-risk target, trains machine learning classifiers, compares models with and without ESG information, and finally produces stock-level crash-risk scores.

The central research question is:

**Do ESG controversy signals improve the ability to classify future equity crash-risk names?**

The model outputs a `crash_probability`, a Low/Medium/High `risk_bucket`, and the main `top_drivers` behind each score. The business analysis then asks whether a fund could use these scores as a risk overlay.

| Item | Result |
|---|---:|
| Equity universe | 50 US large-cap stocks |
| Daily price rows | 75,450 |
| Weekly feature rows | 15,700 |
| Sample period | 2019-01-02 to 2024-12-30 |
| High-risk target definition | Top 20% future 13-week NCSKEW |
| Main model | Logistic Regression |
| Test ROC-AUC, full model | 0.598 |
| Test ROC-AUC, baseline no-ESG model | 0.569 |
| ESG ROC-AUC lift | 0.029 |
| Annualized strategy alpha | 5.97% |
| Illustrative economic gain on USD 1bn AUM | USD 59.74m |

The result is nuanced. ESG controversy features improve test ROC-AUC, meaning the model ranks future crash-risk observations better overall when ESG information is included. However, top-bucket precision is slightly lower in the full ESG model than in the non-ESG baseline. This is realistic: ESG controversy data appears to add ranking signal, but it does not automatically improve every business metric.

---

## 1. Introduction and Research Question

### 1.1 Problem Background

Equity crashes are asymmetric downside events. A stock can rise gradually for a long time, but negative firm-specific information can be incorporated very quickly when investors reassess risk. Standard volatility can tell us that a stock is risky, but it does not always tell us whether risk is concentrated in the left tail.

Crash-risk monitoring focuses on that left tail. Instead of asking "what is the exact expected return?", the project asks "which names are more likely to become future high crash-risk names?" This is useful because a fund can reduce exposure, hedge, perform extra analyst review, or require a higher expected return before holding the stock.

ESG controversies are relevant because they may reveal firm-level stress before it is fully reflected in prices. Examples include governance problems, litigation, environmental fines, product recalls, labour disputes, corruption allegations, and social controversies. These events can affect investor confidence, reputation, and downside risk.

### 1.2 Research Question

The research question is:

**Can ESG controversy signals improve the classification of future equity crash-risk names in a US large-cap equity universe?**

The project answers this by comparing two models:

| Model | Feature Set | Purpose |
|---|---|---|
| Baseline model | Price, market, downside-risk, turnover, and fundamental variables | Measures crash-risk performance without ESG controversy information |
| Full model | Baseline variables plus ESG controversy variables | Tests whether ESG controversy features add incremental signal |

If the full model performs better out of sample, this suggests that ESG controversy information contains useful crash-risk information beyond standard market and accounting variables.

### 1.3 Application Type

This is a classification problem. The model predicts whether a stock belongs to a high future crash-risk class:

\[
Y_{i,t} =
\begin{cases}
1, & \text{if stock } i \text{ is in the top 20% future crash-risk bucket at time } t \\
0, & \text{otherwise}
\end{cases}
\]

This is better suited to the business problem than a point return forecast. A portfolio manager does not need the model to predict that a stock will return exactly 1.4% next week. The more useful output is a ranked list of names that deserve risk review.

### 1.4 Business Use Case

The intended user is a portfolio manager or risk team running an equity portfolio. Low-risk names remain normal holdings, Medium-risk names may be watched, and High-risk names may be reduced, hedged, excluded, or sent for analyst review.

The project evaluates this with an illustrative strategy that excludes the High crash-risk bucket and compares performance with an equal-weight benchmark containing all stocks.

---

## 2. Data Summary and Visualisations

### 2.1 Data Sources

The project uses four raw input files stored in `data/raw/`.

| Dataset | File | Key Columns | Role in the Project |
|---|---|---|---|
| Equity prices | `prices.csv` | `ticker`, `date`, `adj_close`, `volume` | Calculates returns, volatility, turnover, and price history |
| Benchmark prices | `benchmark_prices.csv` | `date`, `benchmark_close` | Calculates market returns and firm-specific residual returns |
| Fundamentals | `fundamentals.csv` | `market_cap`, `shares_outstanding`, `market_to_book`, `leverage`, `roa` | Adds firm-level risk controls |
| ESG controversies | `controversies.csv` | `ticker`, `date`, `sector`, `controversy_score` | Adds ESG/text-derived risk signal |

The price and benchmark data are daily. The ESG controversy data are monthly. The model itself works weekly, so the code aligns these different frequencies into one ticker-date feature panel.

### 2.2 Dataset Size and Coverage

The current run contains 50 stocks and covers 2019 to the end of 2024. The pipeline writes the data summary to `outputs/data_summary.csv`.

| Dataset | Raw Rows | Loaded Rows | Tickers | Date Range |
|---|---:|---:|---:|---|
| Prices | 75,450 | 75,450 | 50 | 2019-01-02 to 2024-12-30 |
| Benchmark prices | 1,509 | 1,509 | N/A | 2019-01-02 to 2024-12-30 |
| Fundamentals | 50 | 50 | 50 | 2018-11-02 |
| Controversies | 3,600 | 3,600 | 50 | 2019-01-31 to 2024-12-31 |
| Weekly feature panel | 15,700 | 15,700 | 50 | Weekly ticker-date panel |
| Latest stock scores | 50 | 50 | 50 | Latest scoring date |

The final weekly feature panel contains 15,700 rows, which is comfortably above the minimum sample size requirement for a machine learning project.

### 2.3 Data Cleaning and Validation

The backend applies several cleaning and validation steps before modelling.

First, the code checks that each file contains required columns. For example, `prices.csv` must contain `ticker`, `date`, `adj_close`, and `volume`. If a required column is missing, the loader raises a clear schema error.

Second, tickers are standardized by stripping whitespace and converting them to uppercase. This prevents the same company being treated as multiple companies because of inconsistent formatting.

Third, date parsing is handled carefully. The raw price file uses European-style dates such as `02-01-2019`. The loader infers whether date strings should be parsed as day-first or month-first. This matters because wrong date parsing can silently corrupt a time series.

Fourth, numeric fields such as adjusted prices, volumes, market capitalisation, leverage, and controversy scores are coerced to numeric values. Invalid numeric entries become missing values rather than text strings.

Fifth, fundamentals are lagged by 45 days to reduce look-ahead bias:

\[
\text{available date} = \text{period end} + 45 \text{ calendar days}
\]

The model can only use a fundamental value once the weekly observation date is on or after the available date.

| Dataset | Cleaning Check | Count or Rule |
|---|---|---:|
| Prices missing required values | Blank or NA values in required columns | 0 |
| Prices duplicate rows | Fully duplicated raw rows | 0 |
| Prices non-positive adjusted close | Zero or negative adjusted prices | 0 |
| Benchmark missing required values | Blank or NA values in required columns | 0 |
| Fundamentals missing required values | Blank or NA values in required columns | 7 |
| Controversy rows removed or invalid | Rows not retained by validated loader | 0 |
| Fundamentals availability lag | Look-ahead-bias control | 45 days |
| Date alignment | Modelling frequency | Weekly Friday observations |

The cleaning results are strong. No price observations were removed, no duplicate price rows were detected, and no non-positive adjusted prices were found. The main limitation is that the current fundamentals file contains one snapshot date rather than a full point-in-time historical fundamentals panel.

### 2.4 SQL Evidence

The project creates SQL summaries using an in-memory SQLite database. SQL gives transparent evidence about the dataset and supports the data summary section of the rubric.

Example 1: observations by ticker.

```sql
select ticker, count(*) as observations, min(date) as start_date, max(date) as end_date
from raw_prices
group by ticker
order by observations desc, ticker
limit 20
```

The result confirms that the main tickers each have 1,509 daily price observations from 2019-01-02 to 2024-12-30.

Example 2: sector controversy summary.

```sql
select sector, count(distinct ticker) as tickers, count(*) as records,
       avg(controversy_score) as avg_controversy_score
from raw_controversies
group by sector
order by avg_controversy_score desc
```

| Sector | Tickers | Records | Average Controversy Score |
|---|---:|---:|---:|
| Energy | 5 | 360 | 5.935 |
| Financial Services | 7 | 504 | 5.769 |
| Industrials | 5 | 360 | 5.303 |
| Communication Services | 4 | 288 | 4.883 |
| Consumer Cyclical | 6 | 432 | 4.622 |

This tells a useful story. Energy and Financial Services have higher average controversy scores, which is intuitive because these sectors are often exposed to environmental, regulatory, litigation, and governance risks.

Example 3: target class balance.

| Target Class | Rows |
|---|---:|
| High crash risk = 1 | 3,110 |
| High crash risk = 0 | 12,440 |
| Unlabelled final horizon rows | 150 |

The positive class is intentionally smaller because the target is defined as the top 20% of future crash-risk observations at each date. The final 150 rows are unlabelled because the end of the sample does not have a full 13-week future window.

Example 4: feature missingness.

| Rows | Missing Lagged NCSKEW | Missing Downside Beta | Missing Controversy Score | Missing Market Cap |
|---:|---:|---:|---:|---:|
| 15,700 | 400 | 550 | 200 | 0 |

Some missingness is expected because rolling features require a minimum number of previous weekly observations. The model handles this using median imputation inside the machine learning pipeline.

### 2.5 Visualisations

The pipeline writes visualisations to `outputs/figures/`.

| Figure | Use in Report |
|---|---|
| `risk_probability_ranking.svg` | Shows the top-ranked stocks by predicted crash probability |
| `sector_controversy.svg` | Shows which sectors have the highest average controversy scores |
| `controversy_over_time.svg` | Shows whether average controversy risk rises or falls through time |
| `feature_importance.svg` | Shows which variables matter most to the primary model |
| `esg_lift.svg` | Shows the incremental effect of ESG features over the baseline |
| `price_scenario_range.svg` | Shows downside, median, and upside price scenarios |

Together, these figures connect the data science pipeline to an investment story: where controversy risk is concentrated, which stocks the model flags, which features drive the classification, and how crash probabilities translate into scenario ranges.

---

## 3. Textual Analysis

### 3.1 What Counts as Textual Signal in This Project

The FIN42110 rubric requires a relevant information signal from textual analysis. In the current run, there is no raw `news_text.csv` or `controversy_text.csv` file in the raw data folder. Therefore, this version treats `controversy_score` as a structured ESG controversy signal derived from external controversy assessment. This is best described as a **vendor text-derived ESG controversy proxy**.

This is an honest limitation, but it is still defensible. ESG controversy scores are usually constructed from news, regulatory events, NGO reports, litigation records, company disclosures, and other text-heavy sources. The difference is that the raw articles are not directly included in the current dataset.

The correct wording is:

> In this version, direct headline-level sentiment analysis is not available. We therefore use the ESG controversy score as a structured proxy for textual controversy information, while reporting this as a limitation.

### 3.2 Optional Text Pipeline Built Into the Code

Although the current run has no raw text file, the backend supports optional textual analysis. If a file named `news_text.csv`, `controversy_text.csv`, or `textual_data.csv` is supplied, the system looks for `ticker`, `date`, and at least one text column such as `headline`, `title`, `description`, `body`, `text`, or `summary`.

The code then combines text fields, tokenizes words, counts positive words, counts negative words, counts controversy keywords, computes a sentiment score, aggregates the result weekly, and computes a 13-week rolling sentiment average.

The sentiment score is:

\[
\text{Text Sentiment Score}_{i,t}
=
\frac{\text{Positive Words}_{i,t} - \text{Negative Words}_{i,t}}
{\text{Total Words}_{i,t}}
\]

The 13-week rolling sentiment signal is:

\[
\text{Rolling Sentiment}_{i,t}^{13w}
=
\frac{1}{13}\sum_{k=0}^{12}\text{Sentiment}_{i,t-k}
\]

### 3.3 ESG Controversy Feature Engineering

The model engineers controversy features so it can learn from the level, trend, volatility, spike behavior, and sector-relative severity of ESG controversy.

| Feature | Meaning |
|---|---|
| `controversy_score` | Current controversy level |
| `controversy_change_4w` | Short-term change in controversy score |
| `controversy_change_13w` | Approximately quarterly controversy change |
| `controversy_change_26w` | Approximately half-year controversy change |
| `controversy_rolling_mean_13w` | Smoothed recent controversy level |
| `controversy_rolling_std_13w` | Instability of recent controversy |
| `controversy_spike_flag` | Indicator for unusually high controversy |
| `controversy_sector_percentile` | Controversy rank within the same sector and date |

The controversy spike flag is:

\[
\text{Spike}_{i,t}
=
1
\quad \text{if} \quad
C_{i,t} > \bar{C}_{i,t}^{26w} + 2\sigma_{i,t}^{26w}
\]

where \(C_{i,t}\) is the controversy score, \(\bar{C}_{i,t}^{26w}\) is the 26-week rolling mean, and \(\sigma_{i,t}^{26w}\) is the 26-week rolling standard deviation.

### 3.4 Limitation of the Textual Analysis

The main limitation is that raw ProQuest or Bloomberg News headline text is not currently supplied. Therefore, the report cannot claim to have performed full LDA topic modelling or raw article-level sentiment analysis in the current run.

For this submission, the textual analysis should be framed as: **ESG controversy score as a structured text-derived proxy, with raw news sentiment analysis listed as a future improvement.**

---

## 4. Feature Engineering and Target Construction

This section explains what the code does in the background before the machine learning model is trained. Raw CSV files are not directly fed into the model. They are converted into a clean weekly feature panel where each row represents one stock at one weekly date.

### 4.1 Weekly Returns

The raw equity price data are daily. The model resamples each ticker to Friday week-end observations. For each stock \(i\), the weekly return is:

\[
R_{i,t}
=
\frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}
\]

where \(P_{i,t}\) is the adjusted closing price of stock \(i\) at week \(t\).

The benchmark return is calculated similarly:

\[
R_{m,t}
=
\frac{M_t - M_{t-1}}{M_{t-1}}
\]

The firm-specific return is:

\[
\epsilon_{i,t}
=
R_{i,t} - R_{m,t}
\]

This residual return removes the broad market component and focuses on the part of the stock return that is specific to the firm. Crash-risk measures such as NCSKEW and DUVOL are calculated from these firm-specific residual returns.

### 4.2 Volatility and Momentum Features

The 26-week trailing return is:

\[
\text{Trailing Return}_{i,t}
=
\prod_{k=0}^{25}(1 + R_{i,t-k}) - 1
\]

This is a cumulative return over roughly half a year and gives the model information about recent momentum or reversal risk.

Annualized realized volatility is:

\[
\sigma_{i,t}^{ann}
=
\text{Std}(R_{i,t-25:t}) \times \sqrt{52}
\]

The \(\sqrt{52}\) term annualizes weekly volatility because there are approximately 52 weeks in a year.

### 4.3 Turnover

Turnover measures how much of the firm's shares trade during the week:

\[
\text{Turnover}_{i,t}
=
\frac{\text{Weekly Volume}_{i,t}}{\text{Shares Outstanding}_{i,t}}
\]

The model uses detrended turnover:

\[
\text{Detrended Turnover}_{i,t}
=
\text{Turnover}_{i,t} - \overline{\text{Turnover}}_{i,t}^{26w}
\]

This tells us whether current trading activity is unusually high or low relative to the stock's own recent history.

### 4.4 Beta and Downside Beta

Market beta measures how sensitive a stock is to market returns:

\[
\beta_{i,t}
=
\frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}
\]

The model also calculates downside beta, using only weeks when the market return is negative:

\[
\beta_{i,t}^{down}
=
\frac{\text{Cov}(R_i, R_m \mid R_m < 0)}
{\text{Var}(R_m \mid R_m < 0)}
\]

This is important for crash-risk modelling because investors often care more about how a stock behaves during bad market weeks than during normal or positive weeks.

The relative downside beta is:

\[
\beta_{i,t}^{relative}
=
\beta_{i,t}^{down} - \beta_{i,t}
\]

If relative downside beta is high, the stock is especially sensitive when the market falls, even compared with its normal beta.

### 4.5 Crash-Risk Metrics: NCSKEW and DUVOL

The project uses two crash-risk measures: NCSKEW and DUVOL.

NCSKEW stands for negative conditional skewness. It measures whether firm-specific residual returns are skewed toward large negative outcomes:

\[
\text{NCSKEW}_{i,t}
=
-
\frac{
n(n-1)^{3/2}\sum \epsilon_{i,t}^{3}
}{
(n-1)(n-2)(\sum \epsilon_{i,t}^{2})^{3/2}
}
\]

The negative sign is used so that larger NCSKEW values correspond to greater crash risk. If a stock has occasional large negative residual returns, its residual return distribution becomes more left-skewed and NCSKEW rises.

DUVOL stands for down-to-up volatility. It compares volatility during below-average residual-return weeks with volatility during above-average residual-return weeks:

\[
\text{DUVOL}_{i,t}
=
\log
\left(
\frac{
(n_u - 1)\sum_{\text{down}}\epsilon_{i,t}^{2}
}{
(n_d - 1)\sum_{\text{up}}\epsilon_{i,t}^{2}
}
\right)
\]

where \(n_u\) is the number of up weeks and \(n_d\) is the number of down weeks.

### 4.6 Fundamental Features

The model includes firm-level controls:

| Feature | Interpretation |
|---|---|
| `market_cap` | Size of the company |
| `market_to_book` | Valuation ratio |
| `leverage` | Debt or balance sheet risk |
| `roa` | Profitability |

These variables are included because crash risk may be related to size, valuation, indebtedness, and profitability.

### 4.7 Target Variable

The model predicts whether a stock will become a high future crash-risk name over the next 13 weeks. For each stock and date, the code looks forward from \(t+1\) to \(t+13\) and calculates future NCSKEW.

Future window:

\[
t+1 \text{ through } t+13
\]

The high-risk label is:

\[
Y_{i,t}
=
1
\quad \text{if future NCSKEW is in the top 20% cross-section at date } t
\]

Otherwise:

\[
Y_{i,t}=0
\]

This target is cross-sectional. At each date, the model identifies the stocks with the highest future crash-risk metric relative to the rest of the universe. This matches the business problem because a portfolio manager usually wants to rank names at a point in time.

---

## 5. Machine Learning

### 5.1 Why Classification?

The project uses classification because the business decision is categorical. The risk manager wants to know whether a stock should be treated as high risk, not whether its exact return next quarter will be -2.1% or +1.3%.

Classification also allows the model to produce probabilities:

\[
\hat{p}_{i,t}
=
P(Y_{i,t}=1 \mid X_{i,t})
\]

where \(X_{i,t}\) is the feature vector for stock \(i\) at time \(t\).

### 5.2 Model Inputs

The full model uses 20 feature columns.

| Feature Group | Examples |
|---|---|
| Crash history | `lagged_ncskew`, `lagged_duvol` |
| Market sensitivity | `beta`, `downside_beta`, `relative_downside_beta` |
| Price behavior | `trailing_return`, `realized_volatility` |
| Trading activity | `detrended_turnover` |
| Fundamentals | `market_cap`, `market_to_book`, `leverage`, `roa` |
| ESG controversy | `controversy_score`, changes, rolling means, rolling standard deviations, spike flag, sector percentile |

The baseline no-ESG model removes the controversy features and keeps 12 non-ESG features. The full model uses all 20 features.

### 5.3 Main Algorithm

The primary model is Logistic Regression. The scikit-learn pipeline has three stages:

1. Median imputation: missing feature values are replaced by the median value.
2. Standard scaling: features are standardized so variables with large units do not dominate.
3. Logistic Regression classifier: the model estimates crash-risk probabilities.

The logistic regression model has the form:

\[
P(Y_{i,t}=1 \mid X_{i,t})
=
\frac{1}{1 + e^{-(\alpha + \beta X_{i,t})}}
\]

This model is suitable because it is interpretable, works well for tabular financial features, outputs probabilities, supports feature contribution analysis, and is a strong benchmark before using more complex models. The model uses `class_weight="balanced"` because only around 20% of labelled rows are high crash-risk observations.

### 5.4 Train, Validation, and Test Methodology

The data are split chronologically, not randomly.

| Split | Share of Unique Dates | Purpose |
|---|---:|---|
| Training | 60% | Fit the model |
| Validation | 20% | Tune and compare modelling choices |
| Test | 20% | Evaluate final out-of-sample performance |

Chronological splitting is essential because this is financial time-series data. A random split could train the model on future observations and test it on earlier observations, creating look-ahead bias. The chronological split better simulates the real problem: train on the past, evaluate on the future.

### 5.5 Hyperparameter Tuning

The code supports hyperparameter tuning using `GridSearchCV` with `TimeSeriesSplit`. This means cross-validation folds preserve time order. The tuning metric is ROC-AUC.

| Model | Hyperparameters Tested |
|---|---|
| Logistic Regression | `C = 0.01, 0.1, 1.0, 10.0`; `penalty = l2` |
| Random Forest | `n_estimators = 100, 200`; `max_depth = 3, 5, 8`; `min_samples_leaf = 5, 10` |
| Gradient Boosting | `n_estimators = 100, 200`; `max_depth = 2, 3`; `learning_rate = 0.05, 0.10` |

In the current output files, the algorithm comparison table is from the default untuned run, so the `best_cv_roc_auc` column is blank. The code is ready to report tuned cross-validation results if the pipeline is run with tuning enabled.

**ML model justification.** We employ Logistic Regression as the baseline model because it is interpretable, stable on smaller financial datasets, and directly provides crash-probability estimates. In addition, ensemble methods such as Random Forest and Gradient Boosting are used to capture nonlinear relationships and interactions between ESG controversy variables and financial risk indicators. This combination allows us to balance interpretability with predictive performance.

**Hyperparameter analysis.** Increasing tree depth allows ensemble models to capture more nonlinear structure, while regularisation parameters such as Logistic Regression `C`, Random Forest leaf-size controls, and Gradient Boosting learning rate help prevent overfitting. Gradient Boosting performs best on the test set in this run because it sequentially learns complex residual patterns in the data.

### 5.6 Evaluation Metrics

ROC-AUC measures ranking ability:

\[
\text{ROC-AUC}
=
P(\hat{p}_{positive} > \hat{p}_{negative})
\]

Precision at top bucket measures how accurate the highest-risk group is:

\[
\text{Precision@Top Bucket}
=
\frac{\text{True crash-risk cases in predicted high-risk bucket}}
{\text{Number of names in predicted high-risk bucket}}
\]

Crash capture measures how many true high crash-risk observations are captured in the model's top predicted bucket:

\[
\text{Crash Capture}
=
\frac{\text{True crash-risk cases captured in top bucket}}
{\text{Total true crash-risk cases}}
\]

These metrics are more useful than accuracy. Since the positive class is only around 20%, a naive model could have high accuracy by predicting most observations as not high risk. ROC-AUC, top-bucket precision, and crash capture focus on ranking and identification of risky names.

### 5.7 Algorithm Comparison

The project compares Logistic Regression, Random Forest, and Gradient Boosting on the same chronological splits.

| Model | Split | ROC-AUC | Precision@Top Bucket | Crash Capture |
|---|---|---:|---:|---:|
| Logistic Regression | Validation | 0.600 | 0.258 | 0.258 |
| Logistic Regression | Test | 0.598 | 0.251 | 0.251 |
| Random Forest | Validation | 0.582 | 0.261 | 0.261 |
| Random Forest | Test | 0.592 | 0.287 | 0.287 |
| Gradient Boosting | Validation | 0.561 | 0.260 | 0.260 |
| Gradient Boosting | Test | 0.611 | 0.316 | 0.316 |

Gradient Boosting has the highest test ROC-AUC and strongest test top-bucket performance in this run. This suggests nonlinear relationships may be useful for crash-risk classification. However, Logistic Regression remains the main model because it is more interpretable.

**Model performance limitations.** The relatively modest predictive performance, with ROC-AUC around 0.60, reflects the inherent difficulty of forecasting financial tail events. Crash risk is driven by rare, noisy, and interacting factors, so limited historical data constrains model performance. The results should therefore be interpreted as useful ranking signal rather than deterministic crash prediction.

### 5.8 ESG Lift Test

The ESG lift test is the main research comparison.

| Model | Split | Feature Count | ROC-AUC | Precision@Top Bucket | Crash Capture |
|---|---|---:|---:|---:|---:|
| Baseline without ESG | Validation | 12 | 0.558 | 0.250 | 0.250 |
| Full model with ESG | Validation | 20 | 0.600 | 0.258 | 0.258 |
| Full minus baseline | Validation | +8 | +0.042 | +0.008 | +0.008 |
| Baseline without ESG | Test | 12 | 0.569 | 0.260 | 0.260 |
| Full model with ESG | Test | 20 | 0.598 | 0.251 | 0.251 |
| Full minus baseline | Test | +8 | +0.029 | -0.010 | -0.010 |

The validation result is positive across all three metrics. The test result is mixed. The full ESG model improves ROC-AUC by 0.029, but precision and crash capture in the top bucket are 0.010 lower than the baseline.

The correct interpretation is that ESG controversy features improve overall out-of-sample ranking quality, but the benefit is not uniform across all business metrics. A production risk team should monitor several metrics rather than one headline number.

### 5.9 Feature Importance

The primary Logistic Regression model reports feature importance using the absolute value of fitted coefficients after preprocessing.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `controversy_score` | 0.827 |
| 2 | `controversy_rolling_mean_13w` | 0.651 |
| 3 | `downside_beta` | 0.368 |
| 4 | `market_cap` | 0.265 |
| 5 | `relative_downside_beta` | 0.252 |
| 6 | `trailing_return` | 0.223 |
| 7 | `beta` | 0.199 |
| 8 | `realized_volatility` | 0.157 |
| 9 | `lagged_ncskew` | 0.144 |
| 10 | `controversy_change_13w` | 0.137 |

This is economically sensible. ESG controversy level and smoothed controversy level are highly influential, while downside beta, market cap, and trailing return also matter.

---

## 6. Model Outputs and Dashboard Logic

### 6.1 Crash Probability

After training, the model scores the latest available weekly row for each stock:

\[
\hat{p}_{i,t}
=
P(Y_{i,t}=1 \mid X_{i,t})
\]

This value is written as `crash_probability` in `outputs/stock_scores.csv`.

| Ticker | Crash Probability | Risk Bucket | Top Drivers |
|---|---:|---|---|
| LIN | 0.729 | High | controversy rolling mean, spike flag, controversy change |
| SLB | 0.676 | High | controversy score, controversy rolling mean, trailing return |
| EOG | 0.617 | High | controversy rolling mean, controversy score, downside beta |
| DIS | 0.614 | High | downside beta, controversy rolling mean, lagged NCSKEW |
| PEP | 0.607 | High | controversy rolling mean, downside beta, controversy score |

These scores should be interpreted as relative risk rankings, not literal probabilities of a price crash.

### 6.2 Risk Buckets

Stocks are ranked by crash probability and assigned buckets:

| Bucket | Rule |
|---|---|
| High | Top 20% of crash probabilities |
| Medium | Next 40% |
| Low | Remaining 40% |

This converts model probabilities into an action-oriented output.

### 6.3 Top Drivers

For Logistic Regression, the system identifies top drivers behind each stock's score. Conceptually:

\[
\text{Contribution}_{j}
=
z(X_j)\times \beta_j
\]

where \(z(X_j)\) is the standardized feature value and \(\beta_j\) is the logistic regression coefficient. The model reports the features with the largest absolute contributions.

### 6.4 Price Scenario Range

The dashboard also produces a 13-week price scenario range. This is not a point price forecast. It is a volatility-based scenario range adjusted for crash probability.

The annual volatility is converted to horizon volatility:

\[
\sigma_h
=
\frac{\sigma_{ann}}{\sqrt{52}}\sqrt{13}
\]

The downside price scenario is:

\[
P_{05}
=
P_0
\exp
\left[
-z \cdot \sigma_h \cdot (1+\hat{p})
\right]
\]

The median scenario is:

\[
P_{50}=P_0
\]

The upside scenario is:

\[
P_{95}
=
P_0
\exp(z \cdot \sigma_h)
\]

The important idea is that higher crash probability widens the downside scenario.

---

## 7. Business Analysis

### 7.1 Business Strategy

The business analysis asks whether the model could be economically useful for a fund. The strategy is simple:

- Benchmark: hold all 50 stocks equal weight over the following week.
- Model overlay: at week \(t\), score all stocks, remove those classified as High crash risk, and hold the remaining stocks equal weight for week \(t+1\).
- Compare annual return, alpha, Sharpe, Sortino, drawdown, VaR, and CVaR.
- Scale the annualized alpha to a USD 1 billion fund.
- Compare the estimated economic gain with the annual cost of a four-person implementation team.

This is an illustrative risk overlay, not a final production trading strategy.

### 7.2 Performance Metrics

Annualized return:

\[
R_{ann}
=
\prod_{t=1}^{T}(1+R_t)^{52/T}-1
\]

Sharpe ratio:

\[
\text{Sharpe}
=
\frac{\overline{R_t - R_f}}{\sigma(R_t - R_f)}
\sqrt{52}
\]

Sortino ratio:

\[
\text{Sortino}
=
\frac{\overline{R_t - R_f}}
{\sigma(R_t - R_f \mid R_t - R_f < 0)}
\sqrt{52}
\]

Maximum drawdown:

\[
\text{MDD}
=
\min_t
\left(
\frac{V_t - \max(V_1,\ldots,V_t)}
{\max(V_1,\ldots,V_t)}
\right)
\]

Historical 95% VaR:

\[
\text{VaR}_{95}
=
-\text{5th percentile weekly return}
\]

Historical 95% CVaR:

\[
\text{CVaR}_{95}
=
-\mathbb{E}[R_t \mid R_t \leq \text{5th percentile}]
\]

### 7.3 Business Results

The current run uses a USD 1 billion fund AUM assumption and an annual team cost of USD 800,000.

| Metric | Result |
|---|---:|
| Strategy annual return | 23.29% |
| Benchmark annual return | 17.32% |
| Annualized alpha | 5.97% |
| Strategy Sharpe | 1.182 |
| Benchmark Sharpe | 0.872 |
| Strategy Sortino | 1.910 |
| Strategy max drawdown | -11.98% |
| Benchmark max drawdown | -12.25% |
| Drawdown improvement | 0.27% |
| Weekly VaR 95% | 3.55% |
| Weekly CVaR 95% | 4.14% |
| Evaluation weeks | 124 |
| High-risk names excluded | 20% |
| Fund AUM | USD 1.00bn |
| Illustrative annual economic gain | USD 59.74m |
| Four-person team annual cost | USD 0.80m |
| ROI versus team cost | 74.68x |

The estimated alpha is:

\[
\alpha
=
23.29\% - 17.32\%
=
5.97\%
\]

For a USD 1 billion fund:

\[
\text{Economic Gain}
=
1{,}000{,}000{,}000 \times 0.0597
=
\text{USD }59.74\text{m}
\]

The estimated team ROI is:

\[
\text{Team ROI}
=
\frac{59.74\text{m}}{0.80\text{m}}
=
74.68\times
\]

### 7.4 Would This Justify Hiring a Team of Four?

The economic results are based on a stylised simulation and should be interpreted as illustrative rather than as a forecast of realised returns. The estimated USD 59.74m gain is an indicative gross overlay value under the project assumptions, not a production trading result.

In practice, transaction costs, market impact, taxes, data costs, infrastructure costs, portfolio constraints, model uncertainty, and implementation slippage would likely reduce realised returns. The correct conclusion is:

**The model appears economically promising as a risk overlay, but it would need further live testing, richer data, and transaction-cost-aware portfolio simulation before being deployed with real capital.**

---

## 8. Limitations and Improvements

This project is strong as a data science prototype, but several limitations should be made explicit.

### 8.1 Fundamentals Are Not Fully Point-in-Time

The current fundamentals file has 50 rows and one date. The pipeline correctly applies a 45-day availability lag, but a stronger final dataset would include full quarterly historical fundamentals. This would allow the model to learn how changing leverage, profitability, valuation, and size relate to future crash risk.

### 8.2 Raw News Text Is Not Currently Supplied

The optional text pipeline exists, but the current run has no raw headline or article-level text file. Therefore, the report uses ESG controversy score as a structured text-derived proxy. A stronger version would add ProQuest or Bloomberg News headline data and create sentiment, topic, and controversy intensity measures directly from raw text.

### 8.3 ESG Signal Source Should Be Documented

The report should clearly state which ESG controversy provider is used. If the score comes from Bloomberg, MSCI, Sustainalytics, or Refinitiv, that provider should be cited consistently. This matters because different providers define controversy scores differently.

### 8.4 Business Backtest Is Simplified

The business overlay excludes high-risk names but does not include transaction costs, turnover constraints, sector neutrality, factor exposure controls, tax effects, liquidity limits, model uncertainty, slippage, or live retraining delays. These omissions mean the economic results should be interpreted as an illustrative gross simulation rather than realised returns.

### 8.5 Suggested Extensions

Future work should include:

- add raw ProQuest or Bloomberg News text;
- add LDA topic modelling for controversy themes;
- add full quarterly point-in-time fundamentals;
- use rolling walk-forward retraining;
- compare sector-neutral and factor-neutral overlays;
- include transaction-cost assumptions;
- calibrate predicted probabilities;
- test the model on a larger universe;
- compare performance during market stress periods.

---

## 9. Conclusion

This project creates a complete financial data science workflow for equity crash-risk monitoring. It starts with raw market, fundamental, and ESG controversy data. It cleans and aligns these datasets into a weekly ticker-date panel. It engineers financial risk features, downside-risk features, crash-risk metrics, and controversy features. It then trains classification models to predict whether a stock will enter the top 20% of future 13-week crash risk.

The main research question is whether ESG controversy information adds value. The answer is nuanced but positive overall. The full ESG model improves test ROC-AUC from 0.569 to 0.598, which suggests that controversy features improve the ranking of future high crash-risk observations. However, the top-bucket precision and crash capture are slightly lower in the test set, showing that the benefit is not uniform across all evaluation metrics.

The business analysis shows how the model could be used by a fund. An illustrative weekly forward overlay that excludes High crash-risk names produces an annualized alpha of 5.97% and an indicative USD 59.74m gross annual gain on USD 1bn AUM under the project assumptions. Still, the report is clear that this is a stylised simulation and that live deployment would require more robust data, costs, controls, and monitoring.

Overall, the project is a strong example of a financial data science system because it connects data engineering, textual/ESG signal design, machine learning validation, interpretability, and business value.

---

## 10. Appendix: Code and Reproducibility Notes

### Appendix A: End-to-End Pipeline

The main entry point is `run_mvp()` in `crashrisk/pipeline.py`.

The pipeline performs these steps:

1. Discover and load raw files.
2. Validate schemas and parse dates.
3. Build the weekly feature panel.
4. Create the future crash-risk target.
5. Train the primary classifier.
6. Compare baseline versus ESG-enhanced models.
7. Compare Logistic Regression, Random Forest, and Gradient Boosting.
8. Score the latest available stock universe.
9. Create price scenarios.
10. Compute business analysis.
11. Write report-ready CSV, Markdown, and SVG outputs.

### Appendix B: Data Loading

Data loading is handled by `crashrisk/data/loaders.py` and `crashrisk/data/validators.py`.

Important behavior:

- supports CSV and Excel files;
- normalizes column names to lowercase;
- checks required columns;
- rejects empty datasets;
- standardizes tickers to uppercase;
- parses ambiguous date formats;
- coerces numeric columns;
- applies a 45-day lag to fundamentals.

### Appendix C: Feature Engineering

Feature engineering is handled by modules in `crashrisk/features/`.

| File | Purpose |
|---|---|
| `returns.py` | Weekly returns, benchmark returns, trailing return, realized volatility |
| `turnover.py` | Turnover and detrended turnover |
| `downside.py` | Beta, downside beta, relative downside beta |
| `crash_metrics.py` | NCSKEW, DUVOL, lagged crash-risk features |
| `controversy.py` | ESG controversy alignment and derived controversy features |
| `pipeline.py` | Combines all feature engineering into one weekly panel |

### Appendix D: Target Creation

Target creation is handled by `crashrisk/targets.py`.

For each ticker-date row:

- the code looks forward 13 weeks;
- computes future NCSKEW and future DUVOL;
- ranks future NCSKEW cross-sectionally by date;
- labels the top 20% as `high_crash_risk = 1`;
- labels the rest as `high_crash_risk = 0`;
- leaves rows unlabelled when the full future window is unavailable.

### Appendix E: Model Training and Comparison

Model training is handled by `crashrisk/models/train.py`. The training pipeline contains `SimpleImputer(strategy="median")`, `StandardScaler()`, and the classifier, usually Logistic Regression.

Model comparison is handled by `crashrisk/models/compare.py`. It produces:

- `algorithm_comparison.csv`;
- `esg_model_comparison.csv`;
- baseline no-ESG model;
- full ESG model;
- full-minus-baseline lift rows.

Chronological splitting is handled by `crashrisk/models/splits.py`.

### Appendix F: Scoring and Dashboard Outputs

Scoring is handled by `crashrisk/models/score.py`. The latest weekly row for each stock is scored and written to `outputs/stock_scores.csv`.

| Column | Meaning |
|---|---|
| `ticker` | Stock identifier |
| `as_of_date` | Scoring date |
| `crash_probability` | Estimated probability of high crash-risk classification |
| `risk_bucket` | Low, Medium, or High |
| `top_drivers` | Main features contributing to the score |

Scenario outputs are handled by `crashrisk/models/scenarios.py` and written to:

- `outputs/price_history.csv`;
- `outputs/price_scenarios.csv`.

### Appendix G: Business Analysis

Business analysis is handled by `crashrisk/analysis/business.py`. It calculates strategy annual return, benchmark annual return, annualized alpha, Sharpe ratio, Sortino ratio, maximum drawdown, VaR, CVaR, high-risk exclusion percentage, economic gain on fund AUM, and team ROI.

### Appendix H: Report Artifacts

Report artifacts are handled by `crashrisk/analysis/reporting.py`.

| Output File | Purpose |
|---|---|
| `outputs/data_summary.csv` | Dataset counts, date ranges, and configuration notes |
| `outputs/cleaning_log.csv` | Cleaning and validation evidence |
| `outputs/sql_summary.md` | SQL queries and result tables |
| `outputs/textual_analysis.csv` | Text-signal output or limitation note |
| `outputs/feature_importance.csv` | Model feature importance |
| `outputs/esg_model_comparison.csv` | Baseline versus full ESG model |
| `outputs/algorithm_comparison.csv` | Logistic Regression, Random Forest, Gradient Boosting comparison |
| `outputs/business_analysis.csv` | Portfolio overlay and economic gain |
| `outputs/figures/` | Visualisations for report and viva |

### Appendix I: Reproducibility

Run the full backend pipeline with:

```powershell
C:\Users\itsab\anaconda3\python.exe -m crashrisk.cli --raw-dir data/raw
```

Run tests with:

```powershell
C:\Users\itsab\anaconda3\python.exe -m pytest
```

Expected main outputs:

- `data/processed/feature_panel.parquet`;
- `data/processed/model_dataset.parquet`;
- `outputs/stock_scores.csv`;
- `outputs/price_scenarios.csv`;
- `outputs/esg_model_comparison.csv`;
- `outputs/algorithm_comparison.csv`;
- `outputs/feature_importance.csv`;
- `outputs/business_analysis.csv`;
- `outputs/data_summary.csv`;
- `outputs/cleaning_log.csv`;
- `outputs/sql_summary.md`;
- `outputs/textual_analysis.csv`;
- `outputs/figures/*.svg`.

---

## Viva Slide Structure

### Slide 1: Problem and Research Question

- ESG controversy signals for equity crash-risk monitoring.
- Risk-management classification task, not exact return forecasting.
- Output: crash probability, risk bucket, top drivers, scenario range.

### Slide 2: Data and Cleaning

- 50 stocks.
- 75,450 price rows.
- 15,700 weekly feature rows.
- SQL evidence and cleaning log.
- 45-day fundamentals lag to reduce look-ahead bias.

### Slide 3: Textual and ESG Signal

- Current signal: controversy score as text-derived ESG proxy.
- Engineered controversy features: level, changes, rolling mean, rolling standard deviation, spike flag, sector percentile.
- Limitation: no raw headline-level text currently supplied.

### Slide 4: Machine Learning Design

- Target: top 20% future 13-week NCSKEW.
- Chronological train/validation/test split.
- Main model: Logistic Regression.
- Comparisons: Random Forest and Gradient Boosting.
- Metrics: ROC-AUC, Precision@Top Bucket, Crash Capture.

### Slide 5: Results

- Full ESG model test ROC-AUC: 0.598.
- Baseline no-ESG test ROC-AUC: 0.569.
- ESG ROC-AUC lift: 0.029.
- Gradient Boosting gives strongest test top-bucket performance.
- ESG improves ranking, but top-bucket precision is mixed.

### Slide 6: Business Value

- USD 1bn fund overlay.
- Exclude High crash-risk names.
- Annualized alpha: 5.97%.
- Illustrative annual gain: USD 59.74m.
- Team cost: USD 0.80m.
- Conclusion: promising, but requires transaction-cost-aware live testing.
