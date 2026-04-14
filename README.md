# Lean Crash-Risk Backend MVP

This project ingests local Bloomberg-style CSV or Excel exports, builds a time-aware feature panel, trains a `LogisticRegression` crash-risk classifier, and writes stock-level crash-risk scores.

## Expected Inputs

Place files in `data/raw/`:

- `prices.csv` or `prices.xlsx`: `ticker`, `date`, `adj_close`, `volume`
- `benchmark_prices.csv` or `benchmark_prices.xlsx`: `date`, `benchmark_close`
- `fundamentals.csv` or `fundamentals.xlsx`: `ticker`, `period_end`, `market_cap`, `shares_outstanding`, `market_to_book`, `leverage`, `roa`
- `controversies.csv` or `controversies.xlsx`: `ticker`, `date`, `sector`, `controversy_score`

The MVP assumes higher `controversy_score` means higher controversy risk.

## Run

```powershell
C:\Users\itsab\anaconda3\python.exe -m crashrisk.cli --raw-dir data/raw
```

Outputs:

- `data/processed/feature_panel.parquet`
- `data/processed/model_dataset.parquet`
- `outputs/stock_scores.csv`
- `outputs/price_history.csv`
- `outputs/price_scenarios.csv`
- `outputs/esg_model_comparison.csv`

## Demo

Run the full demo backend in one command:

```powershell
C:\Users\itsab\anaconda3\python.exe -m crashrisk.demo
```

Serve the project folder so the frontend can read `outputs/stock_scores.csv`:

```powershell
C:\Users\itsab\anaconda3\python.exe -m http.server 8000
```

Open:

```text
http://localhost:8000/frontend/
```

## Frontend

Open `frontend/index.html` in your browser.

The dashboard reads `outputs/stock_scores.csv` when served from a local static server. If you open the HTML file directly, use the file picker to load `outputs/stock_scores.csv`.

The frontend expects this score file:

- `outputs/stock_scores.csv`: `ticker`, `as_of_date`, `crash_probability`, `risk_bucket`, `top_drivers`
- `outputs/price_history.csv`: `ticker`, `date`, `adj_close`
- `outputs/price_scenarios.csv`: `ticker`, `as_of_date`, `latest_price`, `horizon_weeks`, `price_p05`, `price_p50`, `price_p95`, `crash_probability`, `risk_bucket`, `scenario_method`
- `outputs/esg_model_comparison.csv`: out-of-sample baseline vs full ESG controversy metrics for `roc_auc`, `precision_at_top_bucket`, and `crash_capture_at_top_bucket`

The price graph is a 13-week scenario range based on historical volatility and crash probability. It is not a single point price forecast.

The ESG model comparison report is the key research artifact for testing whether controversy features add signal beyond the non-ESG crash-risk benchmark.

## Test

```powershell
C:\Users\itsab\anaconda3\python.exe -m pytest
```
