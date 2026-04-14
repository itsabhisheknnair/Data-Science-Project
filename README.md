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

## Netlify Demo Deployment

This repo is prepared for a static Netlify frontend with `netlify.toml`.

Netlify will:

- publish the `frontend/` folder
- use the dashboard's built-in demo fallback data until you connect the Render API

Deploy settings:

- Build command: `echo 'Static frontend deploy'`
- Publish directory: `frontend`

Recommended deploy path: connect the GitHub repo to Netlify. Netlify only needs to host the static dashboard.

If you use Netlify drag-and-drop instead, run this first and then upload the `frontend/` folder:

```powershell
C:\Users\itsab\anaconda3\python.exe -m crashrisk.demo --processed-dir data/processed --outputs-dir frontend/outputs
```

This static frontend does not create a live Python prediction API. Use Render for the live raw-file scoring backend.

## Do We Need Render?

For the static dashboard, no. Netlify can host the dashboard. For live raw-file scoring, yes.

For a real product where a user uploads raw Bloomberg-style files and receives a fresh model score on demand, yes, use a Python backend host such as Render, Railway, Fly.io, AWS, or similar. Netlify would host the frontend, and the frontend would call that backend API.

This repo now includes a Render-style FastAPI service:

- App path: `crashrisk.api.main:app`
- Health check: `/health`
- Input schema: `/schema`
- Live scoring endpoint: `POST /predict`
- Render config: `render.yaml`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn crashrisk.api.main:app --host 0.0.0.0 --port $PORT`

`POST /predict` expects a multipart form with these file fields:

- `prices`
- `benchmark_prices`
- `fundamentals`
- `controversies`

Once the Render service is deployed, paste its base URL into the dashboard's API URL field, upload the four raw files, and run a live score. The frontend will replace the static demo outputs with the API response.

Important input limitation:

- Price history alone is not enough for the full ESG crash-risk prediction.
- The full model needs `prices`, `benchmark_prices`, `fundamentals`, and `controversies`.
- If you only provide price history, the app can show a historical price chart and a volatility-style scenario range, but it cannot make the full ESG controversy crash-risk score credibly.
- The current browser upload controls load already-produced dashboard CSVs, not raw Bloomberg input files. Raw file ingestion and model training currently happen in the Python backend.

## Test

```powershell
C:\Users\itsab\anaconda3\python.exe -m pytest
```
