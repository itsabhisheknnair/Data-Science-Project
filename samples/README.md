# Sample Upload Files

These files show the schema expected by each upload control in the dashboard.

## Static Dashboard Uploads

- `stock_scores.csv` goes into **Load stock_scores.csv**.
- `price_history.csv` and `price_scenarios.csv` both go into **Load price files**.

## Live Render API Uploads

- `prices.csv` goes into **prices**.
- `benchmark_prices.csv` goes into **benchmark_prices**.
- `fundamentals.csv` goes into **fundamentals**.
- `controversies.csv` goes into **controversies**.

The raw Bloomberg-style examples are schema examples. For a real live score, use a fuller history window with enough dates for feature engineering, training, and the 13-week target horizon.
