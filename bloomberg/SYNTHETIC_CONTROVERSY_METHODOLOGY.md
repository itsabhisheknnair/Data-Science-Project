# Synthetic ESG Controversy Methodology

Due to temporary lack of Bloomberg access, the prototype uses structured synthetic ESG controversy scores for methodology testing. Real Yahoo Finance prices and benchmark data are retained, while controversy scores are simulated solely to validate the feature pipeline, model workflow, and dashboard behavior.

The synthetic file is:

```text
data/raw_yfinance/controversies.csv
```

It contains:

```text
ticker,date,sector,controversy_score
```

Scores are monthly and scaled from 0 to 10. The generator is:

```text
scripts/generate_synthetic_controversies.py
```

## Simulation Design

Each firm-month score is generated from:

```text
C_i,t = 0.70 * C_i,t-1 + 0.20 * firm_base_i + 0.10 * sector_base_s + shock_i,t + epsilon_i,t
```

The process includes:

- Sector effects: Energy, Financials, Industrials, and Basic Materials have higher average controversy levels than Utilities and Consumer Defensive names.
- Firm effects: selected firms have higher or lower ticker-specific controversy tendencies.
- Persistence: elevated controversy carries forward into later months.
- Shock events: occasional spikes represent lawsuits, governance issues, labor events, regulatory scrutiny, product-safety incidents, environmental issues, or data-privacy events.
- Decay: shock impact fades over later months instead of disappearing immediately.
- Noise: small random shocks keep the monthly panel from looking mechanical.
- Weak downside alignment: shock timing is lightly tilted toward months that precede weaker realized Yahoo price outcomes over the next three months, so the prototype model has a realistic ESG-risk pattern to discover. This does not change the price data or prove a real ESG relationship.

## Report Wording

Use this wording in the report:

> Due to temporary lack of Bloomberg access, we constructed a synthetic controversy-score panel for prototype development. The synthetic series was designed to mimic realistic ESG controversy dynamics, including persistence, sector heterogeneity, and shock-driven spikes. Real market prices were retained, while controversy data was simulated solely for methodology testing and pipeline validation.

## Limitation

This synthetic data is not evidence that ESG controversy improves real crash-risk prediction. The final research claim still requires replacing `data/raw_yfinance/controversies.csv` with real Bloomberg/MSCI/Sustainalytics/Refinitiv controversy history and rerunning the baseline-vs-ESG validation.
