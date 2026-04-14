# Bloomberg and Yahoo Finance Data Guide

This project needs four final pipeline files:

- `prices.csv`: `ticker`, `date`, `adj_close`, `volume`
- `benchmark_prices.csv`: `date`, `benchmark_close`
- `fundamentals.csv`: `ticker`, `period_end`, `market_cap`, `shares_outstanding`, `market_to_book`, `leverage`, `roa`
- `controversies.csv`: `ticker`, `date`, `sector`, `controversy_score`

Use `bloomberg/ticker_universe.csv` as the company list.

## Recommended Route

Use Yahoo Finance for fast price/benchmark/fundamental fallback data, then use Bloomberg for ESG controversy history:

```powershell
C:\Users\itsab\anaconda3\python.exe scripts\fetch_yfinance_data.py --output-dir data/raw_yfinance --start 2019-01-01 --end 2024-12-31
```

This writes:

- `data/raw_yfinance/prices.csv`
- `data/raw_yfinance/benchmark_prices.csv`
- `data/raw_yfinance/fundamentals.csv`
- `data/raw_yfinance/sector_map.csv`

Yahoo fundamentals are a latest snapshot, not a proper point-in-time quarterly history. The helper stamps them before the sample start date so the demo pipeline can run, but use Bloomberg historical fundamentals for the final research results. Then use Bloomberg to pull `controversies.csv`, because Yahoo Finance does not provide MSCI/Sustainalytics/Refinitiv controversy score history.

For a plumbing-only test, you can run:

```powershell
C:\Users\itsab\anaconda3\python.exe scripts\fetch_yfinance_data.py --output-dir data/raw_yfinance --placeholder-controversies
```

Do not use placeholder controversies for the final ESG research claim.

## Bloomberg Excel Formulas

Assume the `tickers` sheet has:

| Column | Meaning |
|---|---|
| `A` | Bloomberg ticker, e.g. `AAPL US Equity` |
| `B` | Clean ticker, e.g. `AAPL` |

### Prices

Use BDH:

```excel
=BDH(A2,"PX_LAST,PX_VOLUME","2019-01-01","2024-12-31","periodicitySelection=DAILY","adjustmentSplit=TRUE","adjustmentAbnormal=TRUE")
```

Export and reshape to:

```text
ticker,date,adj_close,volume
```

### Benchmark

Use S&P 500 if your universe is US large-cap:

```excel
=BDH("SPX Index","PX_LAST","2019-01-01","2024-12-31","periodicitySelection=DAILY")
```

Export as:

```text
date,benchmark_close
```

### Fundamentals

Try quarterly BDH first:

```excel
=BDH(A2,"CUR_MKT_CAP,BS_SH_OUT,PX_TO_BOOK_RATIO,TOT_DEBT_TO_TOT_EQY,RETURN_ON_ASSET","2019-01-01","2024-12-31","periodicitySelection=QUARTERLY")
```

If the output is messy, pull one field per tab and reshape later:

- `CUR_MKT_CAP`
- `BS_SH_OUT`
- `PX_TO_BOOK_RATIO`
- `TOT_DEBT_TO_TOT_EQY`
- `RETURN_ON_ASSET`

Export as:

```text
ticker,period_end,market_cap,shares_outstanding,market_to_book,leverage,roa
```

### ESG Controversy History

Use one provider consistently. Preferred:

```excel
=BDH(A2,"MSCI_ESG_CTRVRSY_SCORE","2019-01-01","2024-12-31","periodicitySelection=MONTHLY")
```

Alternatives if MSCI is unavailable:

- `SUSTAINALYTICS_CONTROVERSY_SCORE`
- `REFINITIV_ESG_CONTROVERSY_SCORE`

Pull sector separately:

```excel
=BDS(A2,"GICS_SECTOR_NAME")
```

Export as:

```text
ticker,date,sector,controversy_score
```

## Optional Textual Analysis

If Bloomberg News export is available, create either `news_text.csv` or `controversy_text.csv`:

```text
ticker,date,source,headline,description
```

The backend will create `outputs/textual_analysis.csv` and `outputs/figures/text_word_cloud.svg` when this file exists.

## Final Step

Once the final four files are ready, place them in `data/raw/` and run:

```powershell
C:\Users\itsab\anaconda3\python.exe -m crashrisk.cli --raw-dir data/raw
```

Or score live via the Render API from the frontend.
