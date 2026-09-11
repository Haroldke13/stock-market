# stock-market

A single-file Flask app that downloads historical price data from Yahoo Finance, fits an ARIMA model
to the closing prices, and renders the 30-day forecast on a Plotly candlestick chart.

Live demo (Render, may be asleep or offline): <https://stock-market-1ske.onrender.com>

## What it does

Three working routes in `app.py`:

- **`/yfinance`** — enter a ticker symbol. The app calls `yf.download(symbol, start='2010-01-01', end=today)`,
  saves the result as `<SYMBOL>_stock_data.csv` in the working directory, and renders the raw table.
- **`/stock_market_prediction`** — either enter a ticker or upload a CSV containing `Date` and
  `Close` columns. The app:
  1. cleans the data and reindexes it to business days (`asfreq('B')`),
  2. holds out the last 30 rows, fits `ARIMA(order=(5, 1, 0))` on the rest,
  3. forecasts 30 steps ahead,
  4. computes 21/50/100-day EMAs over the forecast and labels the trend **Bullish** if the last
     predicted value is above the 21-day EMA, otherwise **Bearish**,
  5. renders a Plotly candlestick chart with the forecast and EMAs overlaid, plus an actual-vs-predicted table.
- **`/analyze_data`** — the same ARIMA + EMA pipeline applied to a CSV already saved on disk.
- **`/download_csv/<filename>`, `/download_csv1/<filename>`** — download a generated CSV.

Fewer than 50 usable rows produces a "Not enough data" message instead of a forecast.

## Tech stack

Python 3, Flask 3.1, `yfinance`, `pandas`, `numpy`, `statsmodels` (ARIMA), `scikit-learn`,
Plotly (charts, rendered client-side from JSON), Jinja2 templates, Gunicorn.

## Setup

```bash
git clone https://github.com/Haroldke13/stock-market.git
cd stock-market
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

**The app listens on port 5757**, not the Flask default — open <http://127.0.0.1:5757>.

No database, no API key and no environment variables are required; `yfinance` scrapes Yahoo Finance
directly. Downloaded CSVs are written to the process's current working directory.

## Known issues

- **Path traversal in `/download_csv/<filename>`.** The filename is joined onto the working directory
  and passed to `send_file` without sanitisation, so a crafted path can read arbitrary files the
  process can access. (`/download_csv1` uses `send_from_directory`, which is safe.) Fix before
  exposing this publicly.
- `app.secret_key` is hardcoded in `app.py`. Move it to an environment variable.
- In `/stock_market_prediction`, the ticker-symbol branch still dereferences `uploaded_file.filename`
  at the end of the handler, which raises `AttributeError` on `None`. The CSV-upload branch works.
- Generated CSVs accumulate in the working directory and are listed back to every user of the app.
- ARIMA(5,1,0) is a fixed, unvalidated order — there is no model selection, no differencing check and
  no accuracy metric shown, so the "prediction" should be treated as a demonstration, not a signal.

## Status

**Working prototype.** Last commit January 2025. The Yahoo Finance download, the ARIMA forecast and
the Plotly charting all function; the issues above are unaddressed. No tests. No `.gitignore`.

Not financial advice.

## Licence

MIT (see `LICENSE`).
