# Deploying `stock-market`

Flask app that downloads OHLCV data with **yfinance**, fits an **ARIMA(5,1,0)**
model with statsmodels, and renders **Plotly** candlestick charts.

| Fact | Value |
|---|---|
| WSGI entrypoint | `deploy_wsgi:app` (wraps `app:app`, adds `/healthz`) |
| Listens on | `5757` inside the container |
| Suggested hostname | `stocks.harolditdata.uk` |
| Persistent state | CSV scratch files only, in `/app/downloads` and `/data` |
| Outbound network | **required** — `query*.finance.yahoo.com` (yfinance) |

---

## 1. Environment variables

Derived from the code, not guessed. There are only two, and neither is a
credential.

| Variable | Required | Default | Read by | Notes |
|---|---|---|---|---|
| `SECRET_KEY` | **Yes, in production** | `<set SECRET_KEY in .env - value intentionally not documented>` (hardcoded in `app.py`) | `deploy_wsgi.py` | Flask session signing key. `app.py` sets a literal; `deploy_wsgi.py` overrides it when this variable is set, so you never have to edit the source. Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `PORT` | No | `5757` | `Procfile` only | The Dockerfile binds 5757 unconditionally. Only Heroku/Render-style platforms need this. |
| `MPLBACKEND` | No | `Agg` (set in the image) | matplotlib | `app.py` imports `matplotlib.pyplot` at module scope; `Agg` keeps it headless. |

The app needs **no API key** — yfinance scrapes Yahoo's public endpoints.
There is no database.

Never put `SECRET_KEY` in the image or in `docker run --env` on a shared box;
use `--env-file` with a root-owned `0600` file, or your orchestrator's secret
store.

## 2. Build and run

```bash
docker build -t stock-market:latest .

docker volume create stock_market_downloads
docker volume create stock_market_cwd

docker run -d --name stock-market \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v stock_market_downloads:/app/downloads \
  -v stock_market_cwd:/data \
  -p 127.0.0.1:5757:5757 \
  --env-file /etc/harold/stock-market.env \
  --memory 900m --cpus 1.5 \
  stock-market:latest
```

### The two writable paths, and why they are volumes

`app.py` writes CSVs to two different places:

| Path | Set by | Written by |
|---|---|---|
| `/app/downloads` | `DOWNLOAD_FOLDER = Path(__file__).resolve().parent / "downloads"`, created with `.mkdir()` **at import time** | uploads and `/stock_market_prediction` |
| `/data` (the working directory) | `os.path.join(os.getcwd(), csv_filename)` in `/yfinance` | symbol lookups |

Both are volumes owned by uid `10001`; everything else under `/app` is
root-owned and unwritable by the app user. That is what makes `--read-only`
workable — and `--read-only` is what stops a path-traversal write from reaching
the code tree.

The `DOWNLOAD_FOLDER.mkdir()` call runs at import, so if `/app/downloads` is not
writable the module fails to import and gunicorn never starts. The Dockerfile
creates and chowns it for exactly that reason.

### One remaining sharp edge in the upload path

A concurrent security pass added `resolve_in_download_folder()` and routed the
*read* paths (`/download_csv`, `/download_csv1`, `analyze_data`) through it —
good. The **write** in `/stock_market_prediction` is still

```python
csv_filename = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
uploaded_file.save(csv_filename)
```

`os.path.join(sandbox, "../../x")` still escapes the sandbox, so this is an
arbitrary write as the app user until the filename is sanitised:

```python
from werkzeug.utils import secure_filename
csv_filename = os.path.join(UPLOAD_FOLDER, secure_filename(uploaded_file.filename))
```

The same applies to `f"{stock_symbol}_data.csv"` and to the `os.getcwd()` write
in `/yfinance`, both of which interpolate raw user input into a path. Run
`--read-only` until those land; with it, the blast radius is the two volumes.

## 3. Health check

`GET /healthz` → `{"status": "ok"}`. Added by `deploy_wsgi.py`; `app.py` was not
modified. Docker's `HEALTHCHECK` polls it every 30 s.

## 4. Behaviour worth knowing before you expose it

- **CPU.** Every prediction request fits an ARIMA model over ~15 years of daily
  bars. That is seconds of pegged CPU per request, single-threaded. Gunicorn
  runs 2 workers with `--timeout 180`. An unauthenticated visitor can trivially
  keep both workers busy — put it behind Cloudflare Access or a Cloudflare rate
  limit if that matters to you.
- **Disk.** Every symbol lookup and every upload writes a CSV and nothing ever
  deletes them. Expect slow unbounded growth across both volumes; a weekly
  `find /var/lib/docker/volumes/stock_market_*/_data -name '*.csv' -mtime +7
  -delete` on the host is enough.
- **`/stock_market_prediction` lists the whole download folder.** The dropdown of
  "available CSV files" shows files uploaded by *any* visitor, and any visitor
  can then download them. There is no per-user isolation — treat everything
  uploaded here as public.
- `analyze_data()` references `plotly.utils` before `import plotly` executes at
  module bottom — it works only because the import runs at import time. Don't
  reorder it.

## 5. Cloudflare tunnel

The ingress rule lives in the shared tunnel config, not here. See
`/home/onyango/Projects/DEPLOYMENT_PLAN.md`. The service line is:

```yaml
  - hostname: stocks.harolditdata.uk
    service: http://localhost:5757
```

## 6. Files added by the deployment pass

`Dockerfile`, `.dockerignore`, `Procfile`, `deploy_wsgi.py`, `DEPLOY.md`.
No existing file was modified. `requirements.txt` was already accurate and is
used as-is.
