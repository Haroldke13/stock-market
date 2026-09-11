# syntax=docker/dockerfile:1
###############################################################################
# stock-market — Flask + yfinance + ARIMA + Plotly
#
# Build:  docker build -t stock-market:latest .
# Run:    see DEPLOY.md (the app must be run read-only with a /data volume)
###############################################################################

# ---------- build stage: resolve wheels into a self-contained venv ----------
FROM python:3.11.9-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

# numpy / pandas / scipy / statsmodels / scikit-learn all ship manylinux wheels
# for CPython 3.11, so no compiler is needed. Keep it that way: if you bump the
# base image to a Python version without wheels this stage will start building
# from source and the image will balloon.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.11.9-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH" \
    PORT=5757

COPY --from=builder /opt/venv /opt/venv

RUN useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Application code lives in /app and is NEVER written to at runtime.
WORKDIR /app
COPY --chown=root:root . /app

# Two writable spots, because the code writes to both:
#
#   /app/downloads  DOWNLOAD_FOLDER / UPLOAD_FOLDER, resolved from __file__ and
#                   created at import time by app.py. Must exist and be owned by
#                   the runtime user or the module fails to import.
#   /data           the process working directory. /yfinance still builds its
#                   CSV path with os.path.join(os.getcwd(), ...), so cwd has to
#                   be writable too. Making cwd a volume keeps those files out
#                   of the code tree.
#
# Everything else under /app stays root-owned and read-only to the app user, so
# the container can be run with --read-only and still work.
RUN mkdir -p /app/downloads /data \
 && chown appuser:appuser /app/downloads /data
VOLUME ["/app/downloads", "/data"]

USER appuser
WORKDIR /data

EXPOSE 5757

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5757/healthz', timeout=4).status == 200 else 1)"

# ARIMA fits are slow and CPU-bound; 2 workers keeps peak RSS near ~600 MB and
# --timeout 180 stops gunicorn killing a legitimate long fit.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5757", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "180", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "deploy_wsgi:app"]
