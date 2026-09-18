# ── RiskRadar ─────────────────────────────────────────────────────────
# Multi-stage build: dependencies are cached in their own layer so a code
# change rebuilds in seconds rather than re-downloading LightGBM every time.

FROM python:3.11-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ── runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --shell /bin/bash riskradar

COPY --from=deps /install /usr/local

WORKDIR /app
COPY --chown=riskradar:riskradar src/    ./src/
COPY --chown=riskradar:riskradar frontend/ ./frontend/
COPY --chown=riskradar:riskradar data/   ./data/
COPY --chown=riskradar:riskradar models/ ./models/
COPY --chown=riskradar:riskradar reports/ ./reports/
COPY --chown=riskradar:riskradar run.py requirements.txt ./

USER riskradar

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# If no model artefact was copied in, train one on first boot.
CMD ["sh", "-c", "[ -f models/riskradar_model.joblib ] || python -m riskradar.train; \
     exec uvicorn riskradar.api:app --host 0.0.0.0 --port 8000"]
