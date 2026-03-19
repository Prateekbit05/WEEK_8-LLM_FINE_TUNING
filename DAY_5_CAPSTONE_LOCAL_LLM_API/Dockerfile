# ============================================================
#  DAY 5 CAPSTONE — Dockerfile
# ============================================================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/ ./deploy/
COPY cli_chat.py .
COPY streamlit_ui.py .

RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache

RUN mkdir -p /app/model_cache

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "deploy.app"]