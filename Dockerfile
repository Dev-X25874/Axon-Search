# ── Build stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for lxml, playwright, faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -e ".[dev]" --no-build-isolation

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY src/ ./src/

# Index and model cache dirs
RUN mkdir -p /app/data/index

# Non-root user
RUN useradd -m -u 1001 axon
RUN chown -R axon:axon /app
USER axon

# Pre-download models at build time (optional — comment out to download at runtime)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV INDEX_DIR=/app/data/index
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.server:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--app-dir", "src"]
