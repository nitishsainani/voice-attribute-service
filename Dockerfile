# ─────────────────────────────────────────────────────────────
# Dockerfile for voice-attribute-service
#
# Multi-stage build:
#   1. Base stage installs Python dependencies
#   2. Runtime stage copies only what's needed
#
# Build:
#   docker build -t voice-attribute-service .
#
# Run:
#   docker run -p 8000:8000 voice-attribute-service
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Dependencies ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime-only system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ app/
COPY models/ models/
COPY scripts/ scripts/

# Create data directories
RUN mkdir -p data/raw data/processed data/embeddings pretrained_models

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
# --host 0.0.0.0  → listen on all interfaces (required in Docker)
# --port 8000     → default API port
# --workers 1     → single worker (model is loaded in memory)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
