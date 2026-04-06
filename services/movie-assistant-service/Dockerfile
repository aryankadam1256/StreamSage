# =============================================================================
# Movie Assistant Service - Dockerfile
# =============================================================================
# Fine-tuned Llama 3 8B movie discovery service with RAG pipeline
#
# In Docker, INFERENCE_BACKEND=api uses HuggingFace Inference API by default.
# For local inference with the trained model, mount the fine_tuning/outputs
# volume and set INFERENCE_BACKEND=local (requires GPU passthrough).
# =============================================================================

FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Use BuildKit caching to dramatically speed up Python ML library installations!
# This prevents re-downloading PyTorch (2GB+) if requirements.txt changes slightly.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# =============================================================================
# Production Stage
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories for data and model artifacts
RUN mkdir -p /app/data/chromadb /app/fine_tuning/outputs

# Expose port 8004
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health')" || exit 1

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
