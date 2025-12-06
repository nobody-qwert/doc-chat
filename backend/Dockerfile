# Lightweight backend image (no OCR/GPU runtimes required)

FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS builder
WORKDIR /app

# System libraries for PyMuPDF rendering path
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies directly into the system site-packages
RUN uv pip install --system \
    fastapi\>=0.110 \
    uvicorn[standard]\>=0.29 \
    aiosqlite\>=0.19 \
    PyMuPDF\>=1.24.0 \
    numpy\>=1.24 \
    openai\>=1.40.0 \
    python-multipart\>=0.0.9 \
    httpx\>=0.27.0 \
    jinja2\>=3.1 \
    transformers\>=4.40

# Runtime image
FROM python:3.10-slim-bookworm
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DATA_ROOT=/app_data \
    DATA_DIR=/app_data/docs \
    INDEX_DIR=/app_data/runtime \
    DOC_STORE_PATH=/app_data/rag_meta.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Copy site-packages from the builder image
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy application code
COPY . /app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
