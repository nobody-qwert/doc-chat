# uv-based multi-stage build with CUDA Torch and MinerU

FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS builder
WORKDIR /app

# System libraries for PDF/OCR and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils ghostscript tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled PyTorch (12.4); keep outside uv to control wheel source
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# Project deps: install runtime dependencies into system site-packages
RUN uv pip install --system \
    fastapi\>=0.110 \
    uvicorn[standard]\>=0.29 \
    mineru\>=2.5.4 \
    python-multipart\>=0.0.9 \
    doclayout-yolo\>=0.0.4 \
    ultralytics\>=8.0.0 \
    transformers\>=4.38.0 \
    accelerate\>=0.20.0 \
    datasets\>=2.0.0 \
    pillow\>=9.0.0 \
    opencv-python\>=4.5.0 \
    ftfy\>=6.1.1 \
    shapely\>=2.0.0 \
    pyclipper\>=1.3.0 \
    omegaconf\>=2.3.0

# Runtime image
FROM python:3.10-slim-bookworm
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DATA_ROOT=/app_data \
    DATA_DIR=/app_data/docs \
    INDEX_DIR=/app_data/runtime \
    OCR_OUTPUT_DIR=/app_data/runtime/ocr_outputs \
    OCR_INCOMING_DIR=/app_data/runtime/incoming \
    OCR_WARMUP_DIR=/app_data/runtime/warmup \
    MINERU_DEVICE_MODE=auto \
    HF_HOME=/app_data/runtime/caches/hf \
    TRANSFORMERS_CACHE=/app_data/runtime/caches/hf \
    ULTRALYTICS_HOME=/app_data/runtime/caches/ultralytics

# System libraries for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils ghostscript tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

# Copy site-packages and scripts from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# App code
COPY . /app

# Optional: prefetch model weights at build time (set --build-arg PREFETCH_MODELS=1)
ARG PREFETCH_MODELS=0
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
      python -c "from mineru_wrapper import warmup_mineru; print('Prefetching MinerU models at build time...'); print(warmup_mineru(parse_method='auto', lang='en', table_enable=False, formula_enable=False, device_mode='cpu'))"; \
    fi

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
