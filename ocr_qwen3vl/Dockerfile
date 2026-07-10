# Lightweight Qwen vision OCR adapter. Qwen3.6 inference is provided by the
# shared llm-inference service; this image only renders documents and calls it.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3 -m pip install --no-cache-dir \
      "uvicorn>=0.30.0" \
      "fastapi>=0.110" \
      "python-multipart>=0.0.9" \
      "httpx>=0.27" \
      "pymupdf>=1.24" \
      "pillow>=10.0"

WORKDIR /app
COPY . /app

ENV APP_DATA_ROOT=/app_data \
    DATA_DIR=/app_data/docs \
    INDEX_DIR=/app_data/runtime \
    OCR_OUTPUT_DIR=/app_data/runtime/ocr_outputs \
    OCR_INCOMING_DIR=/app_data/runtime/incoming \
    OCR_WARMUP_DIR=/app_data/runtime/warmup \
    PORT=8000

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
