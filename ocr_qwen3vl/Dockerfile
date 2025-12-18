# CUDA-enabled Qwen3-VL OCR module.
#
# Notes:
# - Builds `llama-server` (llama.cpp) and runs it as a managed subprocess.
# - Uses PyMuPDF to render PDFs to images, then calls the local OpenAI-compatible endpoint.
#
# Building requires network access to fetch python wheels (and llama-cpp-python).

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG LLAMA_CPP_PYTHON_REF=main
ARG LLAMA_CPP_PYTHON_CACHE_BUST=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on" \
    FORCE_CMAKE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    build-essential cmake ninja-build libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
  && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  && echo "Building llama.cpp from source..." \
  && git clone https://github.com/ggml-org/llama.cpp.git \
  && cd llama.cpp \
  && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  && cmake --build build --config Release -j$(nproc) --target llama-server \
  && cp build/bin/llama-server /usr/local/bin/ \
  && find build -name "*.so*" -exec cp -P {} /usr/local/lib/ \; \
  && ldconfig \
  && cd .. && rm -rf llama.cpp \
  && echo "LLAMA_CPP_PYTHON_CACHE_BUST=${LLAMA_CPP_PYTHON_CACHE_BUST}" >/dev/null \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --no-cache-dir \
      "uvicorn>=0.30.0" \
      "fastapi>=0.110" \
      "python-multipart>=0.0.9" \
      "httpx>=0.27" \
      "pymupdf>=1.24" \
      "pillow>=10.0" \
  && rm /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

WORKDIR /app
COPY . /app

ENV APP_DATA_ROOT=/app_data \
    DATA_DIR=/app_data/docs \
    INDEX_DIR=/app_data/runtime \
    OCR_OUTPUT_DIR=/app_data/runtime/ocr_outputs \
    OCR_INCOMING_DIR=/app_data/runtime/incoming \
    OCR_WARMUP_DIR=/app_data/runtime/warmup \
    PORT=8000 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
