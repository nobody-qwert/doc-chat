FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG LLAMA_CPP_PYTHON_REF=main

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on" \
    FORCE_CMAKE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    build-essential cmake ninja-build \
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
  && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --no-cache-dir \
      "llama-cpp-python[cuda] @ git+https://github.com/abetlen/llama-cpp-python.git@${LLAMA_CPP_PYTHON_REF}" \
      "uvicorn>=0.30.0" \
      "anyio>=4.0.0" \
      "fastapi>=0.110" \
      "starlette>=0.37" \
      "starlette-context>=0.3.6" \
      "pydantic-settings>=2.0.0" \
      "sse-starlette>=2.0.0" \
  && rm /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

WORKDIR /app

COPY docker/llm_controller.py /app/llm_controller.py

ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python3", "/app/llm_controller.py"]
