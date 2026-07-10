# Doc Chat

Doc Chat is a local, GPU-accelerated document ingestion and retrieval-augmented generation (RAG) application. It extracts text from PDFs, creates a searchable local index, and answers questions with citations to the source document chunks.

The default configuration uses one Qwen3.6 model process for both text generation and vision OCR. The OCR adapter renders PDF pages and sends images to the already-running multimodal LLM, so Qwen OCR does not load a second copy of the model or compile a separate llama.cpp server.

## Features

- Local PDF upload, OCR, chunking, embedding, and persistence
- Streaming, agentic RAG chat with source citations
- Original-document viewer and extracted-text preview
- Qwen3.6 vision OCR through the shared LLM endpoint
- Optional MinerU OCR provider
- Local Qwen3 embedding model through llama.cpp
- GPU diagnostics and ingestion progress reporting
- Docker Compose deployment with persistent local application data

## Architecture

```text
Browser
  |
  v
Frontend (React + nginx, :5173)
  |
  v
Backend (FastAPI, :8000)
  |-- chat -----------------------> llm-inference
  |                                 Qwen3.6 GGUF + mmproj (:8010)
  |
  |-- Qwen OCR adapter ----------> same llm-inference process
  |   (PDF rendering only)
  |
  |-- embeddings ----------------> embed
  |                                 Qwen3 Embedding GGUF (:8011)
  |
  |-- state/documents -----------> app_data/
  |
  `-- GPU status ----------------> diagnostics (:9001)
```

When the `mineru` profile is selected, MinerU replaces the Qwen OCR adapter. Do not enable both OCR profiles simultaneously because both use the `ocr-module` network alias.

## Requirements

- Linux with an NVIDIA GPU and current NVIDIA driver
- NVIDIA Container Toolkit configured for Docker
- Docker Engine with Docker Compose v2
- Enough disk space for the model files and Docker images
- Internet access for the initial image builds

The current Qwen3.6 Q4 model and F32 projector use approximately 22 GB on disk. This configuration has been tested with a 24 GB RTX 3090. Other model sizes or context settings may require different GPU memory.

## Model files

Model weights are intentionally excluded from Git. Place them under `models/` using this layout:

```text
models/
├── llm/
│   ├── Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
│   └── mmproj-F32.gguf
└── embed/
    └── Qwen3-Embedding-0.6B-Q8_0.gguf
```

If you use different filenames, update these values in `.env`:

```dotenv
LLM_MODEL_FILE=Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
LLM_MMPROJ_FILE=mmproj-F32.gguf
EMBED_MODEL_FILE=Qwen3-Embedding-0.6B-Q8_0.gguf
```

The projector must match the Qwen model. The main server loads it through llama.cpp's MTMD multimodal handler, enabling the same endpoint to process text and images.

## Quick start

1. Create your local configuration:

   ```bash
   cp .env.example .env
   ```

2. Put the model files in the directories shown above.

3. Build the services:

   ```bash
   docker compose --profile qwen3 build
   ```

   The first `llm-inference` build compiles CUDA-enabled llama-cpp-python and can take several minutes. The Qwen OCR adapter itself is a small Python image and does not compile llama.cpp.

4. Start the application:

   ```bash
   docker compose --profile qwen3 up -d
   ```

5. Open the web interface:

   ```text
   http://localhost:5173
   ```

The default `.env.example` also sets `COMPOSE_PROFILES=qwen3`, so ordinary `docker compose up -d` uses the Qwen OCR provider after the configuration is copied.

## Service endpoints

| Service | Host address | Purpose |
| --- | --- | --- |
| Frontend | `http://localhost:5173` | Web application and `/api` proxy |
| Backend | `http://localhost:8000` | FastAPI application API |
| LLM | `http://localhost:8010/v1` | OpenAI-compatible Qwen3.6 text/vision API |
| Embeddings | `http://localhost:8011/v1` | OpenAI-compatible embedding API |
| Diagnostics | `http://localhost:9001` | GPU diagnostics service |

Check that the stack is running:

```bash
docker compose ps
curl http://localhost:8000/healthz
curl http://localhost:8010/v1/models
```

## OCR providers

### Shared Qwen3.6 OCR (default)

Use these root `.env` values:

```dotenv
COMPOSE_PROFILES=qwen3
OCR_PARSER_KEY=qwen3_vl
```

Provider-specific prompting and rendering settings are in `ocr_qwen3vl/qwen3.env`. The adapter has no GPU reservation and no model mount; it renders each PDF page and forwards it to `llm-inference`.

### MinerU OCR

Change the root `.env` values to:

```dotenv
COMPOSE_PROFILES=mineru
OCR_PARSER_KEY=mineru
```

Then rebuild and recreate the relevant services:

```bash
docker compose --profile mineru up -d --build
```

MinerU settings live in `ocr_mineru/mineru.env`. MinerU has its own GPU runtime and model caches under `app_data/runtime/caches/`.

## Configuration

The root `.env` is the main configuration file. Important settings include:

| Variable | Description |
| --- | --- |
| `COMPOSE_PROFILES` | Active OCR Compose profile: `qwen3` or `mineru` |
| `LLM_MODEL` | Model name sent to the OpenAI-compatible endpoint |
| `LLM_MODEL_FILE` | Main GGUF filename under `models/llm/` |
| `LLM_MMPROJ_FILE` | Vision projector filename under `models/llm/` |
| `LLM_CONTEXT_SIZE` | llama.cpp context window allocated by the server |
| `LLM_BATCH_SIZE` | llama.cpp prompt batch size |
| `LLM_TOKENIZER_ID` | Hugging Face tokenizer used by backend token accounting |
| `EMBED_MODEL_DIR` | Host directory containing embedding weights |
| `EMBED_MODEL_FILE` | Embedding GGUF filename |
| `OCR_PARSER_KEY` | Backend OCR provider identifier |
| `HOST_APP_DATA` | Persistent host directory for documents and runtime state |
| `CHAT_COMPLETION_MAX_TOKENS` | Maximum generated tokens per chat completion |
| `MIN_CONTEXT_SIMILARITY` | Minimum semantic retrieval similarity |

The LLM command disables Qwen's visible thinking block. This keeps normal RAG responses and OCR output concise and prevents internal reasoning from consuming the completion-token budget.

## Persistent data

Runtime state is stored under `app_data/` and is excluded from Git:

```text
app_data/
├── docs/                 # Uploaded source documents
├── rag_meta.db           # SQLite metadata and index records
└── runtime/
    ├── incoming/
    ├── ocr_outputs/
    ├── warmup/
    └── caches/           # Provider caches, when applicable
```

Back up `app_data/` if you need to preserve uploaded documents and the local index.

## Common operations

```bash
# Show container status
docker compose ps

# Follow all logs
docker compose logs -f

# Follow the LLM and Qwen OCR adapter
docker compose logs -f llm-inference ocr-qwen3

# Recreate services after editing .env or Compose configuration
docker compose --profile qwen3 up -d --force-recreate

# Stop the stack without deleting application data
docker compose down

# Rebuild one service
docker compose build backend
```

Do not add `-v` to `docker compose down` unless you intentionally want to remove Docker-managed volumes. The bind-mounted `app_data/` directory is managed separately on the host.

## Updating llama.cpp support

The LLM image builds llama-cpp-python from the configured Git reference. New model architectures may require a newer upstream revision. Set or update:

```dotenv
LLAMA_CPP_PYTHON_REF=main
LLAMA_CPP_PYTHON_CACHE_BUST=2
```

Then rebuild the LLM image:

```bash
docker compose build --pull llm-inference
docker compose up -d --force-recreate llm-inference
```

Incrementing `LLAMA_CPP_PYTHON_CACHE_BUST` forces Docker to rerun the native build even when the reference name remains `main`.

## Troubleshooting

### `unknown model architecture: qwen35moe`

The local llama.cpp build is too old. Bump `LLAMA_CPP_PYTHON_CACHE_BUST` and rebuild `llm-inference` as described above.

### Model or projector not found

Check the filenames in `.env` and verify the files exist:

```bash
ls -lh models/llm models/embed
docker compose config | grep -E 'model|clip_model_path'
```

### Docker cannot access the GPU

Verify the host driver and container runtime:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

### Qwen OCR is unavailable

Confirm that the Qwen profile is active and that the adapter can reach the shared LLM:

```bash
docker compose ps llm-inference ocr-qwen3
docker compose logs --tail=200 llm-inference ocr-qwen3
```

The expected root configuration is `COMPOSE_PROFILES=qwen3` with `OCR_PARSER_KEY=qwen3_vl`.

### Port already in use

Change the corresponding host-port value in `.env`, such as `FRONTEND_PORT`, `BACKEND_PORT`, or `LLM_HOST_PORT`, and recreate the stack.

### Out-of-memory errors

Reduce `LLM_CONTEXT_SIZE` or use a smaller quantization. Qwen OCR and chat share the same model allocation, but embeddings and MinerU have separate GPU workloads.

## Repository layout

```text
backend/          FastAPI API, ingestion pipeline, retrieval, and agentic chat
frontend/         React/Vite application served by nginx
llama-cpp/        CUDA llama-cpp-python image and managed LLM controller
ocr_qwen3vl/      Lightweight PDF renderer and shared-LLM OCR adapter
ocr_mineru/       Optional MinerU OCR provider
diagnostics/      NVIDIA GPU diagnostics service
models/           Local model weights (ignored except for .gitkeep)
app_data/         Persistent documents and runtime state (ignored)
docs/             Design and operational notes
```

## Security and privacy

Model inference and document storage remain local to the Docker host with the default configuration. The `.env`, model weights, uploaded documents, database, OCR output, and provider caches are ignored by Git. Review exposed host ports before running the stack on a machine reachable from an untrusted network.
