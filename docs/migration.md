# Plan

Rework the app into a single macOS-native desktop application that embeds the existing React UI in a WebView (no external web server) and calls the ML pipeline directly in-process (no HTTP APIs). The backend FastAPI layer becomes internal service classes, while Qwen3-VL and LLM/embeddings run via Metal-enabled llama-cpp on the M3 Ultra.

## Requirements
- Single native macOS app: no Docker, no separate backend server, no HTTP API layer.
- UI remains the current React frontend, shown via an embedded WebView using local bundled assets.
- Always use Apple Silicon acceleration (Metal) for LLM, embeddings, and Qwen3-VL OCR.
- OCR engine is Qwen3-VL only (MinerU removed/disabled).

## Scope
- In:
  - Desktop shell + WebView packaging.
  - In-process services for LLM/embeddings/OCR.
  - Frontend data layer updated to call local native bridge instead of HTTP.
  - Unified config + local model management.
- Out:
  - FastAPI HTTP server and Docker compose.
  - MinerU OCR and CUDA/NVIDIA paths.

## Files and entry points
- backend/* (logic to be refactored into in-process service classes)
- ocr_qwen3vl/* (used as local OCR pipeline)
- llama-cpp/* (Metal build + local inference)
- frontend/* (build + embedded assets)
- New: desktop shell entry point (Tauri/Electron/pywebview) and native IPC bridge layer.

## Data model / API changes
- Replace HTTP endpoints with local IPC/bridge calls (e.g., invoke/command pattern).
- Keep internal request/response shapes stable to minimize UI changes.

## Action items
[ ] Choose desktop container: Tauri (Rust + WebView) or Electron (Node + WebView); lock the choice and define build pipeline for macOS.
[ ] Build a "core" module from backend that exposes synchronous/async functions for chat, embeddings, ingestion, and OCR without FastAPI.
[ ] Replace LLM usage to call llama-cpp-python directly with Metal (CMAKE_ARGS="-DGGML_METAL=on"), and expose streaming tokens to the UI via the bridge.
[ ] Replace embeddings client with local llama-cpp embeddings (Metal), remove EMBEDDING_BASE_URL dependency.
[ ] Wire ocr_qwen3vl as a local module: render PDF -> Qwen3-VL inference -> return text/metadata; remove MinerU paths.
[ ] Update the frontend data layer to use the native bridge (IPC) rather than HTTP/SSE; preserve streaming UX.
[ ] Bundle the React build output into the app and load it via the WebView file URL (no local server).
[ ] Create a macOS-native run/build script that installs Metal deps, builds models, and packages the app.

## Testing and validation
- macOS launch test: UI loads from embedded assets, no local server running.
- Chat test: streaming output + embeddings success with Metal.
- OCR test: multi-page PDF, Qwen3-VL output, progress updates.
- Memory test: ensure LLM + OCR coexist without OOM on M3 Ultra.

## Risks and edge cases
- Qwen3-VL Metal support may require specific model formats or llama-cpp flags.
- IPC streaming performance vs SSE: may need backpressure handling.
- Large model memory pressure when OCR and LLM run concurrently.

## Open questions
- Preferred desktop shell: Tauri (smaller, Rust) or Electron (faster setup, heavier)?
- Should LLM/OCR run as in-process libraries, or as managed subprocesses inside the same app bundle (still no HTTP)?
