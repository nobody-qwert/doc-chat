# Environment Configuration Cleanup

## Goals
- Fail fast whenever a required configuration value is missing instead of silently using defaults.
- Align every service (backend, OCR module, diagnostics, LLM controller, embedding client) on the same stricter behaviour.
- Ensure the `.env` file documents every parameter that must be provided during local development.

## Work Items
1. **Inventory & Tooling**
   - Add a small script (or documented `rg` invocations) that enumerates every `os.environ.get`/helper call so new defaults cannot creep in.
   - Use this during reviews to keep the required-variable list in sync with `.env`.
2. **Backend settings loader (`backend/config.py`)**
   - Update `_int_env`, `_float_env`, `_str_env` to raise descriptive `RuntimeError`s when the variable is unset; remove their default arguments.
   - Remove all hard-coded fallbacks inside `load_settings()` (paths, LLM endpoints, similarity thresholds, etc.) and require `LLM_CONTROL_URL`, `LLM_ENDPOINT`, `DATA_DIR`, … to be passed explicitly.
   - Decide which knobs (if any) remain optional. If something is intentionally optional, keep a `_optional_*` helper but make the call sites spell out why.
   - Update `frontend_origin` and other derived fields so they read from required env vars instead of embedding defaults.
3. **Backend modules that bypass `AppSettings`**
   - `backend/app.py` should read `BACKEND_PORT` via the stricter helpers or `settings` and fail if unset.
   - `backend/embeddings.py` currently pulls `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE` directly with defaults—switch these to the same helper logic (or wire them through `AppSettings`).
   - `backend/tokenizer_registry.py` should require `TRANSFORMERS_SPEC` instead of silently defaulting to `transformers>=4.40`.
4. **OCR service (`ocr_mineru/`)**
   - Replace the direct `os.environ.get` blocks in `ocr_mineru/app.py` and `ocr_mineru/mineru_wrapper.py` with helper functions that raise when storage paths, MinerU settings, or warmup flags are missing.
   - Make `_env_bool` raise on malformed/missing values instead of returning `default`.
   - Ensure the FastAPI app refuses to boot if `PORT` is absent.
5. **Diagnostics service (`diagnostics/app.py`)**
   - Require `PORT` at startup so the container cannot silently fall back to 9001.
6. **LLM controller (`llama-cpp/llm_controller.py`)**
   - Introduce a simple `require_env("NAME")` helper that throws when `LLM_SERVER_CMD`, `LLM_CONTROL_PORT`, `LLM_SERVER_SHUTDOWN_TIMEOUT`, or `LOG_LEVEL` are missing.
   - Document that these must be set by `docker-compose` (and add them to `.env` for discoverability).
7. **Verification**
   - After removing defaults, run `rg "os\.environ\.get\([^,]+,"` to ensure no `get(..., default)` calls remain.
   - Boot each service without one required var to confirm it now fails fast with a helpful error message.
   - Update `.env.example` alongside `.env` so CI or other developers know which variables to supply.

## Referenced vars missing from `.env`
Variables currently read by the code but absent from `.env`. Some are provided indirectly by `docker-compose`, but documenting them keeps local setups aligned.

| Variable | Referenced at | Notes |
| --- | --- | --- |
| `AGENTIC_MAX_SUBQUERIES` | `backend/config.py:118` | Controls max agentic subqueries; now defaulting to 5.
| `CHAT_COMPLETION_MAX_TOKENS` | `backend/config.py:86` | Completion cap; defaults to 2048 today.
| `CHAT_COMPLETION_RESERVE` | `backend/config.py:87` | Reserved context tokens.
| `DIAGNOSTICS_URL` | `backend/config.py:115`, `backend/utils/gpu.py:21` | Needed so GPU diagnostics work.
| `EMBEDDING_BATCH_SIZE` | `backend/embeddings.py:14`, `backend/routes/system.py:63` | Currently falls back to 1.
| `GPU_PHASE_TIMEOUT` | `backend/config.py:160` | Governs GPU warmup state machine.
| `LLM_CONTROL_PORT` | `llama-cpp/llm_controller.py:67` | Passed into the controller container only via compose.
| `LLM_CONTROL_URL` | `backend/config.py:109` | Used to drive the llama.cpp control plane.
| `LLM_READY_TIMEOUT` | `backend/config.py:161` | Wait duration before backend declares LLM unavailable.
| `LLM_SERVER_CMD` | `llama-cpp/llm_controller.py:65` | Full llama.cpp launch command.
| `LLM_SERVER_SHUTDOWN_TIMEOUT` | `llama-cpp/llm_controller.py:69` | Graceful shutdown timer for controller.
| `LOG_LEVEL` | `llama-cpp/llm_controller.py:72` | Controls logging verbosity for controller.
| `OCR_CONTROL_URL` | `backend/config.py:113` | Backend uses it to call the OCR control plane.
| `PORT` | `diagnostics/app.py:134`, `ocr_mineru/app.py:397` | Required by the diagnostics and OCR services when run outside docker.
| `TRANSFORMERS_SPEC` | `backend/tokenizer_registry.py:33` | Determines which transformers build to auto-install.

Update `.env` (and `.env.example`) with the above values before removing the defaults so developers know what to supply.
