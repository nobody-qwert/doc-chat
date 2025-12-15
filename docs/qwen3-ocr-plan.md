# Qwen3-VL OCR Module Plan (Docker + GPU)

## Goal

Enable swapping OCR engines by replacing the OCR Docker container (MinerU today, Qwen3-VL tomorrow) while keeping the backend integration stable and provider-specific configuration isolated.

Key constraint from current setup:
- **MinerU OCR does not need GPU unload** (leave as-is).
- **Qwen3-VL OCR must support GPU unload** (to free VRAM when switching back to the LLM service).

## Current State (What Exists Today)

### Backend ↔ OCR Service Contract (must remain stable)

The backend only depends on an HTTP OCR service with the following endpoints:

- `POST /parse`
  - multipart form fields:
    - `doc_hash` (string)
    - `file` (uploaded file; typically PDF)
  - optional fields (currently supported by MinerU):
    - `parse_method`, `lang`, `table_enable`, `formula_enable`
  - response: `{ "job_id": "...", "status": "queued|running|..." }`

- `GET /jobs/{job_id}`
  - response includes `status` and (optionally) a `progress` object

- `GET /jobs/{job_id}/result`
  - response: `{ "text": "...", "metadata": { ... } }`

Supporting endpoints:
- `POST /warmup` (backend uses it today as “warmup mineru”)
- Control endpoints used by GPU phase manager:
  - `POST /control/load`
  - `GET /control/status`

Notes:
- The backend stores OCR output under `OCR_PARSER_KEY` in the extractions table, so changing providers is safe if each provider uses a distinct parser key (e.g. `mineru` vs `qwen3_vl`).
- The backend should not need to know MinerU vs Qwen-specific behavior beyond config selection and (optionally) unload behavior.

### MinerU env/config leakage

Today `.env` contains OCR module settings that are **MinerU-specific** (e.g. `MINERU_*`) and Docker Compose injects them into the OCR container.

This makes provider swaps harder because:
- “global” env becomes coupled to a specific OCR provider
- new OCR engines need their own provider-only configuration

## Target Architecture

### Stable, provider-agnostic interface

Keep the existing backend expectations unchanged:
- same HTTP endpoints (`/parse`, `/jobs/...`, `/result`, `/control/load`, `/control/status`)
- same response shapes (`{text, metadata}` for result)

Provider-specific behavior stays inside the OCR container.

### Provider-specific env isolation

Split configuration into:

**Backend-shared OCR settings (keep in root `.env`)**
- `OCR_PARSER_KEY` (selected provider key)
- `OCR_MODULE_URL` (points to the active OCR service container)
- `OCR_CONTROL_URL` (points to that service’s control base URL)
- `OCR_MODULE_TIMEOUT`, `OCR_STATUS_POLL_INTERVAL`
- shared directories (if still used by OCR container): `DATA_DIR`, `INDEX_DIR`, `OCR_OUTPUT_DIR`, `OCR_INCOMING_DIR`, `OCR_WARMUP_DIR`

**Provider-only settings (move out of root `.env`)**
- MinerU-only: `MINERU_*`, plus its HF/transformers cache vars if they’re really only needed there.
- Qwen3-only: model path, vision prompting/config, llama-cpp server args, page rendering settings, etc.

Implementation detail (Compose):
- Use an OCR-provider-specific env file, e.g.:
  - `ocr_mineru/mineru.env` (or `ocr_mineru/mineru.env.example`)
  - `ocr_qwen3vl/qwen3.env` (or example)
- In `docker-compose.yml` (or override), attach provider env files only to that provider service.

## Qwen3-VL OCR Module (New Docker Service)

### External behavior

Expose the exact same API contract as today’s OCR module:
- `POST /parse` → enqueue job
- `GET /jobs/{id}` → status/progress
- `GET /jobs/{id}/result` → `{text, metadata}`
- `POST /warmup`
- `POST /control/load`
- `GET /control/status`

**Additionally required for Qwen3-VL only:**
- `POST /control/unload`
  - Purpose: release VRAM (stop llama-cpp process / free model resources).
  - MinerU does not need this; do not add it to MinerU module.

### Internals (PDF → images → VLM → markdown)

Pipeline inside the Qwen3 OCR container:

1. Accept upload; persist to incoming dir for the job.
2. Parse PDF via **PyMuPDF**:
   - open PDF
   - iterate pages
   - render each page to an image (choose DPI / max dimension to balance speed + OCR quality)
3. For each page:
   - feed the rendered page image to a Qwen3-VL model served via **llama-cpp-python** (vision-capable)
   - prompt for a structured output:
     - primary: clean, reading-order markdown text
     - optional: tables in markdown, formulas as LaTeX, etc.
4. Combine per-page outputs into a single markdown/text payload.
5. Return `{text, metadata}`:
   - metadata should include at least:
     - per-page timing summary
     - render settings (dpi, colorspace)
     - model identifier + inference settings
   - optional (nice-to-have): `assets` + `image_blocks` compatible with the existing UI debug endpoints.

### GPU lifecycle (load/unload)

For Qwen3 OCR:
- `POST /control/load`: start the llama-cpp process (or load model in-process) and mark ready.
- `POST /control/unload`: stop process / free model and mark unloaded.
- `GET /control/status`: report loaded/unloaded + basic health.

For MinerU OCR:
- Keep only `load/status` (no unload) as requested.

### Concurrency + batching considerations

Start with the same execution model as MinerU OCR module:
- queue jobs in-memory
- process each job in a background task
- emit progress updates (page index, percent)

Later enhancements (optional):
- batching multiple pages per inference call if supported
- caching rendered pages on disk for retries
- page-level parallelism (careful with VRAM)

## Compose / Switching Strategy

### Recommended approach: compose override files

Keep `docker-compose.yml` as the “default MinerU stack”.

Add an override file for Qwen3 OCR, for example:
- `docker-compose.ocr-qwen3.yml`

That override:
- replaces the `ocr-module` service image/build context to point to the Qwen3 OCR module
- overrides `OCR_PARSER_KEY=qwen3_vl`
- sets `OCR_MODULE_URL` / `OCR_CONTROL_URL` appropriately if service name changes
- mounts models / enables GPU
- loads `ocr_qwen3vl/qwen3.env`

This keeps switching as:
- MinerU: `docker compose up`
- Qwen3: `docker compose -f docker-compose.yml -f docker-compose.ocr-qwen3.yml up`

## Phased Implementation Plan

### Phase 1 — Env compartmentalization (no behavior change)
- Move MinerU-only vars out of root `.env` into `ocr_mineru/mineru.env` (and add an example template).
- Update Compose to load MinerU env file for the MinerU service only.
- Keep backend OCR vars in root `.env` unchanged.

Acceptance:
- MinerU stack still boots and OCR works with the same API and outputs.
- Root `.env` no longer needs `MINERU_*`.

### Phase 2 — Qwen3 OCR container scaffold (API compatibility)
- Create new service directory (e.g. `ocr_qwen3vl/`) with:
  - FastAPI app mirroring endpoints and response shapes
  - job queue/status/result plumbing
  - placeholder implementation that returns deterministic output

Acceptance:
- Backend can call `/parse` → `/jobs/...` → `/result` without code changes (besides env/compose switch).

### Phase 3 — Implement PDF rendering + VLM inference
- Add PyMuPDF rendering
- Add llama-cpp-python integration for Qwen3-VL
- Define prompts + output normalization (markdown cleanup, page separators)

Acceptance:
- End-to-end OCR produces useful text for multi-page PDFs.
- Progress updates reflect page completion.

### Phase 4 — GPU lifecycle for Qwen3 (unload required)
- Implement `POST /control/unload` in Qwen3 OCR service.
- Update backend GPU phase manager logic to call OCR unload **only when available/configured** for Qwen3 deployments (MinerU remains unchanged).

Acceptance:
- Switching phases frees VRAM when running Qwen3 OCR, without requiring MinerU changes.

### Phase 5 — Parity metadata (optional)
- Optionally match MinerU-style `metadata.assets` and `metadata.image_blocks` so existing debug UI can show assets similarly.

Acceptance:
- `/api/documents/.../assets/...` can serve Qwen OCR artifacts (if implemented).

## Open Questions / Decisions

1. Output format strictness:
   - Do we require MinerU-like `metadata.assets` / `image_blocks`, or is `{text, minimal metadata}` sufficient initially?
2. Page rendering defaults:
   - DPI and max dimension targets (quality vs speed).
3. Llama-cpp vision API shape:
   - Whether to run llama-cpp as an internal library call vs a separate server process inside the OCR container.
4. Model placement:
   - Volume-mount `.gguf` model files vs baking into the image.
