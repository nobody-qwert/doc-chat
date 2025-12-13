# Cleanup Plan: Remove Classification + Invoice Metadata Extraction

## Goal
Simplify the app to only do:
1) OCR parse (extract text)
2) Chunking + embedding generation

Remove entirely:
- Document classification (L1/L2 taxonomy, pipeline step, API surface, DB tables/columns, UI).
- Invoice metadata extraction feature (schemas, endpoints, UI, dynamic metadata tables).

You will redeploy with a fresh DB, so no migration/retention work is required.

## Target Pipeline (After Cleanup)
- Upload -> OCR parse -> persist extracted text
- Second step (manual trigger or part of ingest job, depending on current behavior): chunking + embeddings
- Chat/RAG uses chunks + embeddings only

No classification phase. No extraction phase.

## Backend Work Plan

### 1) Remove classification feature end-to-end
- Persistence/schema:
  - Remove `documents.classification_status`, `documents.classification_error`, `documents.last_classified_at` from `DocumentStore.init()` table definition.
  - Remove `document_classifications` table creation.
  - Remove classification-related CRUD methods in `backend/persistence.py`:
    - `update_classification_status`, `clear_classification`, `save_document_classification`, `get_classification`.
  - Remove any references to those fields in document deletion logic.

- Ingestion pipeline:
  - Remove the classification phase implementation in `backend/services/ingestion/worker.py`.
  - Remove classification job queueing (`queue_classification_job`) and any job/status bookkeeping for it.
  - Remove `backend/services/ingestion/classification.py` and taxonomy plumbing (`backend/services/ingestion/taxonomy.py`) if nothing else uses it.

- API routes:
  - Remove `POST /api/ingest/{doc_hash}/classify` from `backend/routes/ingest.py` and corresponding service function in `backend/services/ingestion/api.py`.
  - Update `reprocess_all` behavior to only queue postprocess (chunking+embedding). Remove references to `classification_job_id`.

- Document payloads:
  - Remove classification fields from API responses:
    - `backend/routes/helpers.py` (`format_document_row`) currently emits `classification_*` fields.
    - `backend/routes/documents.py` currently fetches and returns `classification`.
  - Ensure any downstream code that expects these fields is updated/removed.

- Agentic/RAG tools:
  - Remove classification usage from `backend/services/agentic/tools.py` (it currently includes classification in document metadata and uses it to pick extraction schemas).

### 2) Remove invoice metadata extraction feature
- API routes:
  - Remove `backend/routes/extraction.py` router and unregister it from the main app/router include.

- Extraction service code:
  - Remove `backend/services/extraction/` components:
    - `schemas.py`, `persistence.py` (`MetadataStore`), `engine.py` (and any related helpers).

- Settings/env:
  - Remove any extraction-related environment toggles referenced by the frontend (e.g. `VITE_ENABLE_INVOICE_EXTRACTION`) if they become unused.

- DB surface:
  - Because DB is fresh, it’s enough to remove the code paths that create/expect dynamic tables like `meta_invoices`.

## Frontend/UI Work Plan
- Remove classification UI from `frontend/src/IngestPage.jsx`:
  - Remove status badge, details block, “Re-run classification” action, and all state/handlers (`classifyingHash`, `handleReclassify`, etc.).

- Remove invoice metadata extraction UI from `frontend/src/IngestPage.jsx`:
  - Remove extraction stats polling (`/api/extraction/stats/...`), “Extract Invoices” button, per-doc extraction badge/button, and metadata panel/state.
  - Remove derivations that depend on `d.classification` (e.g. invoice counts) since classification is gone.

- Verify no other UI pages reference these endpoints/fields.

## Docs Cleanup
- Update/remove docs that describe classification and extraction:
  - `docs/architecture/doc_taxonomy_and_prompts.md` (remove or mark obsolete).
  - `docs/extraction_plan.md` (remove/adjust classification/extraction references).
  - `docs/architecture/database_tables.md` (remove classification + extraction metadata tables/columns).

## Validation Checklist
- Backend:
  - `python3 -m py_compile` over touched backend modules.
  - Start backend and verify:
    - Upload + ingest works
    - OCR text preview works
    - Postprocess (chunking+embedding) works
    - Chat works

- Frontend:
  - App loads and Document Library renders without classification/extraction fields.
  - Reprocess flows still work.

## Deliverable Order (Suggested)
1) Backend removal (classification + extraction) and adjust routes
2) Frontend cleanup
3) Docs update
4) Quick smoke run
