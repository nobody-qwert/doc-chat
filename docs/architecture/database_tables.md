# Database Tables

The `DocumentStore` component (`backend/persistence.py`) provisions an on-disk SQLite schema that keeps track of document ingestion, chunking, embeddings, classifications, and chat conversations. The extraction subsystem can also create additional schema-driven metadata tables (e.g., `meta_invoices`) in the same SQLite DB.

By default, the DB lives at `DOC_STORE_PATH` (see `.env`; commonly `app_data/rag_meta.db` via `${APP_DATA_ROOT}/rag_meta.db`).

## Schema overview

```mermaid
erDiagram
    documents {
        TEXT doc_hash PK
        TEXT original_name
        TEXT stored_name
        INTEGER size
        INTEGER token_count
        TEXT status
        TEXT error
        TEXT classification_status
        TEXT classification_error
        TEXT last_classified_at
        TEXT created_at
        TEXT updated_at
        TEXT last_ingested_at
    }
    jobs {
        TEXT job_id PK
        TEXT doc_hash FK
        TEXT status
        TEXT error
        TEXT created_at
        TEXT started_at
        TEXT finished_at
    }
    extractions {
        TEXT doc_hash PK,FK
        TEXT parser PK
        TEXT text
        TEXT meta
        TEXT created_at
    }
    document_classifications {
        TEXT doc_hash PK,FK
        TEXT l1_id
        TEXT l1_name
        TEXT l2_id
        TEXT l2_name
        TEXT l1_confidence
        TEXT l2_confidence
        TEXT l1_reason
        TEXT l2_reason
        TEXT model
        TEXT raw_response
        TEXT updated_at
    }
    chunks {
        TEXT chunk_id PK
        TEXT doc_hash FK
        TEXT chunk_config_id FK
        INTEGER order_index
        TEXT text
        INTEGER token_count
    }
    chunking_configs {
        TEXT config_id PK
        TEXT label
        TEXT description
        INTEGER core_size
        INTEGER left_overlap
        INTEGER right_overlap
        INTEGER step_size
    }
    embeddings {
        TEXT chunk_id PK,FK
        TEXT doc_hash
        INTEGER dim
        TEXT model
        BLOB vector
    }
    performance_metrics {
        TEXT doc_hash PK,FK
        REAL ocr_time_sec
        REAL chunking_time_sec
        REAL embedding_time_sec
        REAL total_time_sec
        TEXT created_at
    }
    meta_invoices {
        TEXT doc_hash PK,FK
        TEXT invoice_number
        TEXT invoice_date
        TEXT due_date
        TEXT vendor_name
        TEXT vendor_address
        TEXT customer_name
        REAL subtotal
        REAL tax_amount
        REAL total_amount
        TEXT currency
        TEXT payment_terms
        TEXT po_number
        TEXT extracted_at
        TEXT extraction_model
    }
    conversations {
        TEXT conversation_id PK
        TEXT summary
        INTEGER total_tokens
        TEXT created_at
        TEXT updated_at
    }
    conversation_messages {
        INTEGER id PK
        TEXT conversation_id FK
        TEXT role
        TEXT content
        INTEGER token_count
        TEXT created_at
    }

    documents ||--o{ jobs : "doc_hash"
    documents ||--o{ extractions : "doc_hash"
    documents ||--|| document_classifications : "doc_hash"
    documents ||--o{ chunks : "doc_hash"
    chunking_configs ||--o{ chunks : "chunk_config_id"
    documents ||--|| performance_metrics : "doc_hash"
    chunks ||--|| embeddings : "chunk_id"
    documents ||--o{ meta_invoices : "doc_hash"
    conversations ||--o{ conversation_messages : "conversation_id"
```

## Table details

### documents

Canonical record for each uploaded file, keyed by a content hash so deduplication is automatic.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `doc_hash` | TEXT | Primary key; stored hash of the raw file contents. |
| `original_name` | TEXT | Original filename from the uploader. |
| `stored_name` | TEXT | On-disk filename inside `DATA_DIR`. |
| `size` | INTEGER | File size in bytes. |
| `token_count` | INTEGER | Total tokenized length captured during ingestion. |
| `status` | TEXT | Current ingestion status (`pending`, `processed`, `error`, etc.). |
| `error` | TEXT | Error string captured when ingestion fails. |
| `classification_status` | TEXT | Classification pipeline status (`pending`, `queued`, `running`, `classified`, `error`, etc.). |
| `classification_error` | TEXT | Error string captured when classification fails. |
| `last_classified_at` | TEXT | ISO timestamp when the document was last classified. |
| `created_at` | TEXT | ISO timestamp when the row was first created. |
| `updated_at` | TEXT | ISO timestamp for the latest mutation. |
| `last_ingested_at` | TEXT | ISO timestamp when the document was last fully ingested. |

**Indexes**

- `idx_docs_status` accelerates dashboard filtering by status.

**Linked tables**

- Referenced by `[jobs](#jobs)`, `[extractions](#extractions)`, `[chunks](#chunks)`, and `[performance_metrics](#performance_metrics)`.

### jobs

Tracks asynchronous ingestion work for each document.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `job_id` | TEXT | Primary key for the async job. |
| `doc_hash` | TEXT | Foreign key to [`documents`](#documents); cascade deletes keep the queue clean. |
| `status` | TEXT | `queued`, `running`, `completed`, `failed`, etc. |
| `error` | TEXT | If populated, explains why the job failed. |
| `created_at` | TEXT | Enqueue timestamp. |
| `started_at` | TEXT | First worker start timestamp. |
| `finished_at` | TEXT | Completion timestamp. |

### extractions

Stores parser-specific raw text and metadata produced during OCR/parsing.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `doc_hash` | TEXT | Part of the composite primary key; foreign key to [`documents`](#documents). |
| `parser` | TEXT | Part of the composite primary key; names the parser variant (e.g., `mineru`). |
| `text` | TEXT | Extracted document text. |
| `meta` | TEXT | JSON metadata string (layout info, parser stats, etc.). |
| `created_at` | TEXT | When this parser output was last refreshed. |

**Indexes**

- `idx_ext_doc` supports quick lookups by `doc_hash`.

### document_classifications

Latest classification result for each document. This is updated by the ingestion worker and drives category-based features (like schema-driven extraction).

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `doc_hash` | TEXT | Primary key and foreign key to [`documents`](#documents). |
| `l1_id` | TEXT | Required; L1 category identifier. |
| `l1_name` | TEXT | Optional display name for L1 category. |
| `l2_id` | TEXT | Optional; L2 category identifier. |
| `l2_name` | TEXT | Optional display name for L2 category. |
| `l1_confidence` | TEXT | Optional confidence value as returned by the classifier. |
| `l2_confidence` | TEXT | Optional confidence value as returned by the classifier. |
| `l1_reason` | TEXT | Optional explanation text for L1 classification. |
| `l2_reason` | TEXT | Optional explanation text for L2 classification. |
| `model` | TEXT | Optional classifier model identifier. |
| `raw_response` | TEXT | JSON blob (stringified) of the classifier output. |
| `updated_at` | TEXT | ISO timestamp when the classification row was last updated. |

### chunking_configs

Defines the sliding-window presets used during chunking. Each config captures the window core size plus overlaps so downstream consumers can tell exactly how a chunk was produced.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `config_id` | TEXT | Primary key (e.g., `chunk-small`). |
| `label` | TEXT | Friendly name shown in diagnostics. |
| `description` | TEXT | Optional human-readable summary. |
| `core_size` | INTEGER | Number of tokens in the center window. |
| `left_overlap` | INTEGER | Tokens included before the core window. |
| `right_overlap` | INTEGER | Tokens included after the core window. |
| `step_size` | INTEGER | Sliding-step in tokens for the next window. |

The startup sequence syncs configured presets into this table so foreign keys can reference them.

### chunks

Holds ordered text snippets that feed embedding generation and retrieval.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `chunk_id` | TEXT | Primary key. |
| `doc_hash` | TEXT | Foreign key to [`documents`](#documents). |
| `chunk_config_id` | TEXT | Foreign key to [`chunking_configs`](#chunking_configs); identifies which window preset produced the row. |
| `order_index` | INTEGER | Maintains the original layout order. |
| `text` | TEXT | Chunk contents. |
| `token_count` | INTEGER | Tokenized length (used for window sizing and metrics). |

**Indexes**

- `idx_chunks_doc` speeds up chunk scans per document.
- `idx_chunks_config` enables quick grouping/filtering by configuration.

### embeddings

Vector store for each chunk. Vectors are stored as packed float32 blobs, and each embedding row is deleted automatically when its chunk disappears.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `chunk_id` | TEXT | Primary key and foreign key to [`chunks`](#chunks). |
| `doc_hash` | TEXT | Denormalized copy of the owning document hash (not enforced by FK, but kept in sync by the service). |
| `dim` | INTEGER | Embedding dimensionality. |
| `model` | TEXT | Model identifier (e.g., `text-embedding-3-large`). |
| `vector` | BLOB | Packed float32 array. |

**Indexes**

- `idx_emb_doc` is used when counting vectors per document.

### performance_metrics

One row per document summarizing ingestion timings.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `doc_hash` | TEXT | Primary key and foreign key to [`documents`](#documents). |
| `ocr_time_sec` | REAL | OCR duration. |
| `chunking_time_sec` | REAL | Time spent chunking. |
| `embedding_time_sec` | REAL | Time taken to embed all chunks. |
| `total_time_sec` | REAL | Wall-clock from job start to finish. |
| `created_at` | TEXT | When the metric snapshot was recorded. |

**Indexes**

- `idx_perf_doc` duplicates the primary key for quick `doc_hash` probing.

### Extraction metadata tables (dynamic)

The extraction subsystem can create additional tables in the same SQLite DB based on `backend/services/extraction/schemas.py`. Each table has:

- `doc_hash TEXT PRIMARY KEY` + FK to `documents(doc_hash)` (`ON DELETE CASCADE`)
- Schema-specific fields (TEXT/REAL/INTEGER)
- `extracted_at TEXT NOT NULL`
- `extraction_model TEXT`

Currently, the only registered schema is `invoices`, which creates:

#### meta_invoices

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `doc_hash` | TEXT | Primary key and foreign key to [`documents`](#documents). |
| `invoice_number` | TEXT | Extracted invoice number. |
| `invoice_date` | TEXT | Extracted invoice date (typically `YYYY-MM-DD`). |
| `due_date` | TEXT | Extracted due date (typically `YYYY-MM-DD`). |
| `vendor_name` | TEXT | Extracted vendor name. |
| `vendor_address` | TEXT | Extracted vendor address. |
| `customer_name` | TEXT | Extracted customer name. |
| `subtotal` | REAL | Extracted subtotal amount. |
| `tax_amount` | REAL | Extracted tax amount. |
| `total_amount` | REAL | Extracted total amount. |
| `currency` | TEXT | Extracted currency code. |
| `payment_terms` | TEXT | Extracted payment terms. |
| `po_number` | TEXT | Extracted PO number. |
| `extracted_at` | TEXT | ISO timestamp when extraction was written. |
| `extraction_model` | TEXT | LLM model identifier used for extraction. |

**Indexes**

- The service attempts to create `idx_<table>_<field>` indexes for TEXT/REAL/INTEGER fields (best-effort).

### conversations

High-level record for every chat thread in the UI.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `conversation_id` | TEXT | Primary key. |
| `summary` | TEXT | Rolling abstractive summary of the chat (used for context compression). |
| `total_tokens` | INTEGER | Running total that is incremented per message. |
| `created_at` | TEXT | Creation timestamp. |
| `updated_at` | TEXT | Timestamp of the last message/summary update. |

### conversation_messages

Individual chat messages plus their token counts; deleting a conversation cascades to its messages.

| Column | Type | Keys & constraints |
| --- | --- | --- |
| `id` | INTEGER | Primary key (`AUTOINCREMENT`). |
| `conversation_id` | TEXT | Foreign key to [`conversations`](#conversations). |
| `role` | TEXT | Message role (`user`, `assistant`, `system`, etc.). |
| `content` | TEXT | Message body. |
| `token_count` | INTEGER | Count used to update the parent's `total_tokens`. |
| `created_at` | TEXT | Message timestamp. |

**Indexes**

- `idx_conv_msgs_conv` (`conversation_id`, `id`) allows fetching ordered transcripts efficiently.
