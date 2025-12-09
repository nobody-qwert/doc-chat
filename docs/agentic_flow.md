# Agentic RAG Workflow

This document explains how the current backend implements the agentic retrieval flow. It ties together the FastAPI routes, the orchestrator, the LLM modes, and the search tools so you can see the full loop end to end.

## Flow Diagram

```mermaid
flowchart TD
    U["User\nPOST /ask/agentic/stream"] --> R["FastAPI route\nvalidate query & GPU"]
    R --> C["Create AsyncOpenAI\n+ EmbeddingClient"]
    C --> O["Orchestrator\nstream_agentic_answer()"]

    subgraph "LLM Agents"
        D[[Decomposer LLM]]
        RV[[Reviewer LLM]]
        CA[[Composer LLM]]
    end

    O --> D
    D --> I["Initial tool calls\nkeyword + semantic search\n(context expansion: full-doc ≤12k tokens\nor ±window for long docs)"]
    I --> E["Evidence store"]
    E --> RV
    RV --> T{Reviewer decision}
    T -- "clarify" --> CL["clarification message\nbuild_clarification_response()"]
    T -- "enough" --> INS["Inspect evidence\n(inspect_evidence)"]
    T -- "more" --> M["Tool request\n(next_tool_call)"]
    M --> TD["execute_tool() dispatcher"]

    subgraph "Search Tools"
        TD --> ST["search_text"]
        TD --> SS["search_semantic"]
        TD --> MD["get_document_metadata"]
    end

    ST --> E
    SS --> E
    MD --> E
    INS --> H{"Inspector hits?"}
    H -- "yes" --> CAI["Compose Answer (Inspector)"]
    H -- "no" --> PR["Prune evidence\n_top 15_"]
    PR --> CA
    CAI --> VC
    CA --> VC["verify_citations()"]
    VC --> RESP["NDJSON stream\nanswer + citations"]
    CL --> RESP
    RESP --> S["Response\napplication/x-ndjson"]
```

## Step-by-step Notes

- **Request intake** – `/ask/agentic/stream` validates the query, confirms that processed documents exist, ensures a GPU-hosted LLM is available, then instantiates `AsyncOpenAI` and `EmbeddingClient` before handing control to `stream_agentic_answer()` (`backend/routes/agentic.py`).
- **Query decomposition** – The orchestrator immediately calls `decompose_query()` to capture the user intent, enumerate subqueries, and optionally record output preferences. No routing or search details are produced at this stage; if parsing fails the orchestrator falls back to a single-subquery structure (`backend/services/agentic/orchestrator.py` and `modes.py`).
- **Search seeding** – After decomposition, the orchestrator mirrors each subquery into hybrid keyword and semantic queries (plus entity/context hints) so both retrieval tools are exercised without another LLM hop.
- **Initial retrieval** – For up to two buckets and two initial queries, the orchestrator calls `execute_tool()` with `search_text` and/or `search_semantic` depending on the strategy, collecting snippets as evidence. `search_text` now returns entire short documents (≤12k tokens) as single evidence blobs and, for longer documents, falls back to semantic chunk selection when keyword hits are weak. Each tool now runs independently—if a call returns zero results the system simply records that outcome and moves on without auto-invoking the other search mode. Evidence is deduplicated by `chunk_id` to keep the context lean.
- **Context expansion** – `backend/services/agentic/tools.py` now promotes short documents (≤12k tokens, or ≤20 chunks when token counts are missing) into single “full_doc” evidence blobs and expands long-document hits by stitching a ±10-chunk window (capped at 20 chunks). This ensures the model sees nearby paragraphs even when the exact fact is not inside the initial hit.
- **Evidence review loop** – `review_evidence()` inspects the current plan and snippets and returns a status:
  - `enough` – break out of the loop and move toward answering.
  - `more` – the response includes a `next_tool_call`; the orchestrator executes it (can be `search_text`, `search_semantic`, or `get_document_metadata`) and re-enters the review step until the tool budget is exhausted.
  - `clarify` – the loop halts and the user receives a clarification prompt built by `build_clarification_response()`.
  The streaming endpoint caps this review loop at two iterations even if the tool budget remains to keep UI latency predictable.
- **Evidence inspection** – Before pruning or composing, the orchestrator calls a dedicated inspector LLM mode that walks the first `max_items` evidence entries (whatever order they currently have—typically highest score after dedup) and attempts to extract key facts. The inspector can gather multiple hits, and when it does, those structured snippets are passed to the composer so the final answer is generated from distilled data rather than raw documents; otherwise the flow falls back to pruning + standard composition.
- **Evidence pruning** – Once gathering stops, evidence is sorted by score and trimmed to the top 15 items to protect the token budget for the final prompt.
- **Answer composition** – `compose_answer()` runs in streaming mode so tokens are forwarded to the client as soon as the model begins responding. It receives only the curated evidence and must cite doc hashes directly.
- **Citation verification** – `verify_citations()` post-processes the answer, replacing any citation that does not point to the known evidence with `[citation needed]` to prevent hallucinated references.
- **Response packaging** – The orchestrator emits NDJSON frames (`step`, `token`, `final`) so clients can visualize tool activity in real time alongside the final cited answer.

## Components and Responsibilities

- **LLM modes & prompts** – `backend/services/agentic/modes.py` and `prompts.py` hold the system messages and user templates that constrain the decomposer, reviewer, and composer LLM behaviors. JSON helpers enforce that each mode returns machine-readable control data.
- **Inspector mode** – The new inspector in `modes.py` uses `INSPECTOR_SYSTEM_PROMPT` / `INSPECTOR_USER_TEMPLATE` to comb through top evidence items, extract structured answers, and short-circuit the flow when a single snippet suffices.
- **Tool layer** – `backend/services/agentic/tools.py` implements keyword search, semantic search, and document metadata lookups, plus the expansion helpers that emit whole-document or chunk-window evidence entries. The orchestrator only ever calls these tools via `execute_tool()` to keep logging and result normalization consistent.
- **Evidence hygiene** – Utility helpers (`_deduplicate_evidence`, `_prune_evidence`, `_build_sources`) guard against bloated context windows and make sure the frontend receives compact previews alongside the answer.

Together, these pieces form a repeatable loop: decompose → plan → search/review (with tool calls) → compose → verify, with targeted handling whenever errors or clarification requests arise.
