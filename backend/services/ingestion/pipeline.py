from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

try:
    from ..chunking import ChunkWindowSpec, chunk_text_multi, count_text_tokens
    from ..embeddings import EmbeddingClient
    from ..persistence import EmbeddingRow
except ImportError:  # pragma: no cover
    from chunking import ChunkWindowSpec, chunk_text_multi, count_text_tokens  # type: ignore
    from embeddings import EmbeddingClient  # type: ignore
    from persistence import EmbeddingRow  # type: ignore

from .context import document_store, settings, embedding_cache

logger = logging.getLogger(__name__)


def build_chunk_specs() -> List[ChunkWindowSpec]:
    return [
        ChunkWindowSpec(
            name="small",
            core_size=settings.chunk_size,
            left_padding=settings.chunk_overlap,
            right_padding=settings.chunk_overlap,
            step_size=settings.chunk_size,
        ),
    ]


def run_chunking(
    text: str,
    chunk_specs: Sequence[ChunkWindowSpec],
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Any]:
    chunk_views = chunk_text_multi(
        text,
        chunk_specs,
        tokenizer_id=settings.embedding_tokenizer_id,
        progress_cb=progress_cb,
    )
    ordered_specs = list(chunk_specs)
    spec_names = [spec.name for spec in ordered_specs] if ordered_specs else list(chunk_views.keys())
    combined: List[Any] = []
    for name in spec_names:
        combined.extend(chunk_views.get(name, []))
    return combined


async def compute_embeddings_for_chunks(
    chunks: List[Dict[str, Any]],
    client: EmbeddingClient,
    doc_hash: str,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[EmbeddingRow]:
    total = len(chunks)
    if total == 0:
        if progress_cb:
            progress_cb({"percent": 100.0, "processed": 0, "total": 0})
        return []

    rows: List[EmbeddingRow] = []
    batch_size = getattr(client, "max_batch", 1) or 1

    last_percent_reported = 0.0

    for start in range(0, total, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch_chunks]
        vectors = await client.embed_batch(texts)
        if len(vectors) != len(batch_chunks):
            raise RuntimeError("Embedding result size mismatch")
        if client.dim is None:
            raise RuntimeError("Embedding dimension is unknown after embedding call")

        for chunk, vector in zip(batch_chunks, vectors):
            rows.append(
                EmbeddingRow(
                    chunk_id=chunk["chunk_id"],
                    doc_hash=doc_hash,
                    dim=client.dim,
                    model=client.model,
                    vector=vector,
                )
            )

        if progress_cb:
            done = min(total, start + len(batch_chunks))
            percent = 100.0 if total == 0 else (done / total) * 100.0
            last_percent_reported = percent
            progress_cb({"percent": percent, "processed": done, "total": total})

    if progress_cb and last_percent_reported < 100.0:
        progress_cb({"percent": 100.0, "processed": total, "total": total})

    return rows


async def process_document_text(
    doc_hash: str,
    ocr_text: str,
    *,
    emb_client: EmbeddingClient,
    chunk_specs: Optional[Sequence[ChunkWindowSpec]] = None,
    chunk_progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    embedding_progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_embedding_start: Optional[Callable[[], None]] = None,
) -> Tuple[
    List[Tuple[str, int, str, int]],
    List[EmbeddingRow],
    float,
    float,
]:
    if not ocr_text:
        raise RuntimeError("OCR text is empty; cannot post-process")

    specs = list(chunk_specs) if chunk_specs else build_chunk_specs()

    def _scoped_chunk_id(raw_id: str) -> str:
        raw_id = raw_id.strip()
        if not raw_id:
            return uuid4().hex
        return raw_id

    start_chunking = time.perf_counter()
    chunks = run_chunking(ocr_text, specs, progress_cb=chunk_progress_cb)
    if not chunks:
        raise RuntimeError("Chunking produced no chunks; check chunking settings")

    def _rows_for(chunks: Sequence[Any]) -> List[Tuple[str, int, str, int]]:
        rows: List[Tuple[str, int, str, int]] = []
        for chunk in chunks:
            scoped_id = _scoped_chunk_id(getattr(chunk, "chunk_id", "") or "")
            rows.append((scoped_id, chunk.order_index, chunk.text, chunk.token_count))
        return rows

    chunk_rows = _rows_for(chunks)
    chunking_time = time.perf_counter() - start_chunking

    if on_embedding_start:
        on_embedding_start()

    start_embedding = time.perf_counter()
    rows = await compute_embeddings_for_chunks(
        [
            {"chunk_id": cid, "order_index": idx, "text": txt, "token_count": tok}
            for (cid, idx, txt, tok) in chunk_rows
        ],
        emb_client,
        doc_hash,
        progress_cb=embedding_progress_cb,
    )
    embedding_time = time.perf_counter() - start_embedding
    return chunk_rows, rows, chunking_time, embedding_time


async def persist_chunk_and_embedding_results(
    *,
    doc_hash: str,
    chunk_rows: Sequence[Tuple[str, int, str, int]],
    embeddings: Sequence[EmbeddingRow],
    chunking_time: float,
    embedding_time: float,
    doc_token_count: int,
    ocr_time: Optional[float] = None,
) -> None:
    await document_store.replace_chunks(doc_hash, settings.chunk_config_small_id, list(chunk_rows))
    embedding_rows = list(embeddings)
    await document_store.replace_embeddings(embedding_rows)
    await embedding_cache.replace_document(doc_hash, embedding_rows)
    await document_store.update_document_token_count(doc_hash, doc_token_count)

    prev_metrics = await document_store.get_performance_metrics(doc_hash)
    ocr_time_val = ocr_time
    if ocr_time_val is None and prev_metrics:
        try:
            raw_val = prev_metrics.get("ocr_time_sec")
            ocr_time_val = float(raw_val) if raw_val is not None else None
        except (TypeError, ValueError):
            ocr_time_val = prev_metrics.get("ocr_time_sec")

    total_time = (ocr_time_val or 0.0) + chunking_time + embedding_time

    try:
        await document_store.save_performance_metrics(
            doc_hash,
            ocr_time_sec=ocr_time_val,
            chunking_time_sec=chunking_time,
            embedding_time_sec=embedding_time,
            total_time_sec=total_time,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to save performance metrics for %s: %s", doc_hash, exc)
    await document_store.mark_document_processed(doc_hash)


async def run_postprocess_pipeline(
    doc_hash: str,
    *,
    ocr_text: str,
    emb_client: EmbeddingClient,
    chunk_specs: Optional[Sequence[ChunkWindowSpec]] = None,
    ocr_time: Optional[float] = None,
    chunk_progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    embedding_progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk_start: Optional[Callable[[], None]] = None,
    on_embedding_start: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    if on_chunk_start:
        on_chunk_start()
    doc_token_count = count_text_tokens(ocr_text, tokenizer_id=settings.embedding_tokenizer_id)
    chunk_rows, rows, chunking_time, embedding_time = await process_document_text(
        doc_hash,
        ocr_text,
        emb_client=emb_client,
        chunk_specs=chunk_specs,
        chunk_progress_cb=chunk_progress_cb,
        embedding_progress_cb=embedding_progress_cb,
        on_embedding_start=on_embedding_start,
    )
    await persist_chunk_and_embedding_results(
        doc_hash=doc_hash,
        chunk_rows=chunk_rows,
        embeddings=rows,
        chunking_time=chunking_time,
        embedding_time=embedding_time,
        doc_token_count=doc_token_count,
        ocr_time=ocr_time,
    )
    chunk_count = len(chunk_rows)
    embedding_count = len(rows)
    return {
        "chunk_count": chunk_count,
        "total_chunks": chunk_count,
        "total_embeddings": embedding_count,
        "chunking_time_sec": chunking_time,
        "embedding_time_sec": embedding_time,
    }
