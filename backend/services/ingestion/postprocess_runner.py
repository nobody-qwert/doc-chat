from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

try:
    from ..chunking import ChunkWindowSpec
    from ..embeddings import EmbeddingClient
except ImportError:  # pragma: no cover
    from chunking import ChunkWindowSpec  # type: ignore
    from embeddings import EmbeddingClient  # type: ignore

from .context import document_store
from .pipeline import build_chunk_specs, run_postprocess_pipeline


async def run_postprocess_for_doc(
    doc_hash: str,
    *,
    ocr_text: str,
    ocr_time: Optional[float] = None,
    emb_client: Optional[EmbeddingClient] = None,
    chunk_specs: Optional[Sequence[ChunkWindowSpec]] = None,
    chunk_progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    embedding_progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_chunk_start: Optional[Callable[[], None]] = None,
    on_embedding_start: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Shared postprocess driver that updates document status and delegates to the pipeline.
    """
    client = emb_client or EmbeddingClient()
    specs = list(chunk_specs) if chunk_specs is not None else build_chunk_specs()
    await document_store.update_document_status(doc_hash, "processing")
    try:
        return await run_postprocess_pipeline(
            doc_hash,
            ocr_text=ocr_text,
            emb_client=client,
            chunk_specs=specs,
            ocr_time=ocr_time,
            chunk_progress_cb=chunk_progress_cb,
            embedding_progress_cb=embedding_progress_cb,
            on_chunk_start=on_chunk_start,
            on_embedding_start=on_embedding_start,
        )
    except Exception as exc:
        await document_store.update_document_status(doc_hash, "error", error=str(exc))
        raise
