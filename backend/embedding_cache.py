from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .persistence import DocumentStore, EmbeddingRow
except ImportError:  # pragma: no cover - script fallback
    from persistence import DocumentStore, EmbeddingRow  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCacheView:
    matrix: np.ndarray
    chunk_ids: Tuple[str, ...]
    doc_hashes: Tuple[str, ...]
    chunks_per_doc: Mapping[str, int]
    dim: int

    @property
    def total(self) -> int:
        return len(self.chunk_ids)


class EmbeddingCache:
    """
    Holds all chunk embeddings in memory for fast cosine similarity search.
    The cache is rebuilt on startup and updated incrementally when documents
    are reprocessed or deleted.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._chunk_ids: Tuple[str, ...] = tuple()
        self._doc_hashes: Tuple[str, ...] = tuple()
        self._chunks_per_doc: Dict[str, int] = {}
        self._dim = 0
        self._model: Optional[str] = None

    async def rebuild(self, store: DocumentStore) -> None:
        rows = await store.fetch_all_embeddings()
        vectors, dim = _build_matrix(rows, None)
        _normalize_rows_inplace(vectors)
        chunk_ids = tuple(row.chunk_id for row in rows)
        doc_hashes = tuple(row.doc_hash for row in rows)
        counts = _count_docs(doc_hashes)
        async with self._lock:
            self._matrix = vectors
            self._chunk_ids = chunk_ids
            self._doc_hashes = doc_hashes
            self._chunks_per_doc = counts
            self._dim = dim
            self._model = rows[0].model if rows else None
        logger.info("Embedding cache loaded %d vectors (dim=%s)", len(chunk_ids), dim)

    def snapshot(self) -> EmbeddingCacheView:
        return EmbeddingCacheView(
            matrix=self._matrix,
            chunk_ids=self._chunk_ids,
            doc_hashes=self._doc_hashes,
            chunks_per_doc=MappingProxyType(self._chunks_per_doc),
            dim=self._dim,
        )

    async def replace_document(self, doc_hash: str, embeddings: Sequence[EmbeddingRow]) -> None:
        async with self._lock:
            keep_indices = [idx for idx, existing in enumerate(self._doc_hashes) if existing != doc_hash]
            kept_matrix = self._matrix[keep_indices] if keep_indices else _empty_matrix(self._dim)
            kept_chunk_ids = tuple(self._chunk_ids[idx] for idx in keep_indices)
            kept_doc_hashes = tuple(self._doc_hashes[idx] for idx in keep_indices)

            target_dim = self._dim
            if embeddings:
                target_dim = embeddings[0].dim
                if self._matrix.size and self._dim not in (0, target_dim):
                    logger.warning(
                        "Embedding dimension changed from %s to %s; replacing cache contents", self._dim, target_dim
                    )
                    kept_matrix = _empty_matrix(target_dim)
                    kept_chunk_ids = tuple()
                    kept_doc_hashes = tuple()

            new_vectors, inferred_dim = _build_matrix(embeddings, target_dim or None)
            _normalize_rows_inplace(new_vectors)

            combined_matrix = _concat_matrices(kept_matrix, new_vectors)
            combined_chunk_ids = kept_chunk_ids + tuple(row.chunk_id for row in embeddings)
            combined_doc_hashes = kept_doc_hashes + tuple(row.doc_hash for row in embeddings)

            self._matrix = combined_matrix
            self._chunk_ids = combined_chunk_ids
            self._doc_hashes = combined_doc_hashes
            self._chunks_per_doc = _count_docs(combined_doc_hashes)
            self._dim = inferred_dim if inferred_dim else target_dim
            if embeddings:
                self._model = embeddings[0].model

    async def remove_document(self, doc_hash: str) -> None:
        await self.replace_document(doc_hash, [])

    @property
    def model(self) -> Optional[str]:
        return self._model


def _build_matrix(rows: Sequence[EmbeddingRow], expected_dim: Optional[int]) -> Tuple[np.ndarray, int]:
    if not rows:
        dim = expected_dim or 0
        return _empty_matrix(dim), dim
    dim = rows[0].dim
    if expected_dim not in (None, 0, dim):
        raise ValueError(f"Incompatible embedding dimension; expected {expected_dim}, got {dim}")
    matrix = np.empty((len(rows), dim), dtype=np.float32)
    for idx, row in enumerate(rows):
        vec = np.asarray(row.vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(f"Embedding vector length mismatch for chunk {row.chunk_id}")
        matrix[idx] = vec
    return matrix, dim


def _empty_matrix(dim: int) -> np.ndarray:
    if dim <= 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.empty((0, dim), dtype=np.float32)


def _normalize_rows_inplace(matrix: np.ndarray) -> None:
    if matrix.size == 0:
        return
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, np.maximum(norms, 1e-8), out=matrix)


def _concat_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size and b.size:
        return np.vstack([a, b])
    if a.size:
        return a
    if b.size:
        return b
    if a.ndim == 2:
        return _empty_matrix(a.shape[1])
    if b.ndim == 2:
        return _empty_matrix(b.shape[1])
    return _empty_matrix(0)


def _count_docs(doc_hashes: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for doc in doc_hashes:
        counts[doc] = counts.get(doc, 0) + 1
    return counts
