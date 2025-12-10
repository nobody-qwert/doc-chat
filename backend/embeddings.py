from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable {name}")
    cleaned = value.strip()
    if not cleaned:
        raise RuntimeError(f"Environment variable {name} cannot be empty")
    return cleaned


class EmbeddingClient:
    def __init__(self) -> None:
        base = _require_env("EMBEDDING_BASE_URL")
        key = _require_env("EMBEDDING_API_KEY")
        model = _require_env("EMBEDDING_MODEL")
        batch_size = _require_env("EMBEDDING_BATCH_SIZE")
        self.base = base
        self.key = key
        self.model = model
        self.dim: Optional[int] = None
        try:
            size_int = int(batch_size)
        except ValueError as exc:  # pragma: no cover - fail fast on invalid configuration
            raise RuntimeError(f"EMBEDDING_BATCH_SIZE must be an integer (got {batch_size!r})") from exc
        if size_int <= 0:
            raise RuntimeError("EMBEDDING_BATCH_SIZE must be a positive integer")
        self.max_batch = size_int

        from openai import AsyncOpenAI  # type: ignore

        self._client = AsyncOpenAI(base_url=self.base, api_key=self.key)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        out: List[np.ndarray] = []
        for start in range(0, len(texts), self.max_batch):
            batch = texts[start : start + self.max_batch]
            vectors = await self._client.embeddings.create(model=self.model, input=batch)  # type: ignore[attr-defined]
            data = vectors.data  # type: ignore
            for item in data:
                v = np.asarray(item.embedding, dtype=np.float32)  # type: ignore[attr-defined]
                current_dim = int(v.shape[0])
                if self.dim is None:
                    self.dim = current_dim
                elif self.dim != current_dim:
                    raise RuntimeError(f"Embedding dimension changed from {self.dim} to {current_dim}")
                out.append(v)

        if self.dim is None:
            raise RuntimeError("Failed to determine embedding dimension from embeddings response")
        return out
