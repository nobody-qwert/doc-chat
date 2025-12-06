from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

try:  # When imported as part of package
    from .config import load_settings
    from .embedding_cache import EmbeddingCache
    from .persistence import DocumentStore
    from .services.gpu_phases import GPUPhaseManager
except ImportError:  # pragma: no cover - script fallback
    from config import load_settings  # type: ignore
    from embedding_cache import EmbeddingCache  # type: ignore
    from persistence import DocumentStore  # type: ignore
    from services.gpu_phases import GPUPhaseManager  # type: ignore

settings = load_settings()
document_store = DocumentStore(settings.doc_store_path, chunking_configs=settings.chunking_configs)
embedding_cache = EmbeddingCache()
jobs_registry: Dict[str, Dict[str, Any]] = {}
gpu_phase_manager = GPUPhaseManager(
    settings.llm_control_url,
    settings.ocr_control_url,
    llm_inference_url=settings.llm_base_url,
    timeout=settings.gpu_phase_timeout,
    ready_check_timeout=settings.llm_ready_timeout,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await document_store.init()
    await embedding_cache.rebuild(document_store)
    yield
