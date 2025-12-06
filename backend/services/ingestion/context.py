from __future__ import annotations

try:  # pragma: no cover - optional relative import for scripts
    from ..dependencies import document_store, jobs_registry, settings, gpu_phase_manager, embedding_cache
except ImportError:  # pragma: no cover
    from dependencies import document_store, jobs_registry, settings, gpu_phase_manager, embedding_cache  # type: ignore

__all__ = [
    "document_store",
    "jobs_registry",
    "settings",
    "gpu_phase_manager",
    "embedding_cache",
]
