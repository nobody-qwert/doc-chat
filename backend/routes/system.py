from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter

try:
    from ..dependencies import document_store, jobs_registry, settings, gpu_phase_manager
    from ..tokenizer_registry import tokenizer_diagnostics
    from .helpers import format_document_row
    from ..services.ingestion import warmup_mineru
    from ..services.rag import warmup_llm
    from ..utils.gpu import get_gpu_snapshot
except ImportError:  # pragma: no cover
    from dependencies import document_store, jobs_registry, settings, gpu_phase_manager  # type: ignore
    from tokenizer_registry import tokenizer_diagnostics  # type: ignore
    from routes.helpers import format_document_row  # type: ignore
    from services.ingestion import warmup_mineru  # type: ignore
    from services.rag import warmup_llm  # type: ignore
    from utils.gpu import get_gpu_snapshot  # type: ignore

router = APIRouter()


def _format_chunk_config(spec) -> str:
    return (
        f"core={spec.core_size} left={spec.left_overlap} "
        f"right={spec.right_overlap} step={spec.step_size}"
    )


def _get_chunk_config(config_id):
    for spec in settings.chunking_configs:
        if spec.config_id == config_id:
            return spec
    return None


def _settings_snapshot() -> Dict[str, Dict[str, Any]]:
    env = os.environ
    tokenizer_diag = tokenizer_diagnostics()
    llm_tok = tokenizer_diag.get(settings.llm_tokenizer_id, {})
    embedding_tok = tokenizer_diag.get(settings.embedding_tokenizer_id, {})
    return {
        "ocr": {
            "parser_key": settings.ocr_parser_key,
            "module_url": settings.ocr_module_url,
            "timeout_sec": settings.ocr_module_timeout,
            "status_poll_sec": settings.ocr_status_poll_interval,
        },
        "chunking": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunk_config_small_id": settings.chunk_config_small_id,
            "window": _format_chunk_config(
                _get_chunk_config(settings.chunk_config_small_id) or settings.chunking_configs[0]
            ),
        },
        "embedding": {
            "base_url": env.get("EMBEDDING_BASE_URL", ""),
            "model": env.get("EMBEDDING_MODEL", ""),
            "batch_size": env.get("EMBEDDING_BATCH_SIZE", "1"),
            "context_size": env.get("EMBED_CONTEXT_SIZE", ""),
            "tokenizer_id": settings.embedding_tokenizer_id,
            "tokenizer_loaded": embedding_tok.get("loaded"),
            "tokenizer_fallbacks": embedding_tok.get("fallback_count"),
            "tokenizer_last_error": embedding_tok.get("error"),
        },
        "llm": {
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "context_window": settings.chat_context_window,
            "max_completion_tokens": settings.chat_completion_max_tokens,
            "reserve_tokens": settings.chat_completion_reserve,
            "tokenizer_id": settings.llm_tokenizer_id,
            "tokenizer_loaded": llm_tok.get("loaded"),
            "tokenizer_fallbacks": llm_tok.get("fallback_count"),
            "tokenizer_last_error": llm_tok.get("error"),
        },
        "retrieval": {
            "min_context_similarity": settings.min_context_similarity,
        },
        "storage": {
            "data_dir": str(settings.data_dir),
            "index_dir": str(settings.index_dir),
            "doc_store": str(settings.doc_store_path),
        },
    }


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
async def system_status() -> dict:
    docs = await document_store.list_documents()
    ready = any((doc.get("status") == "processed") for doc in docs)
    running_jobs = [
        (jid, info)
        for jid, info in jobs_registry.items()
        if info.get("status") in {"queued", "running"}
    ]
    gpu_phase = gpu_phase_manager.snapshot()

    def _job_payload(jid: str, info: Dict[str, Any]) -> Dict[str, Any]:
        docs = info.get("docs") or []
        return {
            "job_id": jid,
            "doc_hash": info.get("hash"),
            "status": info.get("status"),
            "file": info.get("file"),
            "progress": info.get("progress"),
            "docs": docs,
            "total_docs": info.get("total_docs", len(docs)),
            "phase": info.get("phase"),
        }
    return {
        "ready": ready,
        "has_running_jobs": len(running_jobs) > 0,
        "running_jobs": [_job_payload(jid, info) for jid, info in running_jobs],
        "total_jobs": len(jobs_registry),
        "docs_count": sum(1 for doc in docs if str(doc.get("status")).lower() == "processed"),
        "total_docs": len(docs),
        "jobs": [_job_payload(jid, info) for jid, info in jobs_registry.items()],
        "gpu_phase": gpu_phase,
        "documents": [format_document_row(doc) for doc in docs],
        "llm_ready": gpu_phase.get("state") == "llm",
        "settings": _settings_snapshot(),
    }


@router.get("/ready")
async def ready() -> dict[str, bool]:
    ready_flag = (await document_store.count_documents(status="processed")) > 0
    return {"ready": ready_flag}


@router.post("/warmup")
async def warmup() -> dict:
    return await warmup_llm()


@router.post("/warmup/mineru")
async def warmup_mineru_route() -> dict:
    return await warmup_mineru()


@router.get("/diagnostics/gpu")
async def diagnostics_gpu() -> Dict[str, Any]:
    return await get_gpu_snapshot()
