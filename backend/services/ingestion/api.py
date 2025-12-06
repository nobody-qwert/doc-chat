from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException, UploadFile

from .context import document_store, jobs_registry, settings, embedding_cache
from .models import QueuedBatchDoc
from .ocr import warmup_mineru as warmup_mineru_call
from .uploads import prepare_upload
from .worker import queue_batch_job, queue_classification_job, queue_postprocess_job


async def ingest_file(file: UploadFile) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    return await ingest_files([file])


async def ingest_files(files: Sequence[UploadFile]) -> Dict[str, Any]:
    uploads = [f for f in files if f is not None]
    if not uploads:
        raise HTTPException(status_code=400, detail="No files uploaded")

    prepared_docs: List[QueuedBatchDoc] = []
    results: List[Dict[str, Any]] = []

    for upload in uploads:
        doc_info, payload = await prepare_upload(upload)
        results.append(payload)
        if doc_info:
            prepared_docs.append(doc_info)

    job_id: Optional[str] = None
    if prepared_docs:
        job_id = await queue_batch_job(prepared_docs)
        for payload in results:
            if payload.get("status") == "queued":
                payload["job_id"] = job_id

    skipped = sum(1 for r in results if r.get("status") == "skipped")
    return {
        "job_id": job_id,
        "results": results,
        "queued_count": len(prepared_docs),
        "skipped_count": skipped,
        "total": len(results),
    }


async def retry_ingest(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    stored_name = doc.get("stored_name") or doc_hash
    source_path = settings.data_dir / stored_name
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Stored document missing")

    display_name = doc.get("original_name") or stored_name
    job_doc = QueuedBatchDoc(doc_path=source_path, doc_hash=doc_hash, display_name=display_name)
    job_id = await queue_batch_job([job_doc])
    return {
        "job_id": job_id,
        "status": "queued",
        "file": display_name,
        "hash": doc_hash,
        "retry": True,
    }


def get_job_status(job_id: str) -> Dict[str, Any]:
    info = jobs_registry.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **info}


async def delete_document(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    active_jobs = [
        jid
        for jid, info in jobs_registry.items()
        if info.get("status") in {"queued", "running"}
        and any(d.get("hash") == doc_hash for d in info.get("docs", []))
    ]
    if active_jobs:
        raise HTTPException(status_code=409, detail="Document is currently processing and cannot be removed")

    stored_name = doc.get("stored_name") or doc_hash
    file_path = settings.data_dir / stored_name
    file_removed = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_removed = True
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to delete stored file %s: %s", file_path, exc)

    deleted = await document_store.delete_document(doc_hash)
    if deleted:
        await embedding_cache.remove_document(doc_hash)
    removed_jobs: List[str] = []
    for jid, info in list(jobs_registry.items()):
        docs = info.get("docs") or []
        if any(d.get("hash") == doc_hash for d in docs):
            removed_jobs.append(jid)
            jobs_registry.pop(jid, None)

    return {
        "hash": doc_hash,
        "deleted": bool(deleted),
        "file_removed": file_removed,
        "removed_jobs": removed_jobs,
    }


async def classify_document(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    status = str(doc.get("status") or "").strip().lower()
    if status not in settings.completed_doc_statuses:
        raise HTTPException(status_code=400, detail="Document must be processed before classification")
    extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
    if not extraction or not (extraction.get("text") or "").strip():
        raise HTTPException(status_code=400, detail="No OCR extraction available for classification")

    await document_store.update_classification_status(doc_hash, "queued", error=None)
    job_id = await queue_classification_job([doc_hash])
    return {
        "job_id": job_id,
        "status": "queued",
        "hash": doc_hash,
        "file": doc.get("original_name") or doc_hash,
        "phase": "classification",
    }


async def warmup_mineru() -> Dict[str, Any]:
    return await warmup_mineru_call()


def _active_jobs_for_doc(doc_hash: str) -> List[str]:
    return [
        jid
        for jid, info in jobs_registry.items()
        if info.get("status") in {"queued", "running"}
        and any(d.get("hash") == doc_hash for d in info.get("docs", []))
    ]


async def _ensure_doc_ready_for_postprocess(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    active_jobs = _active_jobs_for_doc(doc_hash)
    if active_jobs:
        raise HTTPException(status_code=409, detail=f"Document is currently processing (jobs: {', '.join(active_jobs)})")

    extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
    if not extraction:
        raise HTTPException(status_code=404, detail="No OCR extraction found; re-run OCR first")

    ocr_text = (extraction.get("text") or "").strip()
    if not ocr_text:
        raise HTTPException(status_code=400, detail="OCR extraction is empty; re-run OCR first")

    return doc


async def reprocess_after_ocr(doc_hash: str) -> Dict[str, Any]:
    doc = await _ensure_doc_ready_for_postprocess(doc_hash)
    phases = ["postprocess"]
    job_id = await queue_postprocess_job([doc_hash], phases=phases)
    return {
        "job_id": job_id,
        "status": "queued",
        "hash": doc_hash,
        "file": doc.get("original_name") or doc_hash,
        "phase": phases[0],
        "phases": phases,
        "queued_docs": 1,
    }


async def reprocess_all_documents() -> Dict[str, Any]:
    docs = await document_store.list_documents()
    eligible_hashes: List[str] = []
    skipped: List[Dict[str, Any]] = []

    for doc in docs:
        doc_hash = doc.get("doc_hash")
        if not doc_hash:
            skipped.append({"hash": None, "reason": "missing_hash"})
            continue
        try:
            await _ensure_doc_ready_for_postprocess(doc_hash)
            eligible_hashes.append(doc_hash)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            skipped.append({"hash": doc_hash, "reason": detail, "status_code": exc.status_code})

    if not eligible_hashes:
        return {
            "job_id": None,
            "postprocess_job_id": None,
            "classification_job_id": None,
            "status": "skipped",
            "phase": "postprocess",
            "phases": ["postprocess"],
            "queued_docs": 0,
            "total_docs": len(docs),
            "skipped": len(skipped),
            "skipped_docs": skipped,
        }

    phases = ["postprocess"]
    postprocess_job_id = await queue_postprocess_job(eligible_hashes, phases=phases)
    for doc_hash in eligible_hashes:
        await document_store.update_classification_status(doc_hash, "queued", error=None)
    classification_job_id = await queue_classification_job(eligible_hashes)
    return {
        "job_id": postprocess_job_id,
        "postprocess_job_id": postprocess_job_id,
        "classification_job_id": classification_job_id,
        "status": "queued",
        "phase": phases[0],
        "phases": phases,
        "queued_docs": len(eligible_hashes),
        "total_docs": len(docs),
        "skipped": len(skipped),
        "skipped_docs": skipped,
    }
