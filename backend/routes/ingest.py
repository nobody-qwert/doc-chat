from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile

try:
    from ..services.ingestion import (
        classify_document,
        get_job_status,
        ingest_files,
        reprocess_after_ocr,
        reprocess_all_documents,
        retry_ingest,
    )
except ImportError:  # pragma: no cover
    from services.ingestion import (  # type: ignore
        classify_document,
        get_job_status,
        ingest_files,
        reprocess_after_ocr,
        reprocess_all_documents,
        retry_ingest,
    )

router = APIRouter()


@router.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file:
        uploads.append(file)
    return await ingest_files(uploads)


@router.get("/status/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    return get_job_status(job_id)


@router.post("/ingest/{doc_hash}/retry")
async def retry_ingest_route(doc_hash: str) -> Dict[str, Any]:
    return await retry_ingest(doc_hash)


@router.post("/ingest/{doc_hash}/preprocess")
async def reprocess_doc_route(doc_hash: str) -> Dict[str, Any]:
    return await reprocess_after_ocr(doc_hash)


@router.post("/ingest/reprocess_all")
async def reprocess_all_route() -> Dict[str, Any]:
    return await reprocess_all_documents()


@router.post("/ingest/{doc_hash}/classify")
async def classify_doc_route(doc_hash: str) -> Dict[str, Any]:
    return await classify_document(doc_hash)
