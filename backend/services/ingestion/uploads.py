from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, UploadFile

try:
    from ..utils.files import safe_filename
except ImportError:  # pragma: no cover
    from utils.files import safe_filename  # type: ignore

from .context import document_store, settings
from .models import QueuedBatchDoc


async def prepare_upload(file: UploadFile) -> Tuple[Optional[QueuedBatchDoc], Dict[str, Any]]:
    display_name = safe_filename(file.filename or "upload.bin")
    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{display_name}' is empty")

    doc_hash = hashlib.sha256(content).hexdigest()
    existing = await document_store.get_document(doc_hash)
    if existing and existing.get("status") == "processed":
        return (
            None,
            {
                "job_id": None,
                "status": "skipped",
                "file": existing.get("original_name") or display_name,
                "hash": doc_hash,
                "message": "Document already ingested",
            },
        )

    suffix = Path(display_name).suffix
    stored_name = f"{doc_hash}{suffix.lower()}" if suffix else doc_hash
    dest_path = settings.data_dir / stored_name
    dest_path.write_bytes(content)

    await document_store.upsert_document(
        doc_hash=doc_hash,
        original_name=display_name,
        stored_name=stored_name,
        size=len(content),
    )

    return (
        QueuedBatchDoc(doc_path=dest_path, doc_hash=doc_hash, display_name=display_name),
        {
            "job_id": None,
            "status": "queued",
            "file": display_name,
            "hash": doc_hash,
        },
    )
