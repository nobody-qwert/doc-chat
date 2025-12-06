from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

try:
    from ..chunking import count_text_tokens
    from ..dependencies import document_store, settings
    from .helpers import format_document_row
    from ..services.ingestion import delete_document as delete_document_service
except ImportError:  # pragma: no cover
    from chunking import count_text_tokens  # type: ignore
    from dependencies import document_store, settings  # type: ignore
    from routes.helpers import format_document_row  # type: ignore
    from services.ingestion import delete_document as delete_document_service  # type: ignore

router = APIRouter()


def _parse_meta(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text: Optional[str]
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


@router.get("/documents")
async def list_documents() -> List[Dict[str, Any]]:
    docs = await document_store.list_documents()
    results: List[Dict[str, Any]] = []
    for doc in docs:
        doc_data = format_document_row(doc)
        doc_hash = doc.get("doc_hash")
        status = str(doc_data.get("status") or "").strip().lower()

        ocr_available = False
        ocr_created_at: Optional[str] = None
        small_embeddings = 0
        total_embeddings = 0

        if doc_hash:
            extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
            if extraction:
                ocr_available = True
                ocr_created_at = extraction.get("created_at")

            embedding_counts = await document_store.count_embeddings_by_config(doc_hash)
            small_embeddings = int(embedding_counts.get(settings.chunk_config_small_id, 0))
            total_embeddings = int(sum(embedding_counts.values()))

        classification = None
        if doc_hash:
            classification = await document_store.get_classification(doc_hash)

        doc_data["ocr_available"] = ocr_available
        doc_data["ocr_extracted_at"] = ocr_created_at
        doc_data["small_embeddings"] = small_embeddings
        doc_data["total_embeddings"] = total_embeddings
        doc_data["embedding_available"] = total_embeddings > 0
        doc_data["classification"] = classification
        doc_data["classification_status"] = doc.get("classification_status") or doc_data.get("classification_status") or "pending"
        doc_data["classification_error"] = doc.get("classification_error")
        doc_data["last_classified_at"] = doc.get("last_classified_at")

        if doc_hash and status in settings.completed_doc_statuses:
            metrics = await document_store.get_performance_metrics(doc_hash)
            doc_data["performance"] = metrics

        results.append(doc_data)
    return results


@router.delete("/documents/{doc_hash}")
async def delete_document(doc_hash: str) -> Dict[str, Any]:
    return await delete_document_service(doc_hash)


@router.get("/parsers")
async def list_parsers() -> Dict[str, Any]:
    return {"default": settings.ocr_parser_key, "options": [settings.ocr_parser_key]}


@router.get("/debug/parsed_text/{doc_hash}")
async def debug_parsed_text(
    doc_hash: str,
    parser: Optional[str] = None,
    max_chars: Optional[int] = 2000,
) -> Dict[str, Any]:
    parser_key = parser or settings.ocr_parser_key
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    row = await document_store.get_extraction(doc_hash, parser_key)
    if not row:
        raise HTTPException(status_code=404, detail=f"No extraction found for parser '{parser_key}'")
    text = row["text"] or ""
    total_tokens = count_text_tokens(text)
    embedding_counts = await document_store.count_embeddings_by_config(doc_hash)
    small_embeddings = int(embedding_counts.get(settings.chunk_config_small_id, 0))
    total_embeddings = small_embeddings

    text_length = len(text)
    unlimited_preview = max_chars is None or max_chars <= 0
    preview_limit = text_length if unlimited_preview else int(max_chars)
    preview_chars = min(preview_limit, text_length)

    meta = _parse_meta(row.get("meta"))
    asset_info = meta.get("assets") if isinstance(meta, dict) else {}
    assets_payload: Optional[Dict[str, Any]] = None
    if isinstance(asset_info, dict) and asset_info.get("base_dir"):
        base_url = f"/api/documents/{doc_hash}/assets/"
        images = asset_info.get("image_files") or []
        content_lists = asset_info.get("content_lists") or []
        assets_payload = {
            "base_url": base_url,
            "images": images,
            "content_lists": content_lists,
        }

    return {
        "parser": parser_key,
        "document_name": doc.get("original_name", "unknown"),
        "file_size": doc.get("size", 0),
        "extracted_chars": len(text),
        "total_tokens": total_tokens,
        "preview_chars": preview_chars,
        "truncated": (not unlimited_preview) and text_length > preview_limit,
        "preview": text[:preview_chars],
        "chunk_count": total_embeddings,
        "total_embeddings": total_embeddings,
        "small_embeddings": small_embeddings,
        "assets": assets_payload,
        "image_blocks": meta.get("image_blocks") if isinstance(meta, dict) else None,
    }


@router.get("/documents/{doc_hash}/assets/{asset_path:path}")
async def get_document_asset(doc_hash: str, asset_path: str):
    if not asset_path:
        raise HTTPException(status_code=400, detail="Asset path required")
    row = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    meta = _parse_meta(row.get("meta"))
    assets = meta.get("assets") if isinstance(meta, dict) else None
    base_dir_raw = assets.get("base_dir") if isinstance(assets, dict) else None
    if not base_dir_raw:
        markdown_path = meta.get("markdown_path") if isinstance(meta, dict) else None
        if markdown_path:
            base_dir_raw = str(Path(markdown_path).parent)
    if not base_dir_raw:
        raise HTTPException(status_code=404, detail="No assets available for this document")

    base_dir = Path(base_dir_raw).resolve()
    allowed_root = settings.index_dir.resolve()
    try:
        base_dir.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Asset location is invalid")

    target_path = (base_dir / asset_path.lstrip("/")).resolve()
    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")

    return FileResponse(target_path)
