"""
API routes for metadata extraction.

Provides endpoints to:
- List available extraction schemas
- Trigger extraction for documents
- Query extracted metadata
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/extraction", tags=["extraction"])

try:
    from ..dependencies import document_store, settings, gpu_phase_manager
    from ..services.extraction.schemas import get_schema, list_schemas
    from ..services.extraction.persistence import MetadataStore
    from ..services.extraction.engine import extract_metadata_batch
except ImportError:  # pragma: no cover - script execution fallback
    from dependencies import document_store, settings, gpu_phase_manager  # type: ignore
    from services.extraction.schemas import get_schema, list_schemas  # type: ignore
    from services.extraction.persistence import MetadataStore  # type: ignore
    from services.extraction.engine import extract_metadata_batch  # type: ignore


class ExtractRequest(BaseModel):
    """Request to extract metadata from documents."""
    schema_id: str
    doc_hashes: Optional[List[str]] = None  # If None, extract all matching documents
    force: bool = False  # Re-extract even if already extracted


class ExtractResponse(BaseModel):
    """Response from extraction job."""
    schema_id: str
    total_documents: int
    successful: int
    failed: int
    skipped: int = 0
    extracted: int = 0  # Backwards-compatible alias for successful
    errors: int = 0     # Backwards-compatible alias for failed
    results: List[Dict[str, Any]]


class MetadataQueryRequest(BaseModel):
    """Request to query extracted metadata."""
    schema_id: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 100


@router.get("/schemas")
async def list_extraction_schemas() -> List[Dict[str, Any]]:
    """List all available extraction schemas."""
    return list_schemas()


@router.get("/schemas/{schema_id}")
async def get_extraction_schema(schema_id: str) -> Dict[str, Any]:
    """Get details of a specific extraction schema."""
    schema = get_schema(schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found")
    
    return {
        "schema_id": schema.schema_id,
        "display_name": schema.display_name,
        "description": schema.description,
        "target_l1_category": schema.target_l1_category,
        "target_l2_category": schema.target_l2_category,
        "table_name": schema.table_name,
        "fields": [
            {
                "name": f.name,
                "type": f.sql_type,
                "description": f.description,
                "required": f.required,
                "examples": f.examples,
            }
            for f in schema.fields
        ],
    }


@router.post("/extract", response_model=ExtractResponse)
async def extract_metadata(request: ExtractRequest) -> ExtractResponse:
    """
    Trigger metadata extraction for documents.
    
    If doc_hashes is not provided, extracts from all documents
    matching the schema's target category that haven't been extracted yet.
    """
    from openai import AsyncOpenAI
    
    schema = get_schema(request.schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{request.schema_id}' not found")
    
    # Initialize metadata store
    meta_store = MetadataStore(str(settings.doc_store_path))
    await meta_store.ensure_table_exists(schema)
    
    # Get documents to extract
    if request.doc_hashes:
        doc_hashes = request.doc_hashes
    else:
        # Get all processed documents with matching classification
        all_docs = await document_store.list_documents()
        doc_hashes = []
        
        for doc in all_docs:
            if doc.get("status") != "processed":
                continue
            
            # Check classification matches schema target
            classification = await document_store.get_classification(doc["doc_hash"])
            if not classification:
                continue
            
            l1_match = classification.get("l1_id") == schema.target_l1_category
            l2_match = (
                schema.target_l2_category is None or
                classification.get("l2_id") == schema.target_l2_category
            )
            
            if l1_match and l2_match:
                doc_hashes.append(doc["doc_hash"])
    
    # Deduplicate while preserving order
    seen_hashes: set[str] = set()
    unique_doc_hashes: List[str] = []
    for doc_hash in doc_hashes:
        if doc_hash and doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_doc_hashes.append(doc_hash)
    doc_hashes = unique_doc_hashes
    initial_doc_count = len(doc_hashes)
    skipped_count = 0
    
    if not doc_hashes:
        return ExtractResponse(
            schema_id=request.schema_id,
            total_documents=0,
            successful=0,
            failed=0,
            skipped=0,
            extracted=0,
            errors=0,
            results=[],
        )
    
    # Filter out already extracted (unless force=True)
    if not request.force:
        already_extracted = set(await meta_store.list_extracted_doc_hashes(schema))
        doc_hashes = [h for h in doc_hashes if h not in already_extracted]
        skipped_count = initial_doc_count - len(doc_hashes)
    
    if not doc_hashes:
        message = (
            "All matching documents already extracted"
            if skipped_count > 0 else "No matching documents found"
        )
        return ExtractResponse(
            schema_id=request.schema_id,
            total_documents=0,
            successful=0,
            failed=0,
            skipped=skipped_count,
            extracted=0,
            errors=0,
            results=[{"message": message}],
        )
    
    # Get document texts
    documents = []
    missing_text_hashes: List[str] = []
    for doc_hash in doc_hashes:
        extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
        if extraction and extraction.get("text"):
            documents.append({
                "doc_hash": doc_hash,
                "text": extraction["text"],
            })
        else:
            missing_text_hashes.append(doc_hash)
    
    llm_results = []
    if documents:
        # Ensure LLM is ready
        await gpu_phase_manager.ensure_llm_ready()
        
        if not settings.llm_base_url or not settings.llm_api_key:
            raise HTTPException(status_code=500, detail="LLM not configured")
        
        client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        
        llm_results = await extract_metadata_batch(
            documents=documents,
            schema=schema,
            llm_client=client,
            model=settings.llm_model,
            temperature=0.1,
        )
    
    # Save successful extractions and collect errors
    successful = 0
    failed = 0
    result_details: List[Dict[str, Any]] = []
    
    for missing in missing_text_hashes:
        failed += 1
        result_details.append({
            "doc_hash": missing,
            "success": False,
            "error": "No document text found",
        })
    
    for result in llm_results:
        if result.success and result.data:
            await meta_store.upsert_metadata(
                schema=schema,
                doc_hash=result.doc_hash,
                data=result.data,
                model=settings.llm_model,
            )
            successful += 1
            result_details.append({
                "doc_hash": result.doc_hash,
                "success": True,
                "data": result.data,
            })
        else:
            failed += 1
            result_details.append({
                "doc_hash": result.doc_hash,
                "success": False,
                "error": result.error,
            })
    
    return ExtractResponse(
        schema_id=request.schema_id,
        total_documents=len(doc_hashes),
        successful=successful,
        failed=failed,
        skipped=skipped_count,
        extracted=successful,
        errors=failed,
        results=result_details,
    )


@router.post("/query")
async def query_metadata(request: MetadataQueryRequest) -> Dict[str, Any]:
    """
    Query extracted metadata with optional filters.
    
    Filter examples:
    - Exact match: {"vendor_name": "ACME Corp"}
    - Comparison: {"total_amount": {">": 1000}}
    - Like: {"vendor_name": {"like": "%Corp%"}}
    """
    schema = get_schema(request.schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{request.schema_id}' not found")
    
    meta_store = MetadataStore(str(settings.doc_store_path))
    
    results = await meta_store.query_metadata(
        schema=schema,
        filters=request.filters,
        limit=request.limit,
    )
    
    return {
        "schema_id": request.schema_id,
        "count": len(results),
        "results": results,
    }


@router.get("/metadata/{schema_id}/{doc_hash}")
async def get_document_metadata(schema_id: str, doc_hash: str) -> Dict[str, Any]:
    """Get extracted metadata for a specific document."""
    schema = get_schema(schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found")
    
    meta_store = MetadataStore(str(settings.doc_store_path))
    metadata = await meta_store.get_metadata(schema, doc_hash)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"No metadata found for document '{doc_hash}'")
    
    return metadata


@router.get("/stats/{schema_id}")
async def get_extraction_stats(schema_id: str) -> Dict[str, Any]:
    """Get extraction statistics for a schema."""
    schema = get_schema(schema_id)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found")
    
    meta_store = MetadataStore(str(settings.doc_store_path))
    
    # Gather extracted hashes for UI to highlight completed documents
    extracted_doc_hashes = await meta_store.list_extracted_doc_hashes(schema)
    extracted_count = len(extracted_doc_hashes)
    
    # Count total matching documents
    all_docs = await document_store.list_documents()
    matching_count = 0
    
    for doc in all_docs:
        if doc.get("status") != "processed":
            continue
        
        classification = await document_store.get_classification(doc["doc_hash"])
        if not classification:
            continue
        
        l1_match = classification.get("l1_id") == schema.target_l1_category
        l2_match = (
            schema.target_l2_category is None or
            classification.get("l2_id") == schema.target_l2_category
        )
        
        if l1_match and l2_match:
            matching_count += 1
    
    pending_documents = max(0, matching_count - extracted_count)
    
    return {
        "schema_id": schema_id,
        "display_name": schema.display_name,
        "total_matching_documents": matching_count,
        "extracted_documents": extracted_count,
        "pending_documents": pending_documents,
        "coverage_percent": (
            round(100 * extracted_count / matching_count, 1)
            if matching_count > 0 else 0
        ),
        "extracted_docs": extracted_doc_hashes,
    }
