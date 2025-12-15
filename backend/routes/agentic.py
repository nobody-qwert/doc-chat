"""
Streaming API route for agentic RAG.

Provides:
- /ask/agentic/stream - Streaming agentic RAG endpoint
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from ..dependencies import document_store, settings, embedding_cache, gpu_phase_manager
    from ..embeddings import EmbeddingClient
    from ..services.agentic import stream_agentic_answer
except ImportError:  # pragma: no cover - package import fallback
    from dependencies import document_store, settings, embedding_cache, gpu_phase_manager  # type: ignore
    from embeddings import EmbeddingClient  # type: ignore
    from services.agentic import stream_agentic_answer  # type: ignore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agentic"])


class AgenticRequest(BaseModel):
    """Request for agentic RAG endpoint."""
    query: str
    conversation_id: Optional[str] = None
    max_subqueries: Optional[int] = None


@router.post("/ask/agentic/stream")
async def ask_agentic_stream(req: AgenticRequest):
    """
    Streaming agentic RAG endpoint.
    
    Returns newline-delimited JSON with:
    - {"type": "step", "step": {...}} - Step progress updates
    - {"type": "token", "content": "..."} - Answer tokens
    - {"type": "final", ...} - Final result with metadata
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    max_subqueries = req.max_subqueries or settings.agentic_max_subqueries
    if max_subqueries < 1:
        max_subqueries = settings.agentic_max_subqueries
    
    # Check if we have processed documents
    processed = await document_store.count_documents(status="processed")
    if processed == 0:
        raise HTTPException(status_code=400, detail="No processed documents yet")
    
    # Ensure LLM is ready
    await gpu_phase_manager.ensure_llm_ready()
    
    # Create clients
    if not settings.llm_base_url:
        raise HTTPException(status_code=500, detail="LLM not configured")

    from openai import AsyncOpenAI

    llm_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key="local")
    
    embedding_client = EmbeddingClient()
    
    # Create streaming generator
    async def event_stream():
        try:
            generator = stream_agentic_answer(
                query=query,
                document_store=document_store,
                embedding_client=embedding_client,
                embedding_cache=embedding_cache,
                llm_client=llm_client,
                settings=settings,
                conversation_id=req.conversation_id,
                max_subqueries=max_subqueries,
            )
            async for chunk in generator:
                yield chunk
        except Exception as e:
            logger.exception(f"Agentic streaming failed: {e}")
            import json
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )
