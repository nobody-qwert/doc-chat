"""
API routes for agentic RAG.

Provides:
- /ask/agentic - Non-streaming agentic RAG endpoint
- /ask/agentic/stream - Streaming agentic RAG endpoint
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from ..dependencies import document_store, settings, embedding_cache, gpu_phase_manager
    from ..embeddings import EmbeddingClient
    from ..services.agentic import agentic_answer, stream_agentic_answer
except ImportError:  # pragma: no cover - package import fallback
    from dependencies import document_store, settings, embedding_cache, gpu_phase_manager  # type: ignore
    from embeddings import EmbeddingClient  # type: ignore
    from services.agentic import agentic_answer, stream_agentic_answer  # type: ignore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agentic"])


class AgenticRequest(BaseModel):
    """Request for agentic RAG endpoint."""
    query: str
    conversation_id: Optional[str] = None
    max_tool_calls: int = 5


class AgenticSource(BaseModel):
    """Source reference in response."""
    doc_hash: Optional[str] = None
    chunk_id: Optional[str] = None
    document_name: str = "Unknown"
    score: float = 0.0
    text_preview: str = ""
    match_type: str = "unknown"


class AgenticStepInfo(BaseModel):
    """Step information in response."""
    name: str
    kind: str
    order: int
    duration_seconds: float = 0.0
    state: str = "done"
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgenticResponse(BaseModel):
    """Response from agentic RAG endpoint."""
    answer: str
    success: bool
    sources: List[AgenticSource] = []
    conversation_id: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    steps: List[AgenticStepInfo] = []
    total_tool_calls: int = 0
    evidence_count: int = 0
    finish_reason: Optional[str] = None


@router.post("/ask/agentic", response_model=AgenticResponse)
async def ask_agentic(req: AgenticRequest) -> AgenticResponse:
    """
    Agentic RAG endpoint.
    
    Uses an LLM agent to:
    1. Decompose the query
    2. Plan search strategy
    3. Iteratively search and review evidence
    4. Compose answer with citations
    """
    from openai import AsyncOpenAI
    
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    
    # Check if we have processed documents
    processed = await document_store.count_documents(status="processed")
    if processed == 0:
        raise HTTPException(status_code=400, detail="No processed documents yet")
    
    # Ensure LLM is ready
    await gpu_phase_manager.ensure_llm_ready()
    
    # Create clients
    if not settings.llm_base_url or not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM not configured")
    
    llm_client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    
    embedding_client = EmbeddingClient()
    
    # Run agentic pipeline
    result = await agentic_answer(
        query=query,
        document_store=document_store,
        embedding_client=embedding_client,
        embedding_cache=embedding_cache,
        llm_client=llm_client,
        settings=settings,
        conversation_id=req.conversation_id,
        max_tool_calls=req.max_tool_calls,
    )
    
    return AgenticResponse(
        answer=result.answer,
        success=result.success,
        sources=[AgenticSource(**s) for s in result.sources],
        conversation_id=result.conversation_id,
        needs_clarification=result.needs_clarification,
        clarification_message=result.clarification_message,
        steps=[AgenticStepInfo(**s) for s in result.steps],
        total_tool_calls=result.total_tool_calls,
        evidence_count=result.evidence_count,
        finish_reason=result.finish_reason,
    )


@router.post("/ask/agentic/stream")
async def ask_agentic_stream(req: AgenticRequest):
    """
    Streaming agentic RAG endpoint.
    
    Returns newline-delimited JSON with:
    - {"type": "step", "step": {...}} - Step progress updates
    - {"type": "token", "content": "..."} - Answer tokens
    - {"type": "final", ...} - Final result with metadata
    """
    from openai import AsyncOpenAI
    
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    
    # Check if we have processed documents
    processed = await document_store.count_documents(status="processed")
    if processed == 0:
        raise HTTPException(status_code=400, detail="No processed documents yet")
    
    # Ensure LLM is ready
    await gpu_phase_manager.ensure_llm_ready()
    
    # Create clients
    if not settings.llm_base_url or not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM not configured")
    
    llm_client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    
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
                max_tool_calls=req.max_tool_calls,
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
