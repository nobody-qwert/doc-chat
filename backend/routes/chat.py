from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

try:
    from ..schemas import AskRequest, AskResponse
    from ..services.rag import ask_question, stream_question
except ImportError:  # pragma: no cover - running as top-level modules
    from schemas import AskRequest, AskResponse  # type: ignore
    from services.rag import ask_question, stream_question  # type: ignore

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    return await ask_question(req)


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Stream LLM tokens back to the client as newline-delimited JSON.
    """
    stream = await stream_question(req)
    return StreamingResponse(stream, media_type="application/x-ndjson")
