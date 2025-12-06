from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    top_k: int = Field(5, ge=1, le=50)
    conversation_id: Optional[str] = Field(default=None)
    continue_last: bool = Field(default=False)


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_sources: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_id: str
    context_tokens_used: int
    context_window_limit: int
    context_usage: float
    context_truncated: bool
    finish_reason: Optional[str] = Field(default=None)
    needs_follow_up: bool = Field(default=False)
    time_to_first_token_seconds: Optional[float] = Field(default=None)
    generation_seconds: Optional[float] = Field(default=None)
    tokens_per_second: Optional[float] = Field(default=None)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
