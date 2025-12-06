from __future__ import annotations

from typing import Iterable, Optional

try:
    from .tokenizer_registry import (
        count_chat_tokens,
        count_text_tokens,
        record_tokenizer_fallback,
        truncate_text,
    )
except ImportError:  # pragma: no cover
    from tokenizer_registry import (  # type: ignore
        count_chat_tokens,
        count_text_tokens,
        record_tokenizer_fallback,
        truncate_text,
    )

def estimate_tokens(
    text: str,
    *,
    tokenizer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    identifier = (tokenizer_id or model or "").strip()
    if not identifier:
        raise ValueError("tokenizer_id or model must be provided for token estimation")
    token_count = count_text_tokens(text or "", identifier)
    if token_count is None:
        raise RuntimeError(f"Tokenizer {identifier} unavailable for token estimation")
    return token_count


def estimate_messages_tokens(
    messages: Iterable[dict],
    *,
    tokenizer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    identifier = (tokenizer_id or model or "").strip()
    if not identifier:
        raise ValueError("tokenizer_id or model must be provided for message token estimation")
    cached_messages = list(messages)
    token_count = count_chat_tokens(cached_messages, identifier)
    if token_count is None:
        raise RuntimeError(f"Tokenizer {identifier} unavailable for message token estimation")
    return token_count


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    tokenizer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    identifier = (tokenizer_id or model or "").strip()
    if identifier:
        truncated = truncate_text(text, max_tokens, identifier)
        if truncated is not None:
            return truncated
        record_tokenizer_fallback(identifier, "truncate_fallback")
    approx_chars = max_tokens * 4
    return text[:approx_chars]
