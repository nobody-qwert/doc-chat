from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from ..token_utils import estimate_messages_tokens
except ImportError:  # pragma: no cover
    from token_utils import estimate_messages_tokens  # type: ignore


def summarize_history(messages: List[Dict[str, str]], *, max_entries: int = 6, max_chars: int = 800) -> str:
    if not messages:
        return ""
    recent = messages[-max_entries:]
    parts: List[str] = []
    for entry in recent:
        role = (entry.get("role") or "user").strip().lower()
        label = "User" if role == "user" else "Assistant"
        content = (entry.get("content") or "").strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:197] + "..."
        parts.append(f"{label}: {content}")
    summary = " | ".join(parts)
    return summary[:max_chars]


def build_distilled_query(history_summary: str, question: str) -> str:
    question = question.strip()
    if not history_summary:
        return question
    return f"Conversation summary: {history_summary}\nFocus question: {question}"


def trim_history_for_budget(
    history: List[Dict[str, str]],
    *,
    tokenizer_id: str,
    token_limit: int,
    system_prompt: str,
) -> Tuple[List[Dict[str, str]], bool, int]:
    trimmed = list(history)
    history_truncated = False
    while trimmed:
        total = estimate_messages_tokens(
            [{"role": "system", "content": system_prompt}, *trimmed],
            tokenizer_id=tokenizer_id,
        )
        if total <= token_limit:
            return trimmed, history_truncated, total
        trimmed.pop(0)
        history_truncated = True

    total = estimate_messages_tokens(
        [{"role": "system", "content": system_prompt}],
        tokenizer_id=tokenizer_id,
    )
    return [], history_truncated, total


def render_context(sections: List[Dict[str, Any]]) -> str:
    if not sections:
        return ""
    blocks: List[str] = []
    for idx, entry in enumerate(sections, start=1):
        chunk = entry["chunk"]
        text = chunk.get("text") or ""
        blocks.append(f"[source {idx}]\n{text}")
    return "\n\n".join(blocks)
