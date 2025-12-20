from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HistoryMessage:
    role: str
    content: str


class InMemoryHistoryStore:
    """Simple in-memory queue for single-user chat history."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._messages: List[HistoryMessage] = []

    async def clear(self) -> None:
        async with self._lock:
            self._messages.clear()

    async def get_messages(self) -> List[Dict[str, str]]:
        async with self._lock:
            return [{"role": msg.role, "content": msg.content} for msg in self._messages]

    async def append_message(self, role: str, content: str) -> None:
        normalized_role = (role or "").strip()
        normalized_content = (content or "").strip()
        if not normalized_role or not normalized_content:
            return
        async with self._lock:
            self._messages.append(HistoryMessage(normalized_role, normalized_content))

    async def append_turn(self, user_message: str, assistant_message: str) -> None:
        async with self._lock:
            if user_message:
                self._messages.append(HistoryMessage("user", user_message))
            if assistant_message:
                self._messages.append(HistoryMessage("assistant", assistant_message))
