from __future__ import annotations

from typing import List


def safe_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in (".", "_", "-", " "))
    cleaned = cleaned.strip()
    return cleaned[:255] or "upload.bin"


def unique_ordered(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        key = (value or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered
