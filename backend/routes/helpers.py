from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from ..dependencies import settings
except ImportError:  # pragma: no cover
    from dependencies import settings  # type: ignore


def format_document_row(doc: Dict[str, Any]) -> Dict[str, Any]:
    stored_name = doc.get("stored_name") or doc.get("doc_hash")
    file_path = settings.data_dir / stored_name
    size = doc.get("size")
    try:
        if (size in (None, 0)) and file_path.exists():
            size = file_path.stat().st_size
    except Exception:  # pragma: no cover - best effort
        size = size or 0
    return {
        "name": doc.get("original_name") or stored_name,
        "size": size or 0,
        "status": doc.get("status") or "unknown",
        "hash": doc.get("doc_hash"),
        "stored_name": stored_name,
        "path": str(settings.data_dir / stored_name),
        "last_ingested_at": doc.get("last_ingested_at"),
        "error": doc.get("error"),
        "updated_at": doc.get("updated_at"),
        "created_at": doc.get("created_at"),
    }
