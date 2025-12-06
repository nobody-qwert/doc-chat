from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import httpx

try:
    from ..dependencies import settings
except ImportError:  # pragma: no cover
    from dependencies import settings  # type: ignore


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _diagnostics_url() -> str:
    base = getattr(settings, "diagnostics_url", None)
    if not base:
        raise RuntimeError("DIAGNOSTICS_URL is not configured")
    return base.rstrip("/")


async def _fetch_remote_snapshot() -> Dict[str, Any]:
    url = f"{_diagnostics_url()}/gpu"
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    if "timestamp" not in data:
        data["timestamp"] = _utc_now()
    return data


async def get_gpu_snapshot() -> Dict[str, Any]:
    try:
        return await _fetch_remote_snapshot()
    except Exception as exc:  # pragma: no cover - network error path
        return {
            "timestamp": _utc_now(),
            "gpus": [],
            "processes": [],
            "error": f"Diagnostics service unavailable: {exc}",
        }
