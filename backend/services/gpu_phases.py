from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class GPUPhaseManager:
    """Coordinate which heavy GPU service (LLM or OCR) owns memory."""

    def __init__(
        self,
        llm_control_url: Optional[str],
        ocr_control_url: Optional[str],
        *,
        llm_inference_url: Optional[str] = None,
        timeout: float = 30.0,
        ready_check_timeout: float = 180.0,
    ) -> None:
        self._llm_url = (llm_control_url or "").strip() or None
        self._ocr_url = (ocr_control_url or "").strip() or None
        self._timeout = max(1.0, float(timeout))
        self._ready_timeout = max(1.0, float(ready_check_timeout))
        self._lock = asyncio.Lock()
        self._state: str = "llm"
        self._last_error: Optional[str] = None
        self._llm_inference_target = self._parse_inference_target(llm_inference_url)

    async def switch_to_no_llm(self, reason: Optional[str] = None) -> None:
        await self._transition(target="no_llm", reason=reason)

    async def switch_to_llm(self, reason: Optional[str] = None) -> None:
        await self._transition(target="llm", reason=reason)

    async def ensure_llm_ready(self) -> None:
        await self.switch_to_llm(reason="ensure_llm_ready")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "llm_control_url": self._llm_url,
            "ocr_control_url": self._ocr_url,
            "last_error": self._last_error,
        }

    async def _transition(self, target: str, reason: Optional[str]) -> None:
        if target not in {"llm", "no_llm"}:
            raise ValueError(f"Unknown GPU phase target: {target}")
        urls_missing = self._llm_url is None or self._ocr_url is None
        if urls_missing:
            # No-op when control URLs are not configured
            self._state = target
            if target == "llm":
                await self._wait_for_llm_ready()
            return
        async with self._lock:
            if self._state == target:
                # Refresh to ensure service is still alive
                await self._refresh_current(target)
                return
            logger.info("Switching GPU phase to %s (reason=%s)", target, reason)
            try:
                if target == "no_llm":
                    await self._call_control(self._llm_url, "unload")
                    await self._call_control(self._ocr_url, "load")
                else:
                    await self._call_control(self._llm_url, "load")
                    await self._wait_for_llm_ready()
                self._state = target
                self._last_error = None
            except Exception as exc:  # pragma: no cover - network errors
                self._last_error = str(exc)
                logger.error("GPU phase transition to %s failed: %s", target, exc)
                raise

    async def _refresh_current(self, target: str) -> None:
        try:
            if target == "no_llm" and self._ocr_url:
                await self._call_control(self._ocr_url, "status", method="GET", ignore_missing=True)
            elif target == "llm" and self._llm_url:
                status = await self._call_control(self._llm_url, "status", method="GET", ignore_missing=True)
                if isinstance(status, dict) and not status.get("running"):
                    logger.warning("LLM controller reports process not running; reloading to satisfy current phase")
                    await self._call_control(self._llm_url, "load")
                    await self._wait_for_llm_ready()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("GPU phase refresh failed (%s): %s", target, exc)

    async def _call_control(
        self,
        base_url: str,
        action: str,
        *,
        method: str = "POST",
        ignore_missing: bool = False,
    ) -> Optional[Any]:
        url = f"{base_url.rstrip('/')}/{action.strip('/')}"
        timeout = httpx.Timeout(self._timeout, connect=min(5.0, self._timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                request = client.get if method.upper() == "GET" else client.post
                resp = await request(url)
                resp.raise_for_status()
                if resp.headers.get("content-type", "").lower().startswith("application/json"):
                    with contextlib.suppress(Exception):
                        return resp.json()
                return None
            except Exception:
                if ignore_missing:
                    raise
                raise

    @staticmethod
    def _parse_inference_target(url: Optional[str]) -> Optional[Tuple[str, int]]:
        if not url:
            return None
        cleaned = url.strip()
        if not cleaned:
            return None
        parsed = urlparse(cleaned if "://" in cleaned else f"http://{cleaned}")
        host = parsed.hostname
        if not host:
            return None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return host, port

    async def _wait_for_llm_ready(self) -> None:
        if not self._llm_inference_target:
            return
        host, port = self._llm_inference_target
        logger.info("Waiting for LLM endpoint %s:%s to become ready", host, port)
        deadline = time.perf_counter() + self._ready_timeout
        last_error: Optional[Exception] = None
        while time.perf_counter() < deadline:
            try:
                connect = asyncio.open_connection(host, port)
                reader, writer = await asyncio.wait_for(connect, timeout=min(5.0, self._timeout))
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                logger.info("LLM endpoint %s:%s is ready", host, port)
                return
            except Exception as exc:  # pragma: no cover - best effort wait
                last_error = exc
                await asyncio.sleep(1.0)
        message = f"LLM endpoint {host}:{port} did not become ready within {self._ready_timeout:.1f}s"
        logger.error(message)
        if last_error:
            raise RuntimeError(f"{message}: {last_error}") from last_error
        raise RuntimeError(message)
