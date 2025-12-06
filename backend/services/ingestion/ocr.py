from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx

from .context import settings

logger = logging.getLogger(__name__)


async def call_ocr_module(
    doc_hash: str,
    doc_path: Path,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if not doc_path.exists():
        raise RuntimeError(f"OCR source file missing: {doc_path}")

    form_data = {"doc_hash": doc_hash}
    filename = doc_path.name or f"{doc_hash}.bin"
    content_type = "application/pdf" if doc_path.suffix.lower() == ".pdf" else "application/octet-stream"

    timeout = httpx.Timeout(settings.ocr_module_timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            with doc_path.open("rb") as file_obj:
                files = {"file": (filename, file_obj, content_type)}
                response = await client.post(f"{settings.ocr_module_url}/parse", data=form_data, files=files)
            response.raise_for_status()
            job_info = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"OCR module returned {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"OCR module request failed: {exc}") from exc

        job_id = job_info.get("job_id")
        if not job_id:
            raise RuntimeError("OCR module response missing job_id")
        status_url = f"{settings.ocr_module_url}/jobs/{job_id}"
        result_url = f"{status_url}/result"
        deadline = time.perf_counter() + settings.ocr_module_timeout
        poll_interval = max(0.5, float(settings.ocr_status_poll_interval))

        retry_delay = min(2.0, poll_interval)

        while True:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"OCR module job {job_id} timed out after {settings.ocr_module_timeout} seconds")
            try:
                status_resp = await client.get(status_url)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                if progress_cb:
                    progress_payload = status_data.get("progress")
                    if isinstance(progress_payload, dict):
                        progress_cb(progress_payload)
            except httpx.TransportError as exc:
                logger.warning("OCR status poll for job %s failed: %s — retrying", job_id, exc)
                await asyncio.sleep(retry_delay)
                continue
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                raise RuntimeError(f"OCR module status error {exc.response.status_code}: {detail}") from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"OCR module status request failed: {exc}") from exc

            status_value = (status_data.get("status") or "").lower()
            if status_value == "done":
                try:
                    result_resp = await client.get(result_url)
                    result_resp.raise_for_status()
                    result_data = result_resp.json()
                    if progress_cb:
                        progress_cb({"stage": "completed", "percent": 100.0})
                    return result_data
                except httpx.TransportError as exc:
                    logger.warning("OCR result fetch for job %s failed: %s — retrying", job_id, exc)
                    await asyncio.sleep(retry_delay)
                    continue
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    raise RuntimeError(f"OCR module result error {exc.response.status_code}: {detail}") from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(f"OCR module result request failed: {exc}") from exc
            if status_value == "error":
                error_detail = status_data.get("error") or "unknown error"
                raise RuntimeError(f"OCR module job {job_id} failed: {error_detail}")

            await asyncio.sleep(poll_interval)


async def warmup_mineru() -> Dict[str, Any]:
    timeout = httpx.Timeout(settings.ocr_module_timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(f"{settings.ocr_module_url}/warmup")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"MinerU warmup failed ({exc.response.status_code}): {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"MinerU warmup request failed: {exc}") from exc
