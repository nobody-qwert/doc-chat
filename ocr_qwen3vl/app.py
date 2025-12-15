import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from pdf_render import render_pdf_to_png_pages
from qwen_vl_runtime import QwenVLServer, load_qwen_vl_settings, qwen_vl_ocr_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable {name}")
    cleaned = str(value).strip()
    if not cleaned:
        raise RuntimeError(f"Environment variable {name} cannot be empty")
    return cleaned


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    cleaned = str(raw).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    cleaned = str(raw).strip()
    if not cleaned:
        return default
    try:
        return int(cleaned)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer (got {cleaned!r})") from exc


def _unescape_env(value: str) -> str:
    """Interpret common escape sequences in env-file strings (e.g. \\n)."""
    try:
        return value.encode("utf-8").decode("unicode_escape")
    except Exception:
        return value


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that wrap the entire output.
    
    Qwen VL sometimes wraps its output in code fences like:
    ```markdown
    actual content here
    ```
    
    This breaks ReactMarkdown rendering, so we strip them.
    """
    if not text:
        return text
    stripped = text.strip()
    
    # Pattern to match opening code fence with optional language hint
    # Matches: ```markdown, ```md, ```, etc. at the start
    opening_match = re.match(r'^```(?:markdown|md)?\s*\n', stripped)
    if not opening_match:
        return text
    
    # Check for closing fence at the end
    closing_match = re.search(r'\n```\s*$', stripped)
    if not closing_match:
        return text
    
    # Remove the fences
    content_start = opening_match.end()
    content_end = closing_match.start()
    inner = stripped[content_start:content_end]
    
    return inner.strip()


def _strip_image_syntax(text: str) -> str:
    """Remove markdown image syntax like ![alt](url).
    
    Qwen VL sometimes hallucinates image URLs (e.g., imgur.com links) when it sees
    icons or diagrams in the PDF. These don't exist and cause broken image placeholders
    in the frontend. We strip them entirely.
    """
    if not text:
        return text
    # Match markdown image syntax: ![optional alt text](url)
    # This handles: ![](url), ![alt](url), ![alt text with spaces](url)
    cleaned = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
    # Also remove any leftover empty lines from removed images
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()


DATA_DIR = Path(_require_env("DATA_DIR"))
INDEX_DIR = Path(_require_env("INDEX_DIR"))
OCR_OUTPUT_DIR = Path(_require_env("OCR_OUTPUT_DIR"))
OCR_INCOMING_DIR = Path(_require_env("OCR_INCOMING_DIR"))
OCR_WARMUP_DIR = Path(_require_env("OCR_WARMUP_DIR"))
for path in (DATA_DIR, INDEX_DIR, OCR_OUTPUT_DIR, OCR_INCOMING_DIR, OCR_WARMUP_DIR):
    path.mkdir(parents=True, exist_ok=True)

RENDER_DPI = _int_env("QWEN_VL_RENDER_DPI", 200)
RENDER_MAX_DIM = _int_env("QWEN_VL_RENDER_MAX_DIM", 1800)
SAVE_IMAGES = _env_bool("QWEN_VL_SAVE_IMAGES", False)
WARMUP_ON_STARTUP = _env_bool("QWEN_VL_WARMUP_ON_STARTUP", False)
PAGE_SEPARATOR = _unescape_env(os.environ.get("QWEN_VL_PAGE_SEPARATOR", "\n\n---\n\n"))

vl_settings = load_qwen_vl_settings()
vl_server = QwenVLServer(vl_settings)


@dataclass
class OCRJob:
    job_id: str
    doc_hash: str
    filename: str
    pdf_path: Optional[Path]
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=lambda: {"stage": "queued", "percent": 0.0})

    def set_status(self, status: str, *, error: Optional[str] = None) -> None:
        self.status = status
        if error is not None:
            self.error = error
        self.updated_at = _utc_now()

    def update_progress(self, stage: str, percent: float, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "stage": stage,
            "percent": max(0.0, min(100.0, float(percent))),
        }
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        self.progress = payload
        self.updated_at = _utc_now()

    def as_status_payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "doc_hash": self.doc_hash,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self.progress,
        }


jobs: Dict[str, OCRJob] = {}
_app_lock = asyncio.Lock()


class ParseResponse(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobQueuedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    doc_hash: str
    filename: str
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


def _job_or_404(job_id: str) -> OCRJob:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _persist_upload(doc_hash: str, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".bin"
    token = uuid4().hex
    dest_path = OCR_INCOMING_DIR / f"{doc_hash}_{token}{suffix}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest_path.open("wb") as out_file:
            while True:
                chunk = await upload.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
    except Exception:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return dest_path


def _running_jobs() -> int:
    return sum(1 for job in jobs.values() if job.status in {"queued", "running"})


async def _ensure_loaded() -> None:
    if not vl_settings.enabled:
        return
    status = vl_server.status()
    if status.get("running"):
        return
    await vl_server.start("ensure_loaded")


async def _warmup() -> Dict[str, Any]:
    await _ensure_loaded()
    return {"state": "loaded", "server": vl_server.status()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if WARMUP_ON_STARTUP:
        async def _do_warmup() -> None:
            try:
                info = await _warmup()
                logger.info("Qwen3-VL warmup finished: %s", info.get("server"))
            except Exception as exc:
                logger.warning("Qwen3-VL warmup failed: %s", exc)

        asyncio.create_task(_do_warmup())
    yield


app = FastAPI(title="Qwen3-VL OCR Module", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/parse", response_model=JobQueuedResponse)
async def parse_document(
    doc_hash: str = Form(...),
    file: UploadFile = File(...),
    parse_method: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    table_enable: Optional[bool] = Form(None),
    formula_enable: Optional[bool] = Form(None),
) -> JobQueuedResponse:
    try:
        pdf_path = await _persist_upload(doc_hash, file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {exc}") from exc

    job_id = uuid4().hex
    job = OCRJob(
        job_id=job_id,
        doc_hash=doc_hash,
        filename=file.filename or f"{doc_hash}.pdf",
        pdf_path=pdf_path,
    )
    job.update_progress("queued", 0.0)
    jobs[job_id] = job

    job_options = {
        "parse_method": parse_method,
        "lang": lang,
        "table_enable": table_enable,
        "formula_enable": formula_enable,
    }
    asyncio.create_task(_process_job(job, options=job_options))
    return JobQueuedResponse(job_id=job.job_id, status=job.status)


async def _process_job(job: OCRJob, *, options: Dict[str, Any]) -> None:
    out_dir = (OCR_OUTPUT_DIR / job.doc_hash / "qwen3_vl").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()
    try:
        job.set_status("running")
        job.started_at = _utc_now()
        job.update_progress("initializing", 5.0)

        await _ensure_loaded()
        job.update_progress("rendering", 10.0)

        if not job.pdf_path:
            raise RuntimeError("Missing PDF path")
        rendered_pages, render_cfg = await asyncio.to_thread(
            render_pdf_to_png_pages,
            job.pdf_path,
            dpi=RENDER_DPI,
            max_dim=RENDER_MAX_DIM,
        )
        total_pages = len(rendered_pages)
        if total_pages == 0:
            raise RuntimeError("PDF rendered zero pages")

        image_files: list[str] = []
        image_blocks: list[dict[str, Any]] = []

        page_texts: list[str] = []
        per_page: list[dict[str, Any]] = []
        for idx, page in enumerate(rendered_pages, start=1):
            png_name = f"page_{idx:04}.png"
            rel_path = str(Path("pages") / png_name)
            if SAVE_IMAGES:
                (pages_dir / png_name).write_bytes(page.png_bytes)
                image_files.append(rel_path)
            image_blocks.append({"page": idx, "path": rel_path if SAVE_IMAGES else None, "width": page.size[0], "height": page.size[1]})

            page_start = time.perf_counter()
            job.update_progress("ocr", 10.0 + (idx - 1) / total_pages * 85.0, current=idx, total=total_pages)

            prompt_hint = ""
            if options.get("lang"):
                prompt_hint += f"\nLanguage hint: {options['lang']}\n"
            if options.get("table_enable") is False:
                prompt_hint += "\nDo not attempt to format tables.\n"
            if options.get("formula_enable") is False:
                prompt_hint += "\nDo not attempt to output LaTeX for formulas.\n"
            prompt = (vl_settings.prompt + prompt_hint).strip()

            text = await qwen_vl_ocr_image(settings=vl_settings, image_bytes=page.png_bytes, prompt=prompt)
            # Strip code fences that Qwen VL sometimes wraps around markdown output
            # Strip fake image URLs that Qwen VL hallucinates (e.g., imgur.com links)
            cleaned = _strip_image_syntax(_strip_code_fences((text or "").strip()))
            page_texts.append(cleaned)
            per_page.append({
                "page": idx,
                "seconds": max(0.0, time.perf_counter() - page_start),
                "chars": len(cleaned),
            })

        combined = PAGE_SEPARATOR.join(f"# Page {i}\n\n{t}".rstrip() for i, t in enumerate(page_texts, start=1)).strip()
        if not combined:
            raise RuntimeError("Qwen3-VL returned empty text")

        job.text = combined
        job.metadata = {
            "provider": "qwen3_vl",
            "render": {"dpi": render_cfg.dpi, "max_dim": render_cfg.max_dim},
            "pages": total_pages,
            "timing": {"total_sec": max(0.0, time.perf_counter() - start_total), "per_page": per_page},
            "assets": {
                "base_dir": str(out_dir),
                "image_files": image_files,
                "content_lists": [],
            },
            "image_blocks": image_blocks,
            "server": vl_server.status(),
        }
        job.finished_at = _utc_now()
        job.set_status("done")
        job.update_progress("completed", 100.0)
    except Exception as exc:
        job.error = str(exc)
        job.set_status("error", error=str(exc))
        job.finished_at = _utc_now()
        job.update_progress("error", job.progress.get("percent", 0.0), message=str(exc))
        logger.exception("Qwen3-VL parse failed for %s (job %s)", job.doc_hash, job.job_id)
    finally:
        job.updated_at = _utc_now()
        if job.pdf_path is not None:
            job.pdf_path.unlink(missing_ok=True)
            job.pdf_path = None


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = _job_or_404(job_id)
    return JobStatusResponse(**job.as_status_payload())


@app.get("/jobs/{job_id}/result", response_model=ParseResponse)
async def get_job_result(job_id: str) -> ParseResponse:
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job not complete (status={job.status})")
    if job.text is None:
        raise HTTPException(status_code=500, detail="Job completed without text output")
    return ParseResponse(text=job.text, metadata=job.metadata or {})


@app.post("/warmup")
async def warmup_route() -> Dict[str, Any]:
    try:
        return {"warmup_complete": True, **(await _warmup())}
    except Exception as exc:
        logger.exception("Qwen3-VL warmup failed")
        raise HTTPException(status_code=500, detail=f"Qwen3-VL warmup failed: {exc}") from exc


@app.post("/control/load")
async def control_load() -> Dict[str, Any]:
    try:
        async with _app_lock:
            await _ensure_loaded()
        return {"state": "loaded", "server": vl_server.status()}
    except Exception as exc:
        logger.exception("Failed to load Qwen3-VL server")
        raise HTTPException(status_code=500, detail=f"Failed to load OCR models: {exc}") from exc


@app.post("/control/unload")
async def control_unload(force: bool = False) -> Dict[str, Any]:
    try:
        async with _app_lock:
            running = _running_jobs()
            if running and not force:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot unload while jobs are running (running_jobs={running}). Pass force=true to override.",
                )
            status = await vl_server.stop("manual")
        return {"state": "unloaded", "server": status}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to unload Qwen3-VL server")
        raise HTTPException(status_code=500, detail=f"Failed to unload OCR models: {exc}") from exc


@app.get("/control/status")
async def control_status() -> Dict[str, Any]:
    return {
        "state": "loaded" if vl_server.status().get("running") else "unloaded",
        "running_jobs": _running_jobs(),
        "server": vl_server.status(),
    }


if __name__ == "__main__":
    import uvicorn

    port_raw = os.environ.get("PORT")
    if port_raw is None or not port_raw.strip():
        raise RuntimeError("PORT environment variable must be set for ocr_qwen3vl")
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise RuntimeError(f"PORT must be an integer (got {port_raw!r})") from exc
    uvicorn.run("app:app", host="0.0.0.0", port=port)
