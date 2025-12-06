import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from mineru_wrapper import run_mineru, warmup_mineru

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _startup_logger() -> logging.Logger:
    """Prefer uvicorn's error logger so INFO shows in docker logs."""
    uvicorn_logger = logging.getLogger("uvicorn.error")
    if uvicorn_logger.hasHandlers():
        return uvicorn_logger
    return logger


def _log_cuda_status() -> None:
    """Log whether CUDA is available when the service boots."""
    log = _startup_logger()
    try:
        import torch  # type: ignore
    except Exception as exc:
        log.info("CUDA check skipped: torch unavailable (%s)", exc)
        return

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("CUDA check failed: %s", exc)
        return

    if not available:
        torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        log.info("CUDA available: False (torch cuda=%s)", torch_cuda or "unknown")
        return

    try:
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        device_count = torch.cuda.device_count()
        capability = torch.cuda.get_device_capability(device_idx)
        total_mem_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        log.info(
            "CUDA available: True (devices=%s, active=%s '%s', %.1f GB, capability=%s.%s)",
            device_count,
            device_idx,
            device_name,
            total_mem_gb,
            capability[0],
            capability[1],
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        log.info("CUDA available: True (device inspection failed: %s)", exc)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


DATA_DIR = Path(os.environ.get("DATA_DIR", "/app_data/docs"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/app_data/runtime"))
OCR_OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR") or (INDEX_DIR / "ocr_outputs"))
OCR_INCOMING_DIR = Path(os.environ.get("OCR_INCOMING_DIR") or (INDEX_DIR / "incoming"))
OCR_WARMUP_DIR = Path(os.environ.get("OCR_WARMUP_DIR") or (INDEX_DIR / "warmup"))
for path in (DATA_DIR, INDEX_DIR, OCR_OUTPUT_DIR, OCR_INCOMING_DIR, OCR_WARMUP_DIR):
    path.mkdir(parents=True, exist_ok=True)

MINERU_PARSE_METHOD = (os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
MINERU_LANG = (os.environ.get("MINERU_LANG") or "en").strip()
MINERU_TABLE_ENABLE = _env_bool("MINERU_TABLE_ENABLE", True)
MINERU_FORMULA_ENABLE = _env_bool("MINERU_FORMULA_ENABLE", True)
MINERU_WARMUP_ON_STARTUP = _env_bool("MINERU_WARMUP_ON_STARTUP", False)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class OCRJob:
    job_id: str
    doc_hash: str
    filename: str
    parse_method: str
    lang: str
    table_enable: bool
    formula_enable: bool
    pdf_path: Path
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
        payload = {
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
_ocr_loaded = False
_ocr_lock = asyncio.Lock()
_ocr_ready_event = asyncio.Event()


def _mark_ocr_ready() -> None:
    global _ocr_loaded
    _ocr_loaded = True
    if not _ocr_ready_event.is_set():
        _ocr_ready_event.set()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_cuda_status()
    if MINERU_WARMUP_ON_STARTUP:
        async def _do_warmup() -> None:
            try:
                info = await _load_ocr_models()
                logger.info("MinerU warmup finished: %s", info)
            except Exception as exc:
                logger.warning("MinerU warmup failed: %s", exc)

        asyncio.create_task(_do_warmup())
    yield


app = FastAPI(title="MinerU OCR Module", version="0.1.0", lifespan=lifespan)


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

    selected_parse_method = (parse_method or MINERU_PARSE_METHOD or "auto").strip().lower()
    selected_lang = (lang or MINERU_LANG or "en").strip()
    selected_table = table_enable if table_enable is not None else MINERU_TABLE_ENABLE
    selected_formula = formula_enable if formula_enable is not None else MINERU_FORMULA_ENABLE
    out_dir = OCR_OUTPUT_DIR / doc_hash

    job_id = uuid4().hex
    job = OCRJob(
        job_id=job_id,
        doc_hash=doc_hash,
        filename=file.filename or f"{doc_hash}.pdf",
        parse_method=selected_parse_method,
        lang=selected_lang,
        table_enable=selected_table,
        formula_enable=selected_formula,
        pdf_path=pdf_path,
    )
    job.update_progress("queued", 0.0)
    jobs[job_id] = job
    asyncio.create_task(_process_job(job, out_dir))
    return JobQueuedResponse(job_id=job.job_id, status=job.status)


async def _process_job(job: OCRJob, out_dir: Path) -> None:
    try:
        job.set_status("running")
        job.started_at = _utc_now()
        job.update_progress("initializing", 5.0)
        if not _ocr_loaded:
            job.update_progress("warming_up", 5.0)
        await _ensure_ocr_ready()
        job.update_progress("initializing", 5.0)
        def progress_hook(data: Dict[str, Any]) -> None:
            if not data:
                return
            stage = str(data.get("stage") or "parsing")
            percent = data.get("percent")
            current = data.get("current")
            total = data.get("total")
            try:
                percent_val = float(percent) if percent is not None else job.progress.get("percent", 10.0)
            except (TypeError, ValueError):
                percent_val = job.progress.get("percent", 10.0)
            job.update_progress(stage, percent_val, current=current, total=total)

        parse_task = asyncio.create_task(
            asyncio.to_thread(
                run_mineru,
                job.pdf_path,
                out_dir,
                parse_method=job.parse_method,
                lang=job.lang,
                table_enable=job.table_enable,
                formula_enable=job.formula_enable,
                progress_cb=progress_hook,
            )
        )
        result = await parse_task

        job.text = result.text
        job.metadata = result.metadata
        job.update_progress("finalizing", 95.0)
        job.finished_at = _utc_now()
        job.set_status("done")
        job.update_progress("completed", 100.0)
    except Exception as exc:
        job.error = str(exc)
        job.set_status("error", error=str(exc))
        job.finished_at = _utc_now()
        job.update_progress("error", job.progress.get("percent", 0.0), message=str(exc))
        logger.exception("MinerU parse failed for %s (job %s)", job.pdf_path, job.job_id)
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
        info = await _load_ocr_models()
        return {"warmup_complete": True, **info}
    except Exception as exc:
        logger.exception("MinerU warmup failed")
        raise HTTPException(status_code=500, detail=f"MinerU warmup failed: {exc}") from exc


async def _load_ocr_models() -> Dict[str, Any]:
    async with _ocr_lock:
        if _ocr_loaded:
            if not _ocr_ready_event.is_set():
                _ocr_ready_event.set()
            return {"state": "loaded", "already_loaded": True}
        info = await asyncio.to_thread(
            warmup_mineru,
            parse_method=MINERU_PARSE_METHOD,
            lang=MINERU_LANG,
            table_enable=MINERU_TABLE_ENABLE,
            formula_enable=MINERU_FORMULA_ENABLE,
            tmp_dir=OCR_WARMUP_DIR,
        )
        _mark_ocr_ready()
        return {"state": "loaded", **info}


@app.post("/control/load")
async def control_load() -> Dict[str, Any]:
    try:
        return await _load_ocr_models()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load OCR models")
        raise HTTPException(status_code=500, detail=f"Failed to load OCR models: {exc}") from exc


@app.get("/control/status")
async def control_status() -> Dict[str, Any]:
    return {"state": "loaded" if _ocr_loaded else "unloaded"}


async def _ensure_ocr_ready() -> None:
    if _ocr_loaded and _ocr_ready_event.is_set():
        return
    await _load_ocr_models()
    await _ocr_ready_event.wait()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
