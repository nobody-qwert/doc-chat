from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Set
from uuid import uuid4

from fastapi import HTTPException

try:
    from ..embeddings import EmbeddingClient
except ImportError:  # pragma: no cover
    from embeddings import EmbeddingClient  # type: ignore

from .classification import classify_document_text
from .context import document_store, gpu_phase_manager, jobs_registry, settings
from .models import QueuedBatchDoc, QueuedBatchJob
from .ocr import call_ocr_module
from .pipeline import build_chunk_specs
from .postprocess_runner import run_postprocess_for_doc

logger = logging.getLogger(__name__)

_job_queue: "asyncio.Queue[QueuedBatchJob]" = asyncio.Queue()
_worker_task: Optional["asyncio.Task[None]"] = None
_worker_lock = asyncio.Lock()
DEFAULT_PHASES: List[str] = ["ocr", "postprocess", "classification"]


async def queue_batch_job(docs: Sequence[QueuedBatchDoc], *, phases: Optional[Sequence[str]] = None) -> str:
    phase_order = _normalize_phases(phases)
    if not phase_order:
        raise HTTPException(status_code=400, detail="No phases requested")

    requires_file = "ocr" in phase_order
    batch_docs = [doc for doc in docs if (doc.doc_path.exists() or not requires_file)]
    if not batch_docs:
        raise HTTPException(status_code=400, detail="No documents to ingest")

    job_id = str(uuid4())
    for idx, doc in enumerate(batch_docs, start=1):
        job_record_id = f"{job_id}:{idx}"
        doc.job_record_id = job_record_id
        await document_store.create_job(job_record_id, doc.doc_hash)

    _init_job_entry(job_id, batch_docs, phases=phase_order)
    await _job_queue.put(QueuedBatchJob(job_id=job_id, docs=list(batch_docs), phases=phase_order))
    await ensure_worker_running()
    return job_id


async def queue_ocr_only_job(docs: Sequence[QueuedBatchDoc]) -> str:
    return await queue_batch_job(docs, phases=["ocr"])


async def queue_postprocess_job(doc_hashes: Sequence[str], *, phases: Optional[Sequence[str]] = None) -> str:
    if not doc_hashes:
        raise HTTPException(status_code=400, detail="No documents to postprocess")
    prepared_docs: List[QueuedBatchDoc] = []
    for doc_hash in doc_hashes:
        doc = await document_store.get_document(doc_hash)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_hash} not found")
        stored_name = doc.get("stored_name") or doc_hash
        display_name = doc.get("original_name") or stored_name
        prepared_docs.append(
            QueuedBatchDoc(
                doc_path=settings.data_dir / stored_name,
                doc_hash=doc_hash,
                display_name=display_name,
            )
        )
    requested_phases = list(phases) if phases else ["postprocess"]
    return await queue_batch_job(prepared_docs, phases=requested_phases)


async def queue_classification_job(doc_hashes: Sequence[str]) -> str:
    if not doc_hashes:
        raise HTTPException(status_code=400, detail="No documents to classify")
    prepared_docs: List[QueuedBatchDoc] = []
    for doc_hash in doc_hashes:
        doc = await document_store.get_document(doc_hash)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_hash} not found")
        stored_name = doc.get("stored_name") or doc_hash
        display_name = doc.get("original_name") or stored_name
        prepared_docs.append(
            QueuedBatchDoc(
                doc_path=settings.data_dir / stored_name,
                doc_hash=doc_hash,
                display_name=display_name,
            )
        )
    return await queue_batch_job(prepared_docs, phases=["classification"])


async def ensure_worker_running() -> None:
    global _worker_task
    async with _worker_lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_job_worker())


def _normalize_phases(phases: Optional[Sequence[str]]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for phase in phases or DEFAULT_PHASES:
        phase_name = str(phase).strip().lower()
        if not phase_name or phase_name in seen:
            continue
        seen.add(phase_name)
        ordered.append(phase_name)
    return ordered


def _init_job_entry(job_id: str, docs: Sequence[QueuedBatchDoc], *, phases: Sequence[str]) -> None:
    doc_entries: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        doc_entries.append(
            {
                "hash": doc.doc_hash,
                "file": doc.display_name,
                "order": idx,
                "status": "queued",
                "phase": "queued",
                "progress": {"phase": "queued", "stage": "queued", "percent": 0.0},
            }
        )

    jobs_registry[job_id] = {
        "status": "queued",
        "phase": "queued",
        "progress": {"phase": "queued", "stage": "queued", "percent": 0.0},
        "docs": doc_entries,
        "total_docs": len(doc_entries),
        "file": doc_entries[0]["file"] if len(doc_entries) == 1 else f"{len(doc_entries)} documents",
        "hash": doc_entries[0]["hash"] if len(doc_entries) == 1 else None,
        "phases": list(phases),
    }


async def _ensure_phase_resources(phase: str, job_id: str) -> None:
    if phase in {"ocr", "postprocess"}:
        await gpu_phase_manager.switch_to_no_llm(reason=f"{phase}:{job_id}")
    elif phase == "classification":
        await gpu_phase_manager.ensure_llm_ready()


def _set_job_progress(job_id: str, phase: str, percent: float, *, stage: Optional[str] = None, **extra: Any) -> None:
    info = jobs_registry.get(job_id)
    if not info:
        return
    payload = {
        "phase": phase,
        "stage": stage or phase,
        "percent": max(0.0, min(100.0, float(percent))),
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    info["phase"] = phase
    info["progress"] = payload


def _update_doc_progress(doc_entry: Dict[str, Any], phase: str, percent: float, *, stage: Optional[str] = None, **extra: Any) -> None:
    payload = {
        "phase": phase,
        "stage": stage or phase,
        "percent": max(0.0, min(100.0, float(percent))),
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    doc_entry["phase"] = phase
    doc_entry["progress"] = payload


async def _job_worker() -> None:
    while True:
        job = await _job_queue.get()
        try:
            await _process_job(job.job_id, job.docs, job.phases)
        except asyncio.CancelledError:
            _job_queue.put_nowait(job)
            raise
        except Exception:  # pragma: no cover
            logger.exception("Unhandled error while processing queued job %s", job.job_id)
        finally:
            _job_queue.task_done()


async def _process_job(job_id: str, docs: Sequence[QueuedBatchDoc], phases: Sequence[str]) -> None:
    info = jobs_registry.get(job_id)
    if not info:
        return

    total_docs = len(docs)
    if total_docs == 0:
        info["status"] = "done"
        _set_job_progress(job_id, "completed", 100.0, stage="completed")
        return

    doc_entries = info.get("docs", [])
    doc_by_hash = {entry.get("hash"): entry for entry in doc_entries}
    info["status"] = "running"

    phase_order = _normalize_phases(phases)
    if not phase_order:
        info["status"] = "error"
        _set_job_progress(job_id, "error", 0.0, stage="no_phases", message="No phases requested")
        return

    context: Dict[str, Any] = {"ocr_payloads": {}}
    had_errors = False
    current_docs: Sequence[QueuedBatchDoc] = docs

    phase_count = len(phase_order)
    for idx, phase_name in enumerate(phase_order):
        start_pct = (idx / phase_count) * 100.0
        end_pct = ((idx + 1) / phase_count) * 100.0

        try:
            await _ensure_phase_resources(phase_name, job_id)
        except Exception as exc:
            error_msg = f"GPU not available: {exc}"
            _set_job_progress(job_id, "error", 0.0, stage="gpu_unavailable", message=error_msg)
            for entry in doc_entries:
                entry["status"] = "error"
                _update_doc_progress(entry, "error", 0.0, stage="gpu_unavailable", error=error_msg)
            for doc in docs:
                if doc.job_record_id:
                    await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
            info["status"] = "error"
            return

        if phase_name == "ocr":
            successful_docs, ocr_payloads, ocr_errors = await _run_ocr_phase(
                job_id, current_docs, doc_by_hash, total_docs, start_percent=start_pct, end_percent=end_pct
            )
            context["ocr_payloads"] = ocr_payloads
            current_docs = successful_docs
            had_errors = had_errors or bool(ocr_errors) or not successful_docs
            if not successful_docs:
                break
            continue

        if phase_name == "postprocess":
            postprocess_errors = await _run_postprocess_phase(
                job_id,
                current_docs,
                doc_by_hash,
                context.get("ocr_payloads") or {},
                start_percent=start_pct,
                end_percent=end_pct,
            )
            had_errors = had_errors or bool(postprocess_errors)
            current_docs = [doc for doc in current_docs if doc_by_hash.get(doc.doc_hash, {}).get("status") != "error"]
            if not current_docs:
                break
            continue

        if phase_name == "classification":
            classification_errors = await _run_classification_phase(
                job_id,
                current_docs,
                doc_by_hash,
                context.get("ocr_payloads") or {},
                start_percent=start_pct,
                end_percent=end_pct,
            )
            had_errors = had_errors or bool(classification_errors)
            current_docs = [doc for doc in current_docs if doc_by_hash.get(doc.doc_hash, {}).get("status") != "error"]
            continue

        logger.warning("Unknown phase '%s' for job %s; skipping", phase_name, job_id)
        _set_job_progress(job_id, "warning", start_pct, stage="unknown_phase", phase=phase_name)

    had_errors = had_errors or any(entry.get("status") == "error" for entry in doc_entries)
    info["status"] = "error" if had_errors else "done"
    final_stage = "failed" if had_errors else "done"
    _set_job_progress(job_id, "completed" if not had_errors else "error", 100.0, stage=final_stage)


async def _run_ocr_phase(
    job_id: str,
    docs: Sequence[QueuedBatchDoc],
    doc_by_hash: Dict[str, Dict[str, Any]],
    total_docs: int,
    *,
    start_percent: float = 0.0,
    end_percent: float = 50.0,
) -> tuple[List[QueuedBatchDoc], Dict[str, Dict[str, Any]], Dict[str, str]]:
    _set_job_progress(job_id, "ocr", start_percent, stage="starting", total_docs=total_docs)

    if total_docs == 0:
        _set_job_progress(job_id, "ocr", end_percent, stage="empty")
        return [], {}, {}

    ocr_payloads: Dict[str, Dict[str, Any]] = {}
    successful_docs: List[QueuedBatchDoc] = []
    errors: Dict[str, str] = {}

    for index, doc in enumerate(docs, start=1):
        doc_entry = doc_by_hash.get(doc.doc_hash)
        if not doc_entry:
            continue
        await document_store.update_document_status(doc.doc_hash, "ocr_pending")
        if doc.job_record_id:
            await document_store.mark_job_started(doc.job_record_id)
        doc_entry["status"] = "ocr_running"
        _update_doc_progress(doc_entry, "ocr", 0.0, stage="starting")

        start_ocr = time.perf_counter()

        def ocr_progress_callback(data: Dict[str, Any]) -> None:
            if not data:
                return
            stage = str(data.get("stage") or "ocr")
            percent_raw = data.get("percent")
            try:
                percent_val = float(percent_raw) if percent_raw is not None else float(doc_entry.get("progress", {}).get("percent", 0.0))
            except (TypeError, ValueError):
                percent_val = float(doc_entry.get("progress", {}).get("percent", 0.0))
            _update_doc_progress(
                doc_entry,
                "ocr",
                percent_val,
                stage=stage,
                current=data.get("current"),
                total=data.get("total"),
            )

        try:
            ocr_result = await call_ocr_module(doc.doc_hash, doc.doc_path, progress_cb=ocr_progress_callback)
            ocr_time = time.perf_counter() - start_ocr
            ocr_text_raw = ocr_result.get("text") or ""
            ocr_text = ocr_text_raw.strip()
            if not ocr_text:
                raise RuntimeError(f"OCR parser returned no text for document '{doc.display_name}'")
            ocr_meta = ocr_result.get("metadata") or {}
            await document_store.upsert_extraction(
                doc.doc_hash,
                settings.ocr_parser_key,
                text=ocr_text,
                meta_json=json.dumps({**ocr_meta, "source": "ocr"}),
            )
            await document_store.update_document_status(doc.doc_hash, "waiting_postprocess")
            doc_entry["status"] = "ocr_completed"
            _update_doc_progress(doc_entry, "ocr", 100.0, stage="completed")
            ocr_payloads[doc.doc_hash] = {"text": ocr_text, "metadata": ocr_meta, "ocr_time": ocr_time}
            successful_docs.append(doc)
        except Exception as exc:  # pragma: no cover
            error_msg = str(exc)
            errors[doc.doc_hash] = error_msg
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "ocr", doc_entry.get("progress", {}).get("percent", 0.0), stage="failed", error=error_msg)
            await document_store.update_document_status(doc.doc_hash, "error", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
                doc.job_record_id = None
            logger.exception("OCR failed for %s: %s", doc.doc_path, exc)
        finally:
            phase_span = max(0.0, end_percent - start_percent)
            percent = start_percent + (index / total_docs) * phase_span if total_docs else end_percent
            _set_job_progress(job_id, "ocr", percent, stage=f"{index}/{total_docs}")

    return successful_docs, ocr_payloads, errors


async def _get_ocr_payload_for_doc(doc_hash: str, existing: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if doc_hash in existing and existing[doc_hash].get("text"):
        return existing[doc_hash]
    extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
    if not extraction:
        return {}
    ocr_text = (extraction.get("text") or "").strip()
    if not ocr_text:
        return {}
    meta = extraction.get("metadata") or {}
    payload: Dict[str, Any] = {"text": ocr_text, "metadata": meta}
    try:
        prev_metrics = await document_store.get_performance_metrics(doc_hash)
        if prev_metrics and prev_metrics.get("ocr_time_sec") is not None:
            try:
                payload["ocr_time"] = float(prev_metrics["ocr_time_sec"])
            except (TypeError, ValueError):
                payload["ocr_time"] = prev_metrics.get("ocr_time_sec")
    except Exception:  # pragma: no cover - best-effort metrics lookup
        pass
    existing[doc_hash] = payload
    return payload


async def _run_postprocess_phase(
    job_id: str,
    docs: Sequence[QueuedBatchDoc],
    doc_by_hash: Dict[str, Dict[str, Any]],
    ocr_payloads: Dict[str, Dict[str, Any]],
    *,
    start_percent: float = 50.0,
    end_percent: float = 100.0,
) -> Dict[str, str]:
    chunk_specs = build_chunk_specs()
    _set_job_progress(job_id, "postprocess", start_percent, stage="starting", total_docs=len(docs))
    emb_client = EmbeddingClient()
    post_total = len(docs)
    phase_span = max(0.0, end_percent - start_percent)
    if post_total == 0:
        _set_job_progress(job_id, "postprocess", end_percent, stage="empty")
        return {}
    processed_docs = 0
    errors: Dict[str, str] = {}

    for doc in docs:
        doc_entry = doc_by_hash.get(doc.doc_hash)
        if not doc_entry:
            continue
        payload = await _get_ocr_payload_for_doc(doc.doc_hash, ocr_payloads)
        ocr_text = payload.get("text") or ""
        if not ocr_text:
            error_msg = "missing_ocr_text"
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "chunking", 0.0, stage="failed", error=error_msg)
            errors[doc.doc_hash] = error_msg
            continue

        def chunk_progress_callback(data: Dict[str, Any]) -> None:
            if not data:
                return
            spec_name_raw = data.get("spec")
            spec_stage_raw = data.get("stage")
            spec_name = str(spec_name_raw).strip() if spec_name_raw else "chunk"
            spec_stage = str(spec_stage_raw).strip() if isinstance(spec_stage_raw, str) else ""
            if not spec_stage:
                spec_stage = "processing"
            try:
                spec_percent = float(data.get("percent") or 0.0)
            except (TypeError, ValueError):
                spec_percent = 0.0
            try:
                spec_index = int(data.get("spec_index") or 1)
            except (TypeError, ValueError):
                spec_index = 1
            try:
                spec_total = int(data.get("spec_total") or 1)
            except (TypeError, ValueError):
                spec_total = 1
            spec_index = max(1, spec_index)
            spec_total = max(1, spec_total)
            spec_fraction = max(0.0, min(1.0, spec_percent / 100.0))
            overall_fraction = ((spec_index - 1) + spec_fraction) / spec_total
            overall_percent = max(0.0, min(100.0, overall_fraction * 100.0))
            stage_label = f"{spec_name}:{spec_stage}" if spec_stage else spec_name
            _update_doc_progress(
                doc_entry,
                "chunking",
                overall_percent,
                stage=stage_label,
                spec=spec_name,
                spec_index=spec_index,
                spec_total=spec_total,
                spec_percent=spec_percent,
                chunk_count=data.get("chunk_count"),
            )

        def chunk_start_callback() -> None:
            doc_entry["status"] = "chunking"
            _update_doc_progress(doc_entry, "chunking", 0.0, stage="starting")

        def embedding_start_callback() -> None:
            doc_entry["status"] = "embedding"
            _update_doc_progress(doc_entry, "embedding", 0.0, stage="starting")

        def embedding_progress_callback(data: Dict[str, Any]) -> None:
            if not data:
                return
            try:
                percent_val = float(data.get("percent") or 0.0)
            except (TypeError, ValueError):
                percent_val = 0.0
            _update_doc_progress(
                doc_entry,
                "embedding",
                percent_val,
                stage="batch",
                processed=data.get("processed"),
                total=data.get("total"),
            )

        try:
            result = await run_postprocess_for_doc(
                doc.doc_hash,
                ocr_text=ocr_text,
                emb_client=emb_client,
                chunk_specs=chunk_specs,
                ocr_time=payload.get("ocr_time"),
                chunk_progress_cb=chunk_progress_callback,
                embedding_progress_cb=embedding_progress_callback,
                on_chunk_start=chunk_start_callback,
                on_embedding_start=embedding_start_callback,
            )

            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "done")
                doc.job_record_id = None

            doc_entry["status"] = "completed"
            _update_doc_progress(
                doc_entry,
                "completed",
                100.0,
                stage="done",
                embeddings=result["total_embeddings"],
                chunk_embeddings=result.get("chunk_count", result.get("total_chunks", result["total_embeddings"])),
            )
        except Exception as exc:
            error_msg = str(exc)
            errors[doc.doc_hash] = error_msg
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "chunking", doc_entry.get("progress", {}).get("percent", 0.0), stage="failed", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
                doc.job_record_id = None
            logger.exception("Chunking/embedding failed for %s: %s", doc.doc_hash, exc)
            continue

        processed_docs += 1
        percent = start_percent + (processed_docs / post_total) * phase_span if post_total else end_percent
        _set_job_progress(job_id, "postprocess", percent, stage=f"{processed_docs}/{post_total}")

    return errors


async def _run_classification_phase(
    job_id: str,
    docs: Sequence[QueuedBatchDoc],
    doc_by_hash: Dict[str, Dict[str, Any]],
    ocr_payloads: Dict[str, Dict[str, Any]],
    *,
    start_percent: float = 0.0,
    end_percent: float = 100.0,
) -> Dict[str, str]:
    total_docs = len(docs)
    _set_job_progress(job_id, "classification", start_percent, stage="starting", total_docs=total_docs)
    if total_docs == 0:
        _set_job_progress(job_id, "classification", end_percent, stage="empty")
        return {}
    phase_span = max(0.0, end_percent - start_percent)
    completed = 0
    errors: Dict[str, str] = {}

    for doc in docs:
        doc_entry = doc_by_hash.get(doc.doc_hash)
        if not doc_entry:
            continue
        payload = await _get_ocr_payload_for_doc(doc.doc_hash, ocr_payloads)
        ocr_text = (payload.get("text") or "").strip()
        if not ocr_text:
            error_msg = "missing_ocr_text"
            errors[doc.doc_hash] = error_msg
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "classification", 0.0, stage="failed", error=error_msg)
            await document_store.update_classification_status(doc.doc_hash, "error", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
                doc.job_record_id = None
            completed += 1
            percent = start_percent + (completed / total_docs) * phase_span if total_docs else end_percent
            _set_job_progress(job_id, "classification", percent, stage=f"{completed}/{total_docs}")
            continue

        doc_entry["status"] = "classification_running"
        _update_doc_progress(doc_entry, "classification", 0.0, stage="starting")
        await document_store.update_classification_status(doc.doc_hash, "running")

        try:
            result = await classify_document_text(
                doc_hash=doc.doc_hash,
                text=ocr_text,
                document_name=doc.display_name,
            )
            await document_store.save_document_classification(
                doc.doc_hash,
                l1_id=result["l1_id"],
                l1_name=result.get("l1_name"),
                l2_id=result.get("l2_id"),
                l2_name=result.get("l2_name"),
                l1_confidence=result.get("l1_confidence"),
                l2_confidence=result.get("l2_confidence"),
                l1_reason=result.get("l1_reason"),
                l2_reason=result.get("l2_reason"),
                model=result.get("model"),
                raw_response=result.get("raw_response"),
            )
            doc_entry["status"] = "classification_completed"
            _update_doc_progress(
                doc_entry,
                "classification",
                100.0,
                stage="done",
                classification=result,
            )
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "done")
                doc.job_record_id = None
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
            error_msg = detail or f"Classification failed ({exc.status_code})"
            errors[doc.doc_hash] = error_msg
            doc_entry["status"] = "error"
            _update_doc_progress(
                doc_entry,
                "classification",
                doc_entry.get("progress", {}).get("percent", 0.0),
                stage="failed",
                error=error_msg,
            )
            await document_store.update_classification_status(doc.doc_hash, "error", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
                doc.job_record_id = None
            logger.exception("Classification failed for %s: %s", doc.doc_hash, error_msg)
        except Exception as exc:  # pragma: no cover
            error_msg = str(exc)
            errors[doc.doc_hash] = error_msg
            doc_entry["status"] = "error"
            _update_doc_progress(
                doc_entry,
                "classification",
                doc_entry.get("progress", {}).get("percent", 0.0),
                stage="failed",
                error=error_msg,
            )
            await document_store.update_classification_status(doc.doc_hash, "error", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
                doc.job_record_id = None
            logger.exception("Classification failed for %s: %s", doc.doc_hash, exc)
        finally:
            completed += 1
            percent = start_percent + (completed / total_docs) * phase_span if total_docs else end_percent
            _set_job_progress(job_id, "classification", percent, stage=f"{completed}/{total_docs}")

    return errors
