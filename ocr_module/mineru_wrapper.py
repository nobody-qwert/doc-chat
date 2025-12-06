from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MineruResult:
    text: str
    metadata: Dict[str, Any]


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    v = val.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _generate_warmup_pdf_bytes(text: str = "MinerU warmup") -> bytes:
    """Create a minimal single page PDF for warmup without external deps."""
    safe_text = _escape_pdf_text(text)
    stream_body = f"BT\n/F1 24 Tf\n72 720 Td\n({safe_text}) Tj\nET\n"
    stream_bytes = stream_body.encode("latin-1")
    object_templates = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_body}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header_bytes = b"%PDF-1.4\n"
    doc_parts: List[bytes] = [header_bytes]
    offsets: List[int] = []
    current = len(header_bytes)
    for idx, body in enumerate(object_templates, start=1):
        obj_bytes = f"{idx} 0 obj\n{body}\nendobj\n".encode("latin-1")
        offsets.append(current)
        doc_parts.append(obj_bytes)
        current += len(obj_bytes)
    xref_entries = [
        f"xref\n0 {len(object_templates) + 1}\n",
        "0000000000 65535 f \n",
    ] + [f"{offset:010d} 00000 n \n" for offset in offsets]
    startxref = current
    xref_bytes = "".join(xref_entries).encode("latin-1")
    trailer_str = (
        "trailer\n"
        f"<< /Size {len(object_templates) + 1} /Root 1 0 R >>\n"
        "startxref\n"
        f"{startxref}\n"
        "%%EOF\n"
    )
    doc_parts.append(xref_bytes)
    doc_parts.append(trailer_str.encode("latin-1"))
    return b"".join(doc_parts)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(value.strip())
    except (TypeError, ValueError):
        return None
    return None


def _extract_images_from_markdown(markdown_text: str, *, chunk: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not markdown_text:
        return entries
    for match in _IMAGE_PATTERN.finditer(markdown_text):
        src = (match.group(1) or "").strip()
        if not src or src.startswith("data:") or "://" in src:
            continue
        cleaned = src.lstrip("./")
        entries.append({"chunk": chunk, "path": cleaned})
    return entries


def _extract_images_from_content_list(content_data: Any, *, chunk: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def _walk(node: Any, context: Dict[str, Any]) -> None:
        if isinstance(node, dict):
            next_ctx = context
            for key in ("page", "page_id", "page_index", "page_number"):
                if key in node:
                    page_val = _coerce_int(node.get(key))
                    if page_val is not None:
                        next_ctx = {**context, "page": page_val}
                        break

            node_type = str(node.get("type") or node.get("category") or "").strip().lower()
            candidate_path: Optional[str] = None
            for key in ("image_path", "img_path", "img", "path", "value", "source_path"):
                val = node.get(key)
                if isinstance(val, str) and "images" in val:
                    candidate_path = val.strip()
                    break
            if candidate_path:
                cleaned = candidate_path.lstrip("./")
                entry: Dict[str, Any] = {
                    "chunk": chunk,
                    "page": next_ctx.get("page"),
                    "type": node_type or "image",
                    "path": cleaned,
                }
                if "bbox" in node:
                    entry["bbox"] = node["bbox"]
                elif "box" in node:
                    entry["bbox"] = node["box"]
                elif "position" in node:
                    entry["bbox"] = node["position"]
                if "width" in node:
                    entry["width"] = node.get("width")
                if "height" in node:
                    entry["height"] = node.get("height")
                results.append(entry)
            for value in node.values():
                _walk(value, next_ctx)
        elif isinstance(node, list):
            for item in node:
                _walk(item, context)
        elif isinstance(node, str):
            if "images" in node:
                cleaned = node.strip().lstrip("./")
                results.append({"chunk": chunk, "page": context.get("page"), "type": "image", "path": cleaned})

    _walk(content_data, {})
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for item in results:
        key = (item.get("path"), item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def run_mineru(
    pdf_path: Path,
    out_dir: Path,
    *,
    parse_method: str | None = None,
    lang: str | None = None,
    table_enable: bool | None = None,
    formula_enable: bool | None = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MineruResult:
    """Run MinerU on a single PDF with GPU detection and return Markdown text.

    This follows the pattern proven to work in ocr_bench.
    """
    try:
        from mineru.cli.common import do_parse  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("mineru is not installed: pip install mineru") from exc

    # Resolve device mode: auto|cuda|cpu
    requested = (os.environ.get("MINERU_DEVICE_MODE") or "auto").strip().lower()
    effective = requested
    cuda_available = False
    gpu_name: Optional[str] = None
    gpu_mem_gb: Optional[float] = None
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        cuda_available = False

    if requested in (None, "", "auto"):
        effective = "cuda" if cuda_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "MINERU_DEVICE_MODE=cuda requested but CUDA not available. Install CUDA torch and run with GPU runtime."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem
    pdf_bytes = pdf_path.read_bytes()

    # Defaults can be overridden by env or function args
    p_method = (parse_method or os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
    p_lang = (lang or os.environ.get("MINERU_LANG") or "en").strip()
    p_table = table_enable if table_enable is not None else _env_bool("MINERU_TABLE_ENABLE", True)
    p_formula = formula_enable if formula_enable is not None else _env_bool("MINERU_FORMULA_ENABLE", True)

    combined_parts: List[str] = []
    all_image_entries: List[Dict[str, Any]] = []
    collected_images: List[str] = []
    collected_content_lists: List[str] = []

    final_dir = out_dir / pdf_name / p_method
    final_images_dir = final_dir / "images"
    content_list_dir = final_dir / "content_lists"
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    def emit(stage: str, current: int, total: int) -> None:
        if not progress_cb:
            return
        total = max(1, total)
        percent = max(0.0, min(100.0, (current / total) * 100.0))
        progress_cb({
            "stage": stage,
            "percent": percent,
            "current": current,
            "total": total,
        })

    emit("parsing", 0, 1)

    do_parse(
        output_dir=str(out_dir),
        pdf_file_names=[pdf_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[p_lang],
        backend="pipeline",
        parse_method=p_method,
        formula_enable=p_formula,
        table_enable=p_table,
        start_page_id=0,
        end_page_id=None,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        device_mode=effective,
    )

    emit("parsing", 1, 1)

    if not final_dir.exists():
        raise RuntimeError(f"MinerU did not produce expected output directory at {final_dir}")

    chunk_markdown_path = final_dir / f"{pdf_name}.md"
    if not chunk_markdown_path.exists():
        raise RuntimeError(f"MinerU did not produce expected Markdown at {chunk_markdown_path}")
    chunk_markdown = chunk_markdown_path.read_text(encoding="utf-8")
    combined_parts.append(chunk_markdown)

    chunk_content_list_path = final_dir / "content_list.json"

    if chunk_content_list_path.exists():
        content_list_dir.mkdir(parents=True, exist_ok=True)
        copied_list_path = content_list_dir / f"{pdf_name}.json"
        shutil.copy2(chunk_content_list_path, copied_list_path)
        collected_content_lists.append(f"content_lists/{copied_list_path.name}")
        try:
            loaded_content = json.loads(chunk_content_list_path.read_text(encoding="utf-8"))
            images_from_content = _extract_images_from_content_list(loaded_content, chunk=pdf_name)
            all_image_entries.extend(images_from_content)
        except Exception:
            pass
    else:
        markdown_images = _extract_images_from_markdown(chunk_markdown, chunk=pdf_name)
        all_image_entries.extend(markdown_images)

    if final_images_dir.exists():
        collected_images = [
            f"images/{image_file.name}"
            for image_file in sorted(final_images_dir.iterdir())
            if image_file.is_file()
        ]

    md_path = final_dir / f"{pdf_name}.md"
    text = "\n\n".join(part.strip() for part in combined_parts if part.strip())
    if not text:
        raise RuntimeError("MinerU produced no text output")
    md_path.write_text(text, encoding="utf-8")

    meta = {
        "markdown_path": str(md_path),
        "device_mode": effective,
        "gpu": {"available": cuda_available, "name": gpu_name, "mem_gb": gpu_mem_gb},
        "parse_method": p_method,
        "lang": p_lang,
        "table_enable": p_table,
        "formula_enable": p_formula,
        "chunks": len(combined_parts),
        "assets": {
            "base_dir": str(final_dir),
            "images_dir": "images",
            "image_files": list(dict.fromkeys(collected_images)),
            "content_lists": collected_content_lists,
        },
        "image_blocks": all_image_entries,
    }

    # Write a small summary for debugging
    try:
        (out_dir / "mineru_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    return MineruResult(text=text, metadata=meta)


def warmup_mineru(*,
                  parse_method: str | None = None,
                  lang: str | None = None,
                  table_enable: bool | None = False,
                  formula_enable: bool | None = False,
                  device_mode: str | None = None,
                  tmp_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Trigger MinerU model initialization and weight downloads.

    Generates a tiny in-memory PDF and runs a minimal parse with most dumps disabled.
    This reduces first-ingest latency in production.
    """
    try:
        from mineru.cli.common import do_parse  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("mineru is not installed: pip install mineru") from exc

    # Optionally override device mode for warmup
    if device_mode:
        os.environ["MINERU_DEVICE_MODE"] = device_mode

    # Resolve device mode: auto|cuda|cpu
    requested = (os.environ.get("MINERU_DEVICE_MODE") or "auto").strip().lower()
    effective = requested
    cuda_available = False
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if requested in (None, "", "auto"):
        effective = "cuda" if cuda_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "MINERU_DEVICE_MODE=cuda requested but CUDA not available. Install CUDA torch and run with GPU runtime."
        )

    # Create a tiny single-page PDF in memory without external dependencies
    pdf_bytes = _generate_warmup_pdf_bytes()

    default_base = Path(os.environ.get("INDEX_DIR", "/app_data/runtime")) / "_warmup"
    out_base = Path(tmp_dir) if tmp_dir is not None else default_base
    out_base.mkdir(parents=True, exist_ok=True)

    # Effective config
    p_method = (parse_method or os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
    p_lang = (lang or os.environ.get("MINERU_LANG") or "en").strip()
    p_table = table_enable if table_enable is not None else _env_bool("MINERU_TABLE_ENABLE", True)
    p_formula = formula_enable if formula_enable is not None else _env_bool("MINERU_FORMULA_ENABLE", True)

    # Run a minimal parse over a single page
    do_parse(
        output_dir=str(out_base),
        pdf_file_names=["warmup"],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[p_lang],
        backend="pipeline",
        parse_method=p_method,
        formula_enable=p_formula,
        table_enable=p_table,
        start_page_id=0,
        end_page_id=1,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        device_mode=effective,
    )

    return {
        "ok": True,
        "device_mode": effective,
        "parse_method": p_method,
        "lang": p_lang,
        "table_enable": p_table,
        "formula_enable": p_formula,
        "output_dir": str(out_base),
    }
