from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("pymupdf is required: pip install pymupdf") from exc


@dataclass(frozen=True)
class RenderSettings:
    dpi: int
    max_dim: int


@dataclass(frozen=True)
class RenderedPage:
    page_index: int
    png_bytes: bytes
    size: Tuple[int, int]


def _scale_for_dpi(dpi: int) -> float:
    return max(0.1, float(dpi) / 72.0)


def render_pdf_to_png_pages(
    pdf_path: Path,
    *,
    dpi: int = 200,
    max_dim: int = 1800,
    page_limit: Optional[int] = None,
) -> Tuple[List[RenderedPage], RenderSettings]:
    doc = fitz.open(str(pdf_path))
    pages: List[RenderedPage] = []
    limit = page_limit if page_limit is not None else doc.page_count
    limit = max(0, min(int(limit), int(doc.page_count)))

    for idx in range(limit):
        page = doc.load_page(idx)
        base_scale = _scale_for_dpi(dpi)
        matrix = fitz.Matrix(base_scale, base_scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # If too large, downscale in-memory with PIL to avoid huge payloads.
        if max(pix.width, pix.height) > max_dim:
            png = pix.tobytes("png")
            with Image.open(io.BytesIO(png)) as img:
                img = img.convert("RGB")
                factor = max_dim / max(img.width, img.height)
                new_size = (max(1, int(img.width * factor)), max(1, int(img.height * factor)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                out = io.BytesIO()
                img.save(out, format="PNG", optimize=True)
                png_bytes = out.getvalue()
                size = new_size
        else:
            png_bytes = pix.tobytes("png")
            size = (pix.width, pix.height)

        pages.append(RenderedPage(page_index=idx, png_bytes=png_bytes, size=size))

    doc.close()
    return pages, RenderSettings(dpi=int(dpi), max_dim=int(max_dim))

