from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from .tokenizer_registry import load_tokenizer, record_tokenizer_fallback
except ImportError:  # pragma: no cover
    from tokenizer_registry import load_tokenizer, record_tokenizer_fallback  # type: ignore


@dataclass
class Chunk:
    chunk_id: str
    order_index: int
    text: str
    token_count: int


@dataclass(frozen=True)
class ChunkWindowSpec:
    """Defines a sliding window with a fixed-sized core and optional padding."""

    name: str
    core_size: int
    left_padding: int = 0
    right_padding: int = 0
    step_size: Optional[int] = None

    def step(self) -> int:
        base = self.step_size if self.step_size is not None else self.core_size
        return max(1, base)


def _tokenizer(tokenizer_id: Optional[str] = None) -> Any:
    identifier = (tokenizer_id or "").strip()
    if not identifier:
        return None
    return load_tokenizer(identifier)


def _encode(enc, text: str, identifier: Optional[str]) -> List[int]:
    if enc is None:
        # naive fallback: 1 token per ~4 chars
        if identifier:
            record_tokenizer_fallback(identifier, "chunk_encode_fallback")
        n = max(1, len(text) // 4)
        return list(range(n))
    try:
        return enc.encode(text, add_special_tokens=False)
    except TypeError:
        return enc.encode(text)


def _decode(enc, tokens: Sequence[int]) -> str:
    if enc is None:
        # cannot reconstruct; not used in fallback path
        return ""
    try:
        return enc.decode(list(tokens), skip_special_tokens=True)
    except TypeError:
        return enc.decode(list(tokens))


def _materialize_span(
    enc,
    token_ids: Sequence[int],
    text: str,
    start: int,
    end: int,
) -> str:
    if enc is None:
        approx_chars_per_token = 4
        raw_start = min(len(text), start * approx_chars_per_token)
        span_len = max(1, (end - start) * approx_chars_per_token)
        raw_end = min(len(text), raw_start + span_len)
        return text[raw_start:raw_end]
    return _decode(enc, token_ids[start:end])


def _chunks_for_spec(
    enc,
    token_ids: Sequence[int],
    text: str,
    spec: ChunkWindowSpec,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Chunk]:
    total_tokens = len(token_ids)
    if total_tokens == 0 or spec.core_size <= 0:
        return []

    out: List[Chunk] = []
    step = spec.step()

    if progress_cb:
        progress_cb({"stage": "starting", "percent": 0.0, "chunk_count": 0, "total_tokens": total_tokens})

    for core_start in range(0, total_tokens, step):
        core_end = min(total_tokens, core_start + spec.core_size)
        if core_end <= core_start:
            break

        window_start = max(0, core_start - max(0, spec.left_padding))
        window_end = min(total_tokens, core_end + max(0, spec.right_padding))
        if window_end <= window_start:
            continue

        content = _materialize_span(enc, token_ids, text, window_start, window_end)
        content_norm = (content or "").strip()
        if not content_norm:
            continue

        chunk_hash = md5(f"{spec.name}:{content_norm}".encode("utf-8")).hexdigest()
        chunk_id = f"chunk-{spec.name}-{chunk_hash}"
        out.append(
            Chunk(
                chunk_id=chunk_id,
                order_index=len(out),
                text=content_norm,
                token_count=window_end - window_start,
            )
        )

        if progress_cb:
            percent = 100.0 if total_tokens == 0 else (core_end / total_tokens) * 100.0
            progress_cb(
                {
                    "stage": "processing",
                    "percent": percent,
                    "chunk_count": len(out),
                    "token_end": core_end,
                    "total_tokens": total_tokens,
                }
            )

        if core_end == total_tokens:
            break

    if progress_cb:
        progress_cb({"stage": "completed", "percent": 100.0, "chunk_count": len(out), "total_tokens": total_tokens})

    return out


def chunk_text_multi(
    text: str,
    specs: Sequence[ChunkWindowSpec],
    *,
    tokenizer_id: Optional[str] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, List[Chunk]]:
    """Return chunk lists for each spec using a shared tokenization."""
    enc = _tokenizer(tokenizer_id)
    token_ids = _encode(enc, text, tokenizer_id)
    spec_list = list(specs)
    total_specs = max(1, len(spec_list))
    results: Dict[str, List[Chunk]] = {}

    for index, spec in enumerate(spec_list, start=1):
        cb: Optional[Callable[[Dict[str, Any]], None]] = None
        if progress_cb:
            def _emit(payload: Dict[str, Any], *, _spec=spec, _index=index) -> None:
                enriched = {
                    "spec": _spec.name,
                    "spec_index": _index,
                    "spec_total": total_specs,
                }
                enriched.update(payload)
                progress_cb(enriched)

            cb = _emit
            _emit({"stage": "spec_start", "percent": 0.0})

        results[spec.name] = _chunks_for_spec(enc, token_ids, text, spec, progress_cb=cb)

        if cb:
            cb({"stage": "spec_completed", "percent": 100.0, "chunk_count": len(results[spec.name])})

    return results


def count_text_tokens(text: str, tokenizer_id: Optional[str] = None) -> int:
    """Return the approximate token count used by the chunker."""
    enc = _tokenizer(tokenizer_id)
    return len(_encode(enc, text, tokenizer_id))
