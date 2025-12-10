from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple


@dataclass(frozen=True)
class ChunkingConfigSpec:
    config_id: str
    label: str
    description: str
    core_size: int
    left_overlap: int
    right_overlap: int
    step_size: int


@dataclass(frozen=True)
class AppSettings:
    data_dir: Path
    index_dir: Path
    doc_store_path: Path
    ocr_parser_key: str
    ocr_module_url: str
    ocr_module_timeout: float
    chat_context_window: int
    chat_completion_max_tokens: int
    chat_completion_reserve: int
    min_context_similarity: float
    min_semantic_similarity: float
    min_keyword_score: float
    no_context_response: str
    system_prompt: str
    continue_prompt: str
    completed_doc_statuses: Set[str]
    frontend_origin: str
    chunk_size: int
    chunk_overlap: int
    chunk_config_small_id: str
    chunking_configs: Tuple[ChunkingConfigSpec, ...]
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_tokenizer_id: str
    llm_temperature: float
    llm_top_p: Optional[float]
    llm_top_k: Optional[int]
    llm_min_p: Optional[float]
    llm_repeat_penalty: Optional[float]
    ocr_status_poll_interval: float
    llm_control_url: Optional[str]
    ocr_control_url: Optional[str]
    gpu_phase_timeout: float
    llm_ready_timeout: float
    diagnostics_url: Optional[str]
    embedding_tokenizer_id: str
    agentic_max_subqueries: int
    agentic_short_doc_chunk_limit: int
    agentic_short_doc_token_limit: int
    agentic_expansion_chunks_before: int
    agentic_expansion_chunks_after: int
    agentic_max_expanded_chunks: int
    agentic_expanded_char_multiplier: int
    agentic_min_expanded_chars: int
    agentic_max_expanded_chars: int


def _require_env(name: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"Missing required environment variable {name}")
    value = str(raw).strip()
    if value == "":
        raise RuntimeError(f"Environment variable {name} cannot be empty")
    return value


def _int_env(name: str) -> int:
    value = _require_env(name)
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - fail fast on misconfiguration
        raise RuntimeError(f"Environment variable {name} must be an integer (got {value!r})") from exc


def _float_env(name: str) -> float:
    value = _require_env(name)
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - fail fast on misconfiguration
        raise RuntimeError(f"Environment variable {name} must be a float (got {value!r})") from exc


def _str_env(name: str) -> str:
    return _require_env(name)


def load_settings() -> AppSettings:
    data_dir = Path(_str_env("DATA_DIR"))
    index_dir = Path(_str_env("INDEX_DIR"))
    doc_store_path = Path(_str_env("DOC_STORE_PATH"))

    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    chat_completion_max_tokens = _int_env("CHAT_COMPLETION_MAX_TOKENS")
    chat_completion_reserve = _int_env("CHAT_COMPLETION_RESERVE")
    llm_context_size = _int_env("LLM_CONTEXT_SIZE")
    llm_tokenizer_id = _str_env("LLM_TOKENIZER_ID")

    ocr_parser_key = _str_env("OCR_PARSER_KEY").lower()
    chunk_size = _int_env("CHUNK_SIZE")
    chunk_overlap = _int_env("CHUNK_OVERLAP")
    ocr_status_poll_interval = _float_env("OCR_STATUS_POLL_INTERVAL")

    chunking_configs: Tuple[ChunkingConfigSpec, ...] = (
        ChunkingConfigSpec(
            config_id="chunk-small",
            label="Small window",
            description="Primary retrieval window",
            core_size=chunk_size,
            left_overlap=chunk_overlap,
            right_overlap=chunk_overlap,
            step_size=chunk_size,
        ),
    )

    ocr_module_url = _str_env("OCR_MODULE_URL").rstrip("/")
    llm_control_url = _str_env("LLM_CONTROL_URL")
    ocr_control_url = _str_env("OCR_CONTROL_URL")

    diagnostics_url = _str_env("DIAGNOSTICS_URL").rstrip("/")

    min_context_similarity = _float_env("MIN_CONTEXT_SIMILARITY")
    agentic_max_subqueries = max(1, _int_env("AGENTIC_MAX_SUBQUERIES"))
    agentic_short_doc_chunk_limit = _int_env("AGENTIC_SHORT_DOC_CHUNK_LIMIT")
    agentic_short_doc_token_limit = _int_env("AGENTIC_SHORT_DOC_TOKEN_LIMIT")
    agentic_expansion_chunks_before = _int_env("AGENTIC_EXPANSION_CHUNKS_BEFORE")
    agentic_expansion_chunks_after = _int_env("AGENTIC_EXPANSION_CHUNKS_AFTER")
    agentic_max_expanded_chunks = _int_env("AGENTIC_MAX_EXPANDED_CHUNKS")
    agentic_expanded_char_multiplier = _int_env("AGENTIC_EXPANDED_CHAR_MULTIPLIER")
    agentic_min_expanded_chars = _int_env("AGENTIC_MIN_EXPANDED_CHARS")
    agentic_max_expanded_chars = _int_env("AGENTIC_MAX_EXPANDED_CHARS")
    frontend_port = _str_env("FRONTEND_PORT")
    return AppSettings(
        data_dir=data_dir,
        index_dir=index_dir,
        doc_store_path=doc_store_path,
        ocr_parser_key=ocr_parser_key,
        ocr_module_url=ocr_module_url,
        ocr_module_timeout=_float_env("OCR_MODULE_TIMEOUT"),
        chat_context_window=llm_context_size,
        chat_completion_max_tokens=chat_completion_max_tokens,
        chat_completion_reserve=chat_completion_reserve,
        min_context_similarity=min_context_similarity,
        min_semantic_similarity=min_context_similarity,
        min_keyword_score=_float_env("MIN_KEYWORD_SCORE"),
        no_context_response="I couldn't find relevant information for that in the available documents.",
        system_prompt=(
            "You are a retrieval-augmented assistant. Answer strictly using the provided document snippets and cite them as [source N]. "
            "When the snippets contain only partial information, summarize what they do cover and note any gaps; do not invent details. "
            "Only state that no relevant information exists when no snippets are supplied."
        ),
        continue_prompt=(
            "Continue the previous answer to the user's last question. "
            "Resume exactly where it stopped without repeating earlier content."
        ),
        completed_doc_statuses={s.strip().lower() for s in ("processed", "done", "completed", "ready")},
        frontend_origin=f"http://localhost:{frontend_port}",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_config_small_id="chunk-small",
        chunking_configs=chunking_configs,
        llm_base_url=_str_env("LLM_BASE_URL"),
        llm_api_key=_str_env("LLM_API_KEY"),
        llm_model=_str_env("LLM_MODEL"),
        llm_tokenizer_id=llm_tokenizer_id,
        llm_temperature=_float_env("LLM_TEMPERATURE"),
        llm_top_p=_float_env("LLM_TOP_P"),
        llm_top_k=_int_env("LLM_TOP_K"),
        llm_min_p=_float_env("LLM_MIN_P"),
        llm_repeat_penalty=_float_env("LLM_REPEAT_PENALTY"),
        ocr_status_poll_interval=ocr_status_poll_interval,
        llm_control_url=llm_control_url.strip() or None,
        ocr_control_url=ocr_control_url.strip() or None,
        gpu_phase_timeout=_float_env("GPU_PHASE_TIMEOUT"),
        llm_ready_timeout=_float_env("LLM_READY_TIMEOUT"),
        diagnostics_url=diagnostics_url or None,
        embedding_tokenizer_id=_str_env("EMBEDDING_TOKENIZER_ID"),
        agentic_max_subqueries=agentic_max_subqueries,
        agentic_short_doc_chunk_limit=agentic_short_doc_chunk_limit,
        agentic_short_doc_token_limit=agentic_short_doc_token_limit,
        agentic_expansion_chunks_before=agentic_expansion_chunks_before,
        agentic_expansion_chunks_after=agentic_expansion_chunks_after,
        agentic_max_expanded_chunks=agentic_max_expanded_chunks,
        agentic_expanded_char_multiplier=agentic_expanded_char_multiplier,
        agentic_min_expanded_chars=agentic_min_expanded_chars,
        agentic_max_expanded_chars=agentic_max_expanded_chars,
    )
