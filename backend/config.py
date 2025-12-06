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


def _int_env(name: str, default: str) -> int:
    return int(os.environ.get(name, default) or default)


def _float_env(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw or default)
    except (TypeError, ValueError):
        return float(default)


def _str_env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip()


def load_settings() -> AppSettings:
    data_dir = Path(os.environ.get("DATA_DIR", "/app_data/docs"))
    index_dir = Path(os.environ.get("INDEX_DIR", "/app_data/runtime"))
    doc_store_path = Path(os.environ.get("DOC_STORE_PATH") or (data_dir / "rag_meta.db"))

    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    chat_completion_max_tokens = _int_env("CHAT_COMPLETION_MAX_TOKENS", "2048")
    chat_completion_reserve = _int_env("CHAT_COMPLETION_RESERVE", str(chat_completion_max_tokens))
    llm_context_size = _int_env("LLM_CONTEXT_SIZE", "10000")
    llm_tokenizer_id = _str_env("LLM_TOKENIZER_ID", "Qwen/Qwen3-30B-A3B")

    ocr_parser_key = _str_env("OCR_PARSER_KEY", "mineru").lower()
    chunk_size = _int_env("CHUNK_SIZE", "200")
    chunk_overlap = _int_env("CHUNK_OVERLAP", "60")
    ocr_status_poll_interval = _float_env("OCR_STATUS_POLL_INTERVAL", "5")

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

    ocr_module_url = _str_env("OCR_MODULE_URL", "http://ocr-module:8000").rstrip("/")
    llm_control_url = _str_env("LLM_CONTROL_URL")
    if not llm_control_url:
        llm_endpoint = (os.environ.get("LLM_ENDPOINT") or "llm_big").strip() or "llm_big"
        llm_control_url = f"http://{llm_endpoint}:9000/control"
    ocr_control_url = _str_env("OCR_CONTROL_URL", f"{ocr_module_url}/control")

    diagnostics_url = _str_env("DIAGNOSTICS_URL", "http://diagnostics:9001").rstrip("/")

    return AppSettings(
        data_dir=data_dir,
        index_dir=index_dir,
        doc_store_path=doc_store_path,
        ocr_parser_key=ocr_parser_key,
        ocr_module_url=ocr_module_url,
        ocr_module_timeout=_float_env("OCR_MODULE_TIMEOUT", "120"),
        chat_context_window=llm_context_size,
        chat_completion_max_tokens=chat_completion_max_tokens,
        chat_completion_reserve=chat_completion_reserve,
        min_context_similarity=float(os.environ.get("MIN_CONTEXT_SIMILARITY", "0.35") or 0.35),
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
        frontend_origin=f"http://localhost:{os.environ.get('FRONTEND_PORT', '5173')}",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_config_small_id="chunk-small",
        chunking_configs=chunking_configs,
        llm_base_url=_str_env("LLM_BASE_URL"),
        llm_api_key=_str_env("LLM_API_KEY"),
        llm_model=_str_env("LLM_MODEL", "default"),
        llm_tokenizer_id=llm_tokenizer_id,
        llm_temperature=_float_env("LLM_TEMPERATURE", "0.2"),
        llm_top_p=_optional_float_env("LLM_TOP_P"),
        llm_top_k=_optional_int_env("LLM_TOP_K"),
        llm_min_p=_optional_float_env("LLM_MIN_P"),
        llm_repeat_penalty=_optional_float_env("LLM_REPEAT_PENALTY"),
        ocr_status_poll_interval=ocr_status_poll_interval,
        llm_control_url=llm_control_url.strip() or None,
        ocr_control_url=ocr_control_url.strip() or None,
        gpu_phase_timeout=_float_env("GPU_PHASE_TIMEOUT", "60"),
        llm_ready_timeout=_float_env("LLM_READY_TIMEOUT", "180"),
        diagnostics_url=diagnostics_url or None,
        embedding_tokenizer_id=_str_env("EMBEDDING_TOKENIZER_ID", "nomic-ai/nomic-embed-text-v1.5"),
    )


def _optional_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _optional_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
