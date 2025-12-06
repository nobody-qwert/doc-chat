from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Optional, Sequence

AutoTokenizer: Optional[Any] = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    cleaned = raw.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


logger = logging.getLogger(__name__)

_TOKENIZER_STATS: Dict[str, Dict[str, Any]] = {}
_AUTO_IMPORT_ERROR: Optional[str] = None
_AUTO_IMPORT_LOCK = Lock()
_AUTO_INSTALL_ATTEMPTED = False
_AUTO_INSTALL_TRANSFORMERS = _env_bool("AUTO_INSTALL_TRANSFORMERS", True)
_TRANSFORMERS_SPEC = os.environ.get("TRANSFORMERS_SPEC", "transformers>=4.40")


def _install_transformers_dependency() -> bool:
    global _AUTO_IMPORT_ERROR, _AUTO_INSTALL_ATTEMPTED
    if _AUTO_INSTALL_ATTEMPTED:
        return False
    _AUTO_INSTALL_ATTEMPTED = True
    cmd = [sys.executable or "python", "-m", "pip", "install", _TRANSFORMERS_SPEC]
    logger.warning("AutoTokenizer unavailable; installing %s", _TRANSFORMERS_SPEC)
    try:
        subprocess.check_call(cmd)
        _AUTO_IMPORT_ERROR = None
        logger.info("Successfully installed %s", _TRANSFORMERS_SPEC)
        return True
    except Exception as exc:  # pragma: no cover - logging only
        _AUTO_IMPORT_ERROR = f"auto-install failed: {exc}"
        logger.error("Failed to auto-install transformers: %s", exc)
        return False


def _ensure_auto_tokenizer() -> Optional[Any]:
    global AutoTokenizer, _AUTO_IMPORT_ERROR
    if AutoTokenizer is not None:
        return AutoTokenizer
    with _AUTO_IMPORT_LOCK:
        if AutoTokenizer is not None:
            return AutoTokenizer
        try:
            from transformers import AutoTokenizer as _AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            import_error = str(exc)
            _AUTO_IMPORT_ERROR = import_error
            if not _AUTO_INSTALL_TRANSFORMERS:
                return None
            installed = _install_transformers_dependency()
            if not installed:
                if _AUTO_IMPORT_ERROR and _AUTO_IMPORT_ERROR != import_error:
                    _AUTO_IMPORT_ERROR = f"{import_error}; {_AUTO_IMPORT_ERROR}"
                else:
                    _AUTO_IMPORT_ERROR = import_error
                return None
            try:
                from transformers import AutoTokenizer as _AutoTokenizer  # type: ignore
            except Exception as retry_exc:  # pragma: no cover - optional dependency
                _AUTO_IMPORT_ERROR = str(retry_exc)
                return None
        AutoTokenizer = _AutoTokenizer
        _AUTO_IMPORT_ERROR = None
        return AutoTokenizer


def _ensure_entry(identifier: str) -> Dict[str, Any]:
    entry = _TOKENIZER_STATS.setdefault(
        identifier,
        {
            "loaded": False,
            "error": None,
            "fallback_count": 0,
            "last_fallback_reason": None,
        },
    )
    return entry


def record_tokenizer_fallback(identifier: Optional[str], reason: str) -> None:
    ident = (identifier or "").strip()
    if not ident:
        return
    entry = _ensure_entry(ident)
    entry["fallback_count"] = int(entry.get("fallback_count") or 0) + 1
    entry["last_fallback_reason"] = reason


def tokenizer_diagnostics() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in _TOKENIZER_STATS.items()}


@lru_cache(maxsize=8)
def _load_tokenizer(identifier: str) -> Optional[Any]:
    tokenizer_cls = _ensure_auto_tokenizer()
    if tokenizer_cls is None:
        entry = _ensure_entry(identifier)
        entry["loaded"] = False
        entry["error"] = _AUTO_IMPORT_ERROR or "transformers unavailable"
        return None
    try:
        tokenizer = tokenizer_cls.from_pretrained(identifier, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - logging only
        entry = _ensure_entry(identifier)
        entry["loaded"] = False
        entry["error"] = str(exc)
        logger.warning("Failed to load tokenizer %s: %s", identifier, exc)
        return None
    entry = _ensure_entry(identifier)
    entry["loaded"] = True
    entry["error"] = None
    return tokenizer


def load_tokenizer(tokenizer_id: str) -> Optional[Any]:
    identifier = (tokenizer_id or "").strip()
    if not identifier:
        return None
    return _load_tokenizer(identifier)


def count_text_tokens(text: str, tokenizer_id: str) -> Optional[int]:
    tokenizer = load_tokenizer(tokenizer_id)
    if tokenizer is None:
        return None
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Tokenizer %s failed to encode text: %s", tokenizer_id, exc)
        entry = _ensure_entry((tokenizer_id or "").strip())
        entry["error"] = str(exc)
        return None


def count_chat_tokens(
    messages: Sequence[Dict[str, str]],
    tokenizer_id: str,
    *,
    add_generation_prompt: bool = True,
) -> Optional[int]:
    tokenizer = load_tokenizer(tokenizer_id)
    if tokenizer is None:
        return None
    try:
        input_ids = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return len(input_ids)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Tokenizer %s failed to encode chat template: %s", tokenizer_id, exc)
        entry = _ensure_entry((tokenizer_id or "").strip())
        entry["error"] = str(exc)
        return None


def truncate_text(
    text: str,
    max_tokens: int,
    tokenizer_id: str,
) -> Optional[str]:
    tokenizer = load_tokenizer(tokenizer_id)
    if tokenizer is None:
        return None
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Tokenizer %s failed to encode text for truncation: %s", tokenizer_id, exc)
        entry = _ensure_entry((tokenizer_id or "").strip())
        entry["error"] = str(exc)
        return None
    if len(tokens) <= max_tokens:
        return text
    trimmed = tokens[:max_tokens]
    try:
        return tokenizer.decode(trimmed)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Tokenizer %s failed to decode truncated text: %s", tokenizer_id, exc)
        entry = _ensure_entry((tokenizer_id or "").strip())
        entry["error"] = str(exc)
        return None
