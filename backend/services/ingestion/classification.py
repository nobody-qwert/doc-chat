from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from fastapi import HTTPException

from .context import gpu_phase_manager, settings
from .taxonomy import (
    describe_subcategories,
    describe_top_level_categories,
    get_subcategories,
    get_top_level_category,
    get_top_level_categories,
    get_subcategory,
)

try:
    from ...token_utils import truncate_text_to_tokens
except ImportError:  # pragma: no cover
    from token_utils import truncate_text_to_tokens  # type: ignore

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT_TOKEN_RESERVE = 800
MIN_CLASSIFICATION_TOKENS = 512
VALID_TOP_LEVEL_IDS = {entry["id"] for entry in get_top_level_categories()}


def _classification_token_limit() -> int:
    context_tokens = max(settings.chat_context_window, CLASSIFICATION_PROMPT_TOKEN_RESERVE + MIN_CLASSIFICATION_TOKENS)
    return max(MIN_CLASSIFICATION_TOKENS, context_tokens - CLASSIFICATION_PROMPT_TOKEN_RESERVE)


def _clean_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit and len(text) > limit:
        return text[:limit]
    return text


def _prepare_classification_text(text: str) -> str:
    token_limit = _classification_token_limit()
    truncated = truncate_text_to_tokens(
        text,
        token_limit,
        tokenizer_id=settings.llm_tokenizer_id,
    )
    if truncated:
        return truncated.strip()
    approx_chars = token_limit * 4
    return _clean_text(text, approx_chars)


def _extract_json(payload: str) -> Dict[str, Any]:
    payload = (payload or "").strip()
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _normalize_confidence(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return normalized or None


async def classify_document_text(
    *,
    doc_hash: str,
    text: str,
    document_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clean_text = _prepare_classification_text(text)
    if not clean_text:
        raise HTTPException(status_code=400, detail="Cannot classify empty text")

    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        raise HTTPException(status_code=503, detail="LLM is not configured for classification")

    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="OpenAI dependency missing") from exc

    await gpu_phase_manager.ensure_llm_ready()

    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    filename = document_name or doc_hash
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    top_level_options = describe_top_level_categories()

    l1_messages = [
        {
            "role": "system",
            "content": (
                "You are a document classifier. Choose exactly one top-level category ID from the list.\n"
                "If uncertain, pick 'miscellaneous'. Respond with a JSON object containing "
                "'top_level_category_id', 'confidence', and 'reasoning'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify this document into ONE top-level category.\n\n"
                f"Available categories:\n{top_level_options}\n\n"
                f"Filename: {filename}\n"
                f"Metadata: {metadata_json}\n\n"
                f"Text:\n{clean_text}"
            ),
        },
    ]

    try:
        l1_response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=l1_messages,
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:  # pragma: no cover - passthrough
        logger.exception("L1 classification failed for %s: %s", doc_hash, exc)
        raise HTTPException(status_code=502, detail=f"L1 classification failed: {exc}") from exc

    if not l1_response.choices:
        raise HTTPException(status_code=500, detail="L1 classification returned no choices")
    l1_content = (l1_response.choices[0].message.content or "").strip()
    l1_json = _extract_json(l1_content)
    l1_id = str(l1_json.get("top_level_category_id") or "").strip()
    if not l1_id or l1_id not in VALID_TOP_LEVEL_IDS:
        l1_id = "miscellaneous"
    l1_entry = get_top_level_category(l1_id) or get_top_level_category("miscellaneous")
    l1_confidence = _normalize_confidence(l1_json.get("confidence"))
    l1_reason = l1_json.get("reasoning") or l1_json.get("reason")

    subcategories = get_subcategories(l1_id)
    l2_result: Dict[str, Any] = {}
    l2_raw = ""
    if subcategories:
        allowed = describe_subcategories(l1_id)
        l2_messages = [
            {
                "role": "system",
                "content": (
                    "You are a document classifier. You already know the top-level category. "
                    "Choose exactly one subcategory ID from the provided list. "
                    "Respond with a JSON object containing 'subcategory_id', 'confidence', and 'reasoning'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The document has top-level category (L1): {l1_id}\n"
                    f"Allowed subcategory IDs:\n{allowed}\n\n"
                    "Pick EXACTLY ONE subcategory.\n\n"
                    f"Filename: {filename}\n"
                    f"Metadata: {metadata_json}\n\n"
                    f"Text:\n{clean_text}"
                ),
            },
        ]
        try:
            l2_response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=l2_messages,
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as exc:  # pragma: no cover - passthrough
            logger.exception("L2 classification failed for %s: %s", doc_hash, exc)
            raise HTTPException(status_code=502, detail=f"L2 classification failed: {exc}") from exc
        if not l2_response.choices:
            raise HTTPException(status_code=500, detail="L2 classification returned no choices")
        l2_raw = (l2_response.choices[0].message.content or "").strip()
        l2_result = _extract_json(l2_raw)

    l2_id = str(l2_result.get("subcategory_id") or "").strip()
    l2_entry = get_subcategory(l1_id, l2_id)
    if not l2_entry and subcategories:
        fallback = next((s for s in subcategories if s["id"].startswith("other_")), subcategories[-1])
        l2_entry = fallback
        l2_id = fallback["id"]

    l2_confidence = _normalize_confidence(l2_result.get("confidence"))
    l2_reason = l2_result.get("reasoning") or l2_result.get("reason")

    result_payload = {
        "doc_hash": doc_hash,
        "l1_id": l1_entry["id"] if l1_entry else l1_id,
        "l1_name": l1_entry.get("name") if l1_entry else l1_id,
        "l2_id": (l2_entry or {}).get("id"),
        "l2_name": (l2_entry or {}).get("name"),
        "l1_confidence": l1_confidence,
        "l2_confidence": l2_confidence,
        "l1_reason": l1_reason,
        "l2_reason": l2_reason,
        "model": settings.llm_model,
        "raw_response": {
            "l1": l1_json or {},
            "l1_raw": l1_content,
            "l2": l2_result or {},
            "l2_raw": l2_raw,
        },
    }
    return result_payload
