from __future__ import annotations

from typing import Any, Dict


async def warmup_llm() -> Dict[str, Any]:
    """
    Warm up the configured OpenAI-compatible LLM endpoint.

    Kept separate from any RAG/chat implementation so the system warmup route
    doesn't depend on unused endpoints.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"warmup_complete": False, "error": f"OpenAI client unavailable: {exc}"}

    try:
        from ..dependencies import gpu_phase_manager, settings
    except ImportError:  # pragma: no cover - script fallback
        from dependencies import gpu_phase_manager, settings  # type: ignore

    if not settings.llm_base_url or not settings.llm_model:
        return {"warmup_complete": False, "error": "LLM env missing"}

    try:
        await gpu_phase_manager.ensure_llm_ready()
        client = AsyncOpenAI(base_url=settings.llm_base_url, api_key="local")
        await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "Warmup"},
                {"role": "user", "content": "Say ready."},
            ],
            max_tokens=4,
            temperature=0,
        )
        return {"warmup_complete": True, "status": "ready"}
    except Exception as exc:
        return {"warmup_complete": False, "error": str(exc)}

