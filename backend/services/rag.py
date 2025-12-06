from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Mapping, Tuple
from uuid import uuid4

import numpy as np
from fastapi import HTTPException

try:
    from ..embeddings import EmbeddingClient
    from ..schemas import AskRequest, AskResponse
    from ..token_utils import estimate_messages_tokens, estimate_tokens, truncate_text_to_tokens
    from ..dependencies import document_store, settings, gpu_phase_manager, embedding_cache
    from ..utils.conversation import (
        build_distilled_query,
        render_context,
        summarize_history,
        trim_history_for_budget,
    )
except ImportError:  # pragma: no cover
    from embeddings import EmbeddingClient  # type: ignore
    from schemas import AskRequest, AskResponse  # type: ignore
    from token_utils import estimate_messages_tokens, estimate_tokens, truncate_text_to_tokens  # type: ignore
    from dependencies import document_store, settings, gpu_phase_manager, embedding_cache  # type: ignore
    from utils.conversation import (  # type: ignore
        build_distilled_query,
        render_context,
        summarize_history,
        trim_history_for_budget,
    )


logger = logging.getLogger(__name__)
_VECTOR_MIN_CANDIDATES = 50
_VECTOR_CANDIDATE_MULTIPLIER = 5


async def _prepare_question_payload(
    req: AskRequest,
    step_emitter: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Prepare the request and build retrieval context. When step_emitter is provided,
    it will be awaited with dict events shaped like {"type":"step","step":{...}}
    for started/done transitions.
    """
    steps: List[Dict[str, Any]] = []
    step_counter = 0

    async def _emit_step(
        state: str,
        name: str,
        kind: str,
        order: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not step_emitter:
            return
        payload: Dict[str, Any] = {
            "type": "step",
            "step": {"name": name, "kind": kind, "order": step_counter if order is None else order, "state": state},
        }
        if extra:
            payload["step"].update(extra)
        await step_emitter(payload)

    def _record_step(name: str, kind: str, start: float, extra: Optional[Dict[str, Any]] = None) -> None:
        duration = max(0.0, time.perf_counter() - start)
        nonlocal step_counter
        entry: Dict[str, Any] = {"name": name, "kind": kind, "duration_seconds": duration, "order": step_counter}
        step_counter += 1
        if extra:
            entry.update(extra)
        steps.append(entry)

    is_continuation = bool(req.continue_last)
    raw_query = (req.query or "").strip()
    if not is_continuation and not raw_query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if is_continuation and not (req.conversation_id or "").strip():
        raise HTTPException(status_code=400, detail="conversation_id is required to continue a response")

    processed = await document_store.count_documents(status="processed")
    if processed == 0:
        raise HTTPException(status_code=400, detail="No processed documents yet")

    conversation_id = (req.conversation_id or "").strip() or str(uuid4())
    existing_conversation = await document_store.get_conversation(conversation_id)
    if not existing_conversation:
        await document_store.create_conversation(conversation_id)

    history_rows = await document_store.fetch_conversation_messages(conversation_id)
    history_messages = [{"role": row["role"], "content": row["content"]} for row in history_rows]

    if is_continuation:
        last_user_message: Optional[Dict[str, str]] = None
        for entry in reversed(history_messages):
            if (entry.get("role") or "").lower() == "user":
                last_user_message = entry
                break
        if not last_user_message:
            raise HTTPException(status_code=400, detail="Cannot continue because no prior user question was found")
        question = (last_user_message.get("content") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Last user question is empty; cannot continue")
    else:
        question = raw_query

    history_summary = summarize_history(history_messages)
    distilled_query = build_distilled_query(history_summary, question)

    emb_client = EmbeddingClient()
    embed_start = time.perf_counter()
    await _emit_step("started", "Embed distilled query", "embedding", order=step_counter)
    distilled_vec = await _embed_query(distilled_query, emb_client)
    _record_step("Embed distilled query", "embedding", embed_start)
    await _emit_step(
        "done",
        "Embed distilled query",
        "embedding",
        order=step_counter - 1,
        extra={"duration_seconds": max(0.0, time.perf_counter() - embed_start)},
    )
    if distilled_vec is None:
        raise HTTPException(status_code=500, detail="Failed to embed query")

    query_vec = distilled_vec
    await gpu_phase_manager.ensure_llm_ready()
    try:
        hyde_start = time.perf_counter()
        await _emit_step("started", "HyDE LLM", "llm", order=step_counter)
        hyde_text = await _generate_hyde_answer(question, history_summary)
        _record_step("HyDE LLM", "llm", hyde_start)
        await _emit_step(
            "done",
            "HyDE LLM",
            "llm",
            order=step_counter - 1,
            extra={"duration_seconds": max(0.0, time.perf_counter() - hyde_start)},
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("HyDE generation raised an unexpected error: %s", exc)
        await _emit_step(
            "done",
            "HyDE LLM",
            "llm",
            order=step_counter,
            extra={"error": str(exc)},
        )
        step_counter += 1
        hyde_text = None

    if hyde_text:
        hyde_embed_start = time.perf_counter()
        await _emit_step("started", "Embed HyDE answer", "embedding", order=step_counter)
        hyde_vec = await _embed_query(hyde_text, emb_client)
        _record_step("Embed HyDE answer", "embedding", hyde_embed_start)
        await _emit_step(
            "done",
            "Embed HyDE answer",
            "embedding",
            order=step_counter - 1,
            extra={"duration_seconds": max(0.0, time.perf_counter() - hyde_embed_start)},
        )
        if hyde_vec is not None:
            try:
                query_vec = (np.asarray(distilled_vec, dtype=np.float32) + np.asarray(hyde_vec, dtype=np.float32)) / 2.0
            except Exception as exc:  # pragma: no cover - fallback
                logger.warning("HyDE vector blend failed: %s", exc)
                query_vec = distilled_vec

    retrieval_start = time.perf_counter()
    await _emit_step("started", "Vector search", "retrieval", order=step_counter)
    scored, chunk_counts = await _score_chunks_with_cache(query_vec, req.top_k)
    _record_step("Vector search", "retrieval", retrieval_start)
    await _emit_step(
        "done",
        "Vector search",
        "retrieval",
        order=step_counter - 1,
        extra={
            "duration_seconds": max(0.0, time.perf_counter() - retrieval_start),
            "retrieved": len(scored),
        },
    )

    chunks_per_doc = dict(chunk_counts)
    top_k = max(1, min(req.top_k, 20))
    filtered_sections = [entry for entry in scored if entry["score"] >= settings.min_context_similarity]
    context_sections = filtered_sections[:top_k]

    docs = await document_store.list_documents()
    docs_by_hash = {doc["doc_hash"]: doc for doc in docs}

    if not settings.llm_base_url or not settings.llm_api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_BASE_URL and LLM_API_KEY must be set",
        )

    prompt_token_limit = min(
        max(512, settings.chat_context_window - settings.chat_completion_reserve),
        settings.chat_context_window - 16,
    )
    if prompt_token_limit <= 0:
        raise HTTPException(status_code=500, detail="Context window too small for configured reserve")

    trimmed_history, history_truncated, _ = trim_history_for_budget(
        history_messages,
        tokenizer_id=settings.llm_tokenizer_id,
        token_limit=prompt_token_limit,
        system_prompt=settings.system_prompt,
    )

    history_for_prompt = list(trimmed_history)
    user_message_for_llm = (
        f"{settings.continue_prompt}\n\nOriginal question: {question}"
        if is_continuation
        else question
    )
    context_truncated = False

    while True:
        context_text = render_context(context_sections)
        context_content = (
            "Use only the following retrieved document snippets as context. "
            "Cite them as [source N].\n"
            + (context_text if context_text else "(No retrieved snippets available.)")
        )
        combined_user_message = f"{context_content}\n\nQuestion:\n{user_message_for_llm}"
        messages = [
            {"role": "system", "content": settings.system_prompt},
            *history_for_prompt,
            {"role": "user", "content": combined_user_message},
        ]
        prompt_tokens = _estimate_prompt_tokens(messages)
        if prompt_tokens <= prompt_token_limit:
            break
        if history_for_prompt:
            history_for_prompt.pop(0)
            history_truncated = True
            continue
        if context_sections:
            context_sections.pop()
            context_truncated = True
            continue
        without_user = messages[:-1]
        remaining = prompt_token_limit - _estimate_prompt_tokens(without_user)
        if remaining <= 0:
            raise HTTPException(status_code=500, detail="Prompt exceeds available context even after trimming")
        truncated = truncate_text_to_tokens(
            question,
            remaining,
            tokenizer_id=settings.llm_tokenizer_id,
        )
        if not truncated:
            truncated = question[:200]
        if truncated == user_message_for_llm:
            raise HTTPException(status_code=500, detail="Unable to fit prompt within context window")
        user_message_for_llm = truncated
        context_truncated = True
    context_tokens_used = prompt_tokens
    context_usage_ratio = min(1.0, context_tokens_used / float(settings.chat_context_window))

    final_context_text = render_context(context_sections)
    final_context_content = (
        "Use only the following retrieved document snippets as context. "
        "Cite them as [source N].\n"
        + (final_context_text if final_context_text else "(No retrieved snippets available.)")
    )
    messages = [
        {"role": "system", "content": settings.system_prompt},
        *history_for_prompt,
        {"role": "user", "content": f"{final_context_content}\n\nQuestion:\n{user_message_for_llm}"},
    ]

    return {
        "is_continuation": is_continuation,
        "question": question,
        "conversation_id": conversation_id,
        "history_messages": history_messages,
        "history_for_prompt": history_for_prompt,
        "history_truncated": history_truncated,
        "context_sections": context_sections,
        "retrieval_sections": filtered_sections,
        "context_truncated": context_truncated,
        "context_tokens_used": context_tokens_used,
        "context_usage_ratio": context_usage_ratio,
        "docs_by_hash": docs_by_hash,
        "chunks_per_doc": chunks_per_doc,
        "final_context_content": final_context_content,
        "messages": messages,
        "user_message_for_llm": user_message_for_llm,
        "steps": steps,
    }


async def ask_question(req: AskRequest) -> AskResponse:
    prepared = await _prepare_question_payload(req)
    is_continuation: bool = prepared["is_continuation"]
    question: str = prepared["question"]
    conversation_id: str = prepared["conversation_id"]
    history_messages: List[Dict[str, str]] = prepared["history_messages"]
    history_for_prompt: List[Dict[str, str]] = prepared["history_for_prompt"]
    history_truncated: bool = prepared["history_truncated"]
    context_sections: List[Dict[str, Any]] = prepared["context_sections"]
    context_truncated: bool = prepared["context_truncated"]
    retrieval_sections: List[Dict[str, Any]] = prepared.get("retrieval_sections", context_sections)
    context_tokens_used: int = prepared["context_tokens_used"]
    context_usage_ratio: float = prepared["context_usage_ratio"]
    docs_by_hash: Dict[str, Any] = prepared["docs_by_hash"]
    chunks_per_doc: Dict[str, int] = prepared["chunks_per_doc"]
    final_context_content: str = prepared["final_context_content"]
    messages: List[Dict[str, str]] = prepared["messages"]
    user_message_for_llm: str = prepared["user_message_for_llm"]
    steps: List[Dict[str, Any]] = prepared.get("steps", [])

    answer = ""
    finish_reason: Optional[str] = None
    time_to_first_token: Optional[float] = None
    generation_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    call_start = time.perf_counter()
    if not context_sections:
        answer = settings.no_context_response
        finish_reason = "no_context"
        time_to_first_token = time.perf_counter() - call_start
        generation_seconds = time_to_first_token
    else:
        await gpu_phase_manager.ensure_llm_ready()
        try:
            from openai import AsyncOpenAI  # type: ignore

            client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
            completion_args = _build_completion_args(
                model=settings.llm_model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=settings.chat_completion_max_tokens,
            )
            response = await client.chat.completions.create(**completion_args)
            if not response.choices:
                raise HTTPException(status_code=500, detail="LLM returned no choices")
            choice = response.choices[0]
            raw_answer = (choice.message.content or "").strip()
            time_to_first_token = time.perf_counter() - call_start
            generation_seconds = time_to_first_token
            answer = raw_answer
            finish_reason = getattr(choice, "finish_reason", None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}")

    sources = _build_sources(context_sections, docs_by_hash, chunks_per_doc)
    retrieval_sources = _build_sources(retrieval_sections, docs_by_hash, chunks_per_doc)

    await _persist_conversation_turn(
        conversation_id=conversation_id,
        history_messages=history_messages,
        question=question,
        answer=answer,
        is_continuation=is_continuation,
    )

    token_count = estimate_tokens(answer, tokenizer_id=settings.llm_tokenizer_id)
    if generation_seconds is None:
        generation_seconds = 0.0
    tokens_per_second = token_count / max(generation_seconds, 1e-6) if token_count and generation_seconds is not None else None
    time_to_first_token = time_to_first_token or generation_seconds

    steps.append(
        {
            "name": "Main LLM",
            "kind": "llm",
            "duration_seconds": generation_seconds,
            "time_to_first_token_seconds": time_to_first_token,
            "tokens_per_second": tokens_per_second,
        }
    )

    needs_follow_up = finish_reason not in (None, "stop")

    return AskResponse(
        answer=answer,
        sources=sources,
        retrieval_sources=retrieval_sources,
        conversation_id=conversation_id,
        context_tokens_used=context_tokens_used,
        context_window_limit=settings.chat_context_window,
        context_usage=context_usage_ratio,
        context_truncated=(history_truncated or context_truncated),
        finish_reason=finish_reason,
        needs_follow_up=needs_follow_up,
        time_to_first_token_seconds=time_to_first_token,
        generation_seconds=generation_seconds,
        tokens_per_second=tokens_per_second,
        steps=steps,
    )


async def warmup_llm() -> Dict[str, Any]:
    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        return {"warmup_complete": False, "error": "LLM env missing"}
    try:
        await gpu_phase_manager.ensure_llm_ready()
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
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


async def _embed_query(text: str, client: EmbeddingClient) -> Any:
    vectors = await client.embed_batch([text])
    return vectors[0] if vectors else None


def _estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    return estimate_messages_tokens(messages, tokenizer_id=settings.llm_tokenizer_id)


async def _generate_hyde_answer(question: str, history_summary: str) -> Optional[str]:
    """Produce a hypothetical ideal answer used for HyDE-style retrieval."""
    question_clean = (question or "").strip()
    if not question_clean:
        return None
    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        return None
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None

    prompt_header = (
        "You draft concise, factual answers solely to improve document retrieval. "
        "Rely only on broad world knowledge and avoid speculation."
    )
    summary_text = history_summary.strip() if history_summary else "(no prior conversation)"
    user_prompt = (
        f"Conversation summary: {summary_text}\n"
        f"Question: {question_clean}\n"
        "Provide a single, information-dense paragraph that would ideally answer the question."
    )
    max_tokens = max(64, min(256, settings.chat_completion_max_tokens))

    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        completion_args = _build_completion_args(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": prompt_header},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=max_tokens,
        )
        response = await client.chat.completions.create(**completion_args)
    except Exception as exc:  # pragma: no cover - passthrough logging
        logger.debug("HyDE LLM call failed: %s", exc)
        return None

    if not response or not getattr(response, "choices", None):
        return None
    choice = response.choices[0]
    content = getattr(choice, "message", None)
    answer_text = (getattr(content, "content", "") or "").strip()
    return answer_text or None


def _build_completion_args(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    # Native OpenAI params
    if settings.llm_top_p is not None:
        args["top_p"] = settings.llm_top_p
    # Non-standard params (llama.cpp/OpenAI-compatible servers support these via extra_body)
    extra_body: Dict[str, Any] = {}
    if settings.llm_top_k is not None:
        extra_body["top_k"] = settings.llm_top_k
    if settings.llm_min_p is not None:
        extra_body["min_p"] = settings.llm_min_p
    if settings.llm_repeat_penalty is not None:
        extra_body["repeat_penalty"] = settings.llm_repeat_penalty
    if extra_body:
        args["extra_body"] = extra_body
    return args


async def stream_question(req: AskRequest):
    async def _event_stream():
        steps: List[Dict[str, Any]] = []
        order = 0
        answer_parts: List[str] = []
        finish_reason: Optional[str] = None
        final_answer = ""
        time_to_first_token: Optional[float] = None
        generation_seconds: Optional[float] = None
        tokens_per_second: Optional[float] = None

        try:
            is_continuation = bool(req.continue_last)
            raw_query = (req.query or "").strip()
            if not is_continuation and not raw_query:
                raise HTTPException(status_code=400, detail="Query must not be empty")
            if is_continuation and not (req.conversation_id or "").strip():
                raise HTTPException(status_code=400, detail="conversation_id is required to continue a response")

            processed = await document_store.count_documents(status="processed")
            if processed == 0:
                raise HTTPException(status_code=400, detail="No processed documents yet")

            conversation_id = (req.conversation_id or "").strip() or str(uuid4())
            existing_conversation = await document_store.get_conversation(conversation_id)
            if not existing_conversation:
                await document_store.create_conversation(conversation_id)

            history_rows = await document_store.fetch_conversation_messages(conversation_id)
            history_messages = [{"role": row["role"], "content": row["content"]} for row in history_rows]

            if is_continuation:
                last_user_message: Optional[Dict[str, str]] = None
                for entry in reversed(history_messages):
                    if (entry.get("role") or "").lower() == "user":
                        last_user_message = entry
                        break
                if not last_user_message:
                    raise HTTPException(status_code=400, detail="Cannot continue because no prior user question was found")
                question = (last_user_message.get("content") or "").strip()
                if not question:
                    raise HTTPException(status_code=400, detail="Last user question is empty; cannot continue")
            else:
                question = raw_query

            history_summary = summarize_history(history_messages)
            distilled_query = build_distilled_query(history_summary, question)

            emb_client = EmbeddingClient()

            # Step 1: embed distilled query
            embed_start = time.perf_counter()
            yield _json_line({"type": "step", "step": {"name": "Embed distilled query", "kind": "embedding", "order": order, "state": "started"}})
            distilled_vec = await _embed_query(distilled_query, emb_client)
            if distilled_vec is None:
                raise HTTPException(status_code=500, detail="Failed to embed query")
            duration = max(0.0, time.perf_counter() - embed_start)
            steps.append({"name": "Embed distilled query", "kind": "embedding", "duration_seconds": duration, "order": order})
            yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
            order += 1

            await gpu_phase_manager.ensure_llm_ready()

            # Step 2: HyDE LLM
            hyde_text: Optional[str]
            try:
                hyde_start = time.perf_counter()
                yield _json_line({"type": "step", "step": {"name": "HyDE LLM", "kind": "llm", "order": order, "state": "started"}})
                hyde_text = await _generate_hyde_answer(question, history_summary)
                duration = max(0.0, time.perf_counter() - hyde_start)
                steps.append({"name": "HyDE LLM", "kind": "llm", "duration_seconds": duration, "order": order})
                yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("HyDE generation raised an unexpected error: %s", exc)
                duration = max(0.0, time.perf_counter() - hyde_start)
                steps.append({"name": "HyDE LLM", "kind": "llm", "duration_seconds": duration, "order": order, "error": str(exc)})
                yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
                hyde_text = None
            order += 1

            # Step 3: embed HyDE answer (if available)
            if hyde_text:
                hyde_embed_start = time.perf_counter()
                yield _json_line({"type": "step", "step": {"name": "Embed HyDE answer", "kind": "embedding", "order": order, "state": "started"}})
                hyde_vec = await _embed_query(hyde_text, emb_client)
                duration = max(0.0, time.perf_counter() - hyde_embed_start)
                steps.append({"name": "Embed HyDE answer", "kind": "embedding", "duration_seconds": duration, "order": order})
                yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
                order += 1
                if hyde_vec is not None:
                    try:
                        query_vec = (np.asarray(distilled_vec, dtype=np.float32) + np.asarray(hyde_vec, dtype=np.float32)) / 2.0
                    except Exception as exc:  # pragma: no cover - fallback
                        logger.warning("HyDE vector blend failed: %s", exc)
                        query_vec = distilled_vec
                else:
                    query_vec = distilled_vec
            else:
                query_vec = distilled_vec

            # Step 4: vector search
            retrieval_start = time.perf_counter()
            yield _json_line({"type": "step", "step": {"name": "Vector search", "kind": "retrieval", "order": order, "state": "started"}})
            scored, chunk_counts = await _score_chunks_with_cache(query_vec, req.top_k)
            duration = max(0.0, time.perf_counter() - retrieval_start)
            steps.append({"name": "Vector search", "kind": "retrieval", "duration_seconds": duration, "order": order, "retrieved": len(scored)})
            yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
            order += 1

            chunks_per_doc = dict(chunk_counts)
            docs = await document_store.list_documents()
            docs_by_hash = {doc["doc_hash"]: doc for doc in docs}

            top_k = max(1, min(req.top_k, 20))
            filtered_sections = [entry for entry in scored if entry["score"] >= settings.min_context_similarity]
            context_sections = filtered_sections[:top_k]

            if not settings.llm_base_url or not settings.llm_api_key:
                raise HTTPException(status_code=500, detail="LLM_BASE_URL and LLM_API_KEY must be set")

            prompt_token_limit = min(
                max(512, settings.chat_context_window - settings.chat_completion_reserve),
                settings.chat_context_window - 16,
            )
            if prompt_token_limit <= 0:
                raise HTTPException(status_code=500, detail="Context window too small for configured reserve")

            trimmed_history, history_truncated, _ = trim_history_for_budget(
                history_messages,
                tokenizer_id=settings.llm_tokenizer_id,
                token_limit=prompt_token_limit,
                system_prompt=settings.system_prompt,
            )

            history_for_prompt = list(trimmed_history)
            user_message_for_llm = (
                f"{settings.continue_prompt}\n\nOriginal question: {question}"
                if is_continuation
                else question
            )
            context_truncated = False

            # Optional: Build prompt step
            build_prompt_start = time.perf_counter()
            yield _json_line({"type": "step", "step": {"name": "Build prompt", "kind": "prompt", "order": order, "state": "started"}})

            while True:
                context_text = render_context(context_sections)
                context_content = (
                    "Use only the following retrieved document snippets as context. "
                    "Cite them as [source N].\n"
                    + (context_text if context_text else "(No retrieved snippets available.)")
                )
                combined_user_message = f"{context_content}\n\nQuestion:\n{user_message_for_llm}"
                messages = [
                    {"role": "system", "content": settings.system_prompt},
                    *history_for_prompt,
                    {"role": "user", "content": combined_user_message},
                ]
                prompt_tokens = _estimate_prompt_tokens(messages)
                if prompt_tokens <= prompt_token_limit:
                    break
                if history_for_prompt:
                    history_for_prompt.pop(0)
                    history_truncated = True
                    continue
                if context_sections:
                    context_sections.pop()
                    context_truncated = True
                    continue
                without_user = messages[:-1]
                remaining = prompt_token_limit - _estimate_prompt_tokens(without_user)
                if remaining <= 0:
                    raise HTTPException(status_code=500, detail="Prompt exceeds available context even after trimming")
                truncated = truncate_text_to_tokens(
                    question,
                    remaining,
                    tokenizer_id=settings.llm_tokenizer_id,
                )
                if not truncated:
                    truncated = question[:200]
                if truncated == user_message_for_llm:
                    raise HTTPException(status_code=500, detail="Unable to fit prompt within context window")
                user_message_for_llm = truncated
                context_truncated = True
            context_tokens_used = prompt_tokens
            context_usage_ratio = min(1.0, context_tokens_used / float(settings.chat_context_window))

            final_context_text = render_context(context_sections)
            final_context_content = (
                "Use only the following retrieved document snippets as context. "
                "Cite them as [source N].\n"
                + (final_context_text if final_context_text else "(No retrieved snippets available.)")
            )
            messages = [
                {"role": "system", "content": settings.system_prompt},
                *history_for_prompt,
                {"role": "user", "content": f"{final_context_content}\n\nQuestion:\n{user_message_for_llm}"},
            ]

            prompt_build_duration = max(0.0, time.perf_counter() - build_prompt_start)
            steps.append({"name": "Build prompt", "kind": "prompt", "duration_seconds": prompt_build_duration, "order": order})
            yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
            order += 1

            sources = _build_sources(context_sections, docs_by_hash, chunks_per_doc)
            retrieval_sources = _build_sources(filtered_sections, docs_by_hash, chunks_per_doc)

            call_start = time.perf_counter()

            if not context_sections:
                final_answer = settings.no_context_response
                finish_reason = "no_context"
                time_to_first_token = time.perf_counter() - call_start
                generation_seconds = time_to_first_token
                yield _json_line({"type": "token", "content": final_answer})
            else:
                await gpu_phase_manager.ensure_llm_ready()
                from openai import AsyncOpenAI  # type: ignore

                client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
                yield _json_line({"type": "step", "step": {"name": "Main LLM", "kind": "llm", "order": order, "state": "started"}})
                completion_args = _build_completion_args(
                    model=settings.llm_model,
                    messages=messages,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.chat_completion_max_tokens,
                    stream=True,
                )
                stream = await client.chat.completions.create(**completion_args)
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                    delta = getattr(choice, "delta", None)
                    piece = ""
                    if delta is not None:
                        piece = getattr(delta, "content", None) or ""
                    if not piece:
                        message_obj = getattr(choice, "message", None)
                        piece = getattr(message_obj, "content", None) or ""
                    if piece:
                        answer_parts.append(piece)
                        if time_to_first_token is None:
                            time_to_first_token = time.perf_counter() - call_start
                        yield _json_line({"type": "token", "content": piece})
                raw_answer = "".join(answer_parts).strip()
                final_answer = raw_answer

            if not final_answer:
                final_answer = "".join(answer_parts).strip()
            total_time = time.perf_counter() - call_start
            time_to_first_token = time_to_first_token or total_time
            generation_seconds = max(0.0, total_time - time_to_first_token)
            token_count = estimate_tokens(final_answer, tokenizer_id=settings.llm_tokenizer_id)
            tokens_per_second = token_count / max(generation_seconds, 1e-6) if generation_seconds is not None else None
            steps.append(
                {
                    "name": "Main LLM",
                    "kind": "llm",
                    "duration_seconds": generation_seconds,
                    "time_to_first_token_seconds": time_to_first_token,
                    "tokens_per_second": tokens_per_second,
                    "order": order,
                }
            )
            yield _json_line({"type": "step", "step": {**steps[-1], "state": "done"}})
            order += 1

            needs_follow_up = finish_reason not in (None, "stop")
            await _persist_conversation_turn(
                conversation_id=conversation_id,
                history_messages=history_messages,
                question=question,
                answer=final_answer,
                is_continuation=is_continuation,
            )
            payload = {
                "type": "final",
                "answer": final_answer,
                "conversation_id": conversation_id,
                "sources": sources,
                "retrieval_sources": retrieval_sources,
                "context_tokens_used": context_tokens_used,
                "context_window_limit": settings.chat_context_window,
                "context_usage": context_usage_ratio,
                "context_truncated": history_truncated or context_truncated,
                "finish_reason": finish_reason,
                "needs_follow_up": needs_follow_up,
                "time_to_first_token_seconds": time_to_first_token,
                "generation_seconds": generation_seconds,
                "tokens_per_second": tokens_per_second,
                "steps": steps,
            }
            yield _json_line(payload)
        except Exception as exc:
            logger.exception("Streaming ask failed: %s", exc)
            yield _json_line({"type": "error", "error": str(exc)})

    return _event_stream()


async def _score_chunks_with_cache(
    query_vec: np.ndarray,
    requested_top_k: int,
) -> Tuple[List[Dict[str, Any]], Mapping[str, int]]:
    snapshot = embedding_cache.snapshot()
    if snapshot.total == 0:
        return [], snapshot.chunks_per_doc
    vector = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if snapshot.dim and vector.shape[0] != snapshot.dim:
        raise HTTPException(status_code=500, detail="Embedding dimension mismatch; please re-ingest documents")
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return [], snapshot.chunks_per_doc
    normalized_query = vector / max(norm, 1e-8)
    scores = snapshot.matrix @ normalized_query
    candidate_goal = max(requested_top_k * _VECTOR_CANDIDATE_MULTIPLIER, _VECTOR_MIN_CANDIDATES)
    candidate_count = min(snapshot.total, candidate_goal)
    if candidate_count <= 0:
        return [], snapshot.chunks_per_doc
    top_indices = _select_top_indices(scores, candidate_count)
    candidate_ids = [snapshot.chunk_ids[idx] for idx in top_indices]
    candidate_scores = [float(scores[idx]) for idx in top_indices]
    if not candidate_ids:
        return [], snapshot.chunks_per_doc
    chunk_rows = await document_store.fetch_chunks_by_ids(candidate_ids)
    chunk_by_id = {row["chunk_id"]: row for row in chunk_rows}
    scored: List[Dict[str, Any]] = []
    for chunk_id, score_value in zip(candidate_ids, candidate_scores):
        chunk_row = chunk_by_id.get(chunk_id)
        if not chunk_row:
            continue
        chunk_copy = dict(chunk_row)
        config_id_raw = chunk_copy.get("chunk_config_id")
        config_id = str(config_id_raw) if config_id_raw is not None else None
        chunk_copy["scale"] = _scale_for_chunk_config(config_id)
        scored.append({"chunk": chunk_copy, "score": score_value})
    return scored, snapshot.chunks_per_doc


def _select_top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    total = scores.shape[0]
    if count >= total:
        return np.argsort(scores)[::-1]
    top_idx = np.argpartition(scores, -count)[-count:]
    return top_idx[np.argsort(scores[top_idx])[::-1]]


def _scale_for_chunk_config(config_id: Optional[str]) -> Optional[str]:
    if not config_id:
        return None
    if config_id == settings.chunk_config_small_id:
        return "small"
    return None


def _build_sources(
    context_sections: List[Dict[str, Any]],
    docs_by_hash: Dict[str, Any],
    chunks_per_doc: Dict[str, int],
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for entry in context_sections:
        chunk = entry["chunk"]
        doc_hash = chunk["doc_hash"]
        doc = docs_by_hash.get(doc_hash, {})
        total_chunks = chunks_per_doc.get(doc_hash, 0)
        sources.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": entry["score"],
                "doc_hash": doc_hash,
                "order_index": chunk["order_index"],
                "document_name": doc.get("original_name", "unknown"),
                "total_chunks": total_chunks,
                "chunk_text": chunk.get("text", ""),
                "chunk_text_preview": chunk.get("text", "")[:200],
                "parser": chunk.get("parser", settings.ocr_parser_key),
                "scale": chunk.get("scale"),
            }
        )
    return sources


async def _persist_conversation_turn(
    conversation_id: str,
    history_messages: List[Dict[str, str]],
    question: str,
    answer: str,
    is_continuation: bool,
) -> None:
    if not is_continuation:
        user_token_count = estimate_tokens(question, tokenizer_id=settings.llm_tokenizer_id)
        await document_store.append_conversation_message(
            conversation_id,
            role="user",
            content=question,
            token_count=user_token_count,
        )
    assistant_token_count = estimate_tokens(answer, tokenizer_id=settings.llm_tokenizer_id)
    await document_store.append_conversation_message(
        conversation_id,
        role="assistant",
        content=answer,
        token_count=assistant_token_count,
    )

    updated_history = list(history_messages)
    if not is_continuation:
        updated_history.append({"role": "user", "content": question})
    updated_history.append({"role": "assistant", "content": answer})
    summary_text = summarize_history(updated_history)
    await document_store.update_conversation_summary(conversation_id, summary_text)


def _json_line(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"
