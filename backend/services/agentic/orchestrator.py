"""
Agentic RAG Orchestrator.

Implements the streaming `stream_agentic_answer()` loop along with shared
helpers for evidence collection, pruning, and citation verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from .modes import (
    decompose_query,
    build_initial_plan,
    compose_answer,
    inspect_evidence,
    verify_citations,
    rewrite_semantic_query,
)
from .tools import execute_tool, ToolResult

logger = logging.getLogger(__name__)

INSPECTOR_MAX_ITEMS = 10

@dataclass
class AgenticStep:
    """Record of a single step in the agentic loop."""
    step_number: int
    name: str
    kind: str  # "decompose", "plan", "tool", "rewrite", "inspect", "compose"
    duration_seconds: float = 0.0
    state: str = "done"  # "started", "done", "error"
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


async def stream_agentic_answer(
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    llm_client: Any,
    settings: Any,
    conversation_id: Optional[str] = None,
    max_subqueries: int = 5,
) -> AsyncGenerator[str, None]:
    """
    Streaming version of agentic RAG.
    
    Yields JSON-lines with:
    - {"type": "step", "step": {...}} - Step progress
    - {"type": "token", "content": "..."} - Answer tokens
    - {"type": "final", ...} - Final result metadata
    """
    steps: List[AgenticStep] = []
    evidence: List[Dict[str, Any]] = []
    total_tool_calls = 0
    max_subqueries = max(1, int(max_subqueries or 1))
    model = settings.llm_model
    
    def _emit_step(step: AgenticStep, state: str = "done") -> str:
        step_dict = _step_to_dict(step)
        step_dict["state"] = state
        return json.dumps({"type": "step", "step": step_dict}) + "\n"
    
    # Step 0: Decompose query
    decomp_start = time.perf_counter()
    yield _emit_step(AgenticStep(0, "Decompose Query", "decompose"), "started")
    
    decomp_result = await decompose_query(query, llm_client, model)
    decomp_duration = time.perf_counter() - decomp_start
    
    step = AgenticStep(
        step_number=0,
        name="Decompose Query",
        kind="decompose",
        duration_seconds=decomp_duration,
        details=_llm_step_details(
            {"decomposition": decomp_result.data} if decomp_result.data else None,
            decomp_result,
        ),
    )
    steps.append(step)
    yield _emit_step(step)
    
    if not decomp_result.success:
        decomposition = {"subqueries": [{"query": query}]}
    else:
        decomposition = decomp_result.data or {}

    # Deterministic search plan derived directly from subqueries
    plan_result = build_initial_plan(query, decomposition, max_subqueries=max_subqueries)
    max_subqueries = plan_result.max_subqueries or max_subqueries
    for subplan in plan_result.subquery_plans:
        if not subplan.initial_queries:
            primary = (subplan.subquery or query).strip() or query
            subplan.initial_queries = [primary]
        else:
            primary = (subplan.subquery or query).strip() or query
            if primary:
                subplan.initial_queries[0] = primary
    
    # Step 2: Per-subquery search + inspect (streaming)
    step_num = len(steps)
    subquery_inspector_evidence: List[Dict[str, Any]] = []
    inspector_hit_counter = 0
    inspector_found: Optional[bool] = None
    
    for idx, subplan in enumerate(plan_result.subquery_plans, start=1):
        subquery_label = subplan.subquery or f"Subquery {idx}"
        primary_query = (subplan.initial_queries[0] if subplan.initial_queries else subquery_label).strip() or subquery_label
        subquery_evidence: List[Dict[str, Any]] = []
        search_queries = [primary_query]
        for search_query in search_queries:
            if subplan.strategy in ("keyword", "hybrid"):
                label = "search_text"
                args = {"query": search_query, "top_k": 5}
                yield _emit_step(AgenticStep(step_num, label, "tool"), "started")
                tool_start = time.perf_counter()
                
                text_result = await execute_tool(
                    tool_name="search_text",
                    args=args,
                    document_store=document_store,
                    embedding_client=embedding_client,
                    embedding_cache=embedding_cache,
                    settings=settings,
                )
                total_tool_calls += 1
                tool_duration = time.perf_counter() - tool_start
                
                step = AgenticStep(
                    step_number=step_num,
                    name=label,
                    kind="tool",
                    duration_seconds=tool_duration,
                    details=_tool_step_details("search_text", args, text_result),
                )
                steps.append(step)
                yield _emit_step(step)
                step_num += 1
                
                if text_result.success:
                    evidence.extend(text_result.results)
                    subquery_evidence.extend(text_result.results)
            
            if subplan.strategy in ("semantic", "hybrid"):
                args = {"query": search_query, "top_k": 10}
                if (search_query or "").strip():
                    rewrite_label = "Enhance Semantic Query"
                    yield _emit_step(AgenticStep(step_num, rewrite_label, "rewrite"), "started")
                    rewrite_start = time.perf_counter()
                    rewrite_result = await rewrite_semantic_query(
                        original_query=search_query,
                        user_query=query,
                        llm_client=llm_client,
                        model=model,
                    )
                    rewrite_duration = time.perf_counter() - rewrite_start
                    rewritten_query = rewrite_result.rewritten_query or search_query
                    args["query"] = rewritten_query
                    rewrite_step = AgenticStep(
                        step_number=step_num,
                        name=rewrite_label,
                        kind="rewrite",
                        duration_seconds=rewrite_duration,
                        details=_llm_step_details(
                            {
                                "original_query": search_query,
                                "rewritten_query": rewritten_query,
                            },
                            rewrite_result,
                        ),
                        error=rewrite_result.error if not rewrite_result.success else None,
                    )
                    steps.append(rewrite_step)
                    yield _emit_step(rewrite_step)
                    step_num += 1
                label = "search_semantic"
                yield _emit_step(AgenticStep(step_num, label, "tool"), "started")
                tool_start = time.perf_counter()
                
                semantic_result = await execute_tool(
                    tool_name="search_semantic",
                    args=args,
                    document_store=document_store,
                    embedding_client=embedding_client,
                    embedding_cache=embedding_cache,
                    settings=settings,
                )
                total_tool_calls += 1
                tool_duration = time.perf_counter() - tool_start
                
                step = AgenticStep(
                    step_number=step_num,
                    name=label,
                    kind="tool",
                    duration_seconds=tool_duration,
                    details=_tool_step_details("search_semantic", args, semantic_result),
                )
                steps.append(step)
                yield _emit_step(step)
                step_num += 1
                
                if semantic_result.success:
                    evidence.extend(semantic_result.results)
                    subquery_evidence.extend(semantic_result.results)
        
        if not subquery_evidence:
            continue
        subquery_evidence = _deduplicate_evidence(subquery_evidence)
        inspector_input = _prioritize_full_doc_evidence(subquery_evidence)
        inspector_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        async def _capture_inspector_event(event: Dict[str, Any]) -> None:
            await inspector_queue.put(event)

        inspector_step = AgenticStep(step_num, f"Inspect Evidence ({subquery_label})", "inspect")
        steps.append(inspector_step)
        yield _emit_step(inspector_step, "started")
        inspector_start = time.perf_counter()
        inspector_task = asyncio.create_task(
            inspect_evidence(
                primary_query,
                inspector_input,
                llm_client,
                model,
                max_items=min(INSPECTOR_MAX_ITEMS, len(inspector_input)),
                max_hits=INSPECTOR_MAX_ITEMS,
                progress_callback=_capture_inspector_event,
            )
        )

        next_step_num = step_num + 1
        while True:
            if inspector_task.done() and inspector_queue.empty():
                break
            try:
                event = await asyncio.wait_for(inspector_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            event = dict(event)
            event["subquery"] = subquery_label
            sub_step = _inspector_event_to_step(event, next_step_num)
            steps.append(sub_step)
            yield _emit_step(sub_step)
            next_step_num += 1

        inspector_result = await inspector_task
        inspector_duration = time.perf_counter() - inspector_start
        inspector_step.duration_seconds = inspector_duration
        inspector_step.details = _llm_step_details(
            {
                "subquery": subquery_label,
                "inspected_items": inspector_result.inspected_items if inspector_result else 0,
                "hits_count": len(inspector_result.hits) if inspector_result else 0,
                "inspected_docs": _summarize_inspected_docs(inspector_result.inspected_docs) if inspector_result else None,
            },
            inspector_result,
        )
        inspector_step.error = (
            inspector_result.error
            if inspector_result and not inspector_result.success
            else None
        )
        yield _emit_step(inspector_step)
        step_num = next_step_num

        if inspector_result and inspector_result.success and inspector_result.hits:
            hits_evidence = _inspector_hits_to_evidence(inspector_result.hits)
            for _, hit in enumerate(hits_evidence, 1):
                inspector_hit_counter += 1
                hit["chunk_id"] = f"inspector_{inspector_hit_counter}"
                hit["subquery"] = subquery_label
            subquery_inspector_evidence.extend(hits_evidence)
            inspector_found = True
    
    evidence = _deduplicate_evidence(evidence)
    
    if subquery_inspector_evidence:
        _assign_citation_ids(subquery_inspector_evidence)
        yield _emit_step(AgenticStep(step_num, "Compose Answer (Inspector)", "compose"), "started")
        compose_start = time.perf_counter()
        answer_parts: List[str] = []
        compose_stream = await compose_answer(
            query,
            subquery_inspector_evidence,
            llm_client,
            model,
            stream=True,
            output_preferences=decomposition.get("output_preferences"),
        )
        answer_generator = compose_stream.stream
        async for token in answer_generator:
            answer_parts.append(token)
            yield json.dumps({"type": "token", "content": token}) + "\n"
        compose_duration = time.perf_counter() - compose_start
        full_answer = "".join(answer_parts)
        verified_answer = verify_citations(full_answer, subquery_inspector_evidence)
        step = AgenticStep(
            step_number=step_num,
            name="Compose Answer (Inspector)",
            kind="compose",
            duration_seconds=compose_duration,
            details=_llm_step_details({}, {
                "prompt": compose_stream.prompt,
                "prompt_messages": compose_stream.prompt_messages,
                "raw_response": full_answer,
            }),
        )
        steps.append(step)
        yield _emit_step(step)
        yield json.dumps({
            "type": "final",
            "answer": verified_answer,
            "needs_clarification": False,
            "sources": _build_sources(subquery_inspector_evidence),
            "conversation_id": conversation_id,
            "steps": [_step_to_dict(s) for s in steps],
            "total_tool_calls": total_tool_calls,
            "evidence_count": len(evidence),
            "finish_reason": "inspector",
            "inspector_found": True,
        }) + "\n"
        return
    
    # Inspector pass
    if evidence:
        inspector_input = _prioritize_full_doc_evidence(evidence)
        inspector_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        async def _capture_inspector_event(event: Dict[str, Any]) -> None:
            await inspector_queue.put(event)

        inspector_step = AgenticStep(step_num, "Inspect Evidence", "inspect")
        steps.append(inspector_step)
        yield _emit_step(inspector_step, "started")
        inspector_start = time.perf_counter()
        inspector_task = asyncio.create_task(
            inspect_evidence(
                query,
                inspector_input,
                llm_client,
                model,
                max_items=min(INSPECTOR_MAX_ITEMS, len(inspector_input)),
                max_hits=INSPECTOR_MAX_ITEMS,
                progress_callback=_capture_inspector_event,
            )
        )

        next_step_num = step_num + 1
        while True:
            if inspector_task.done() and inspector_queue.empty():
                break
            try:
                event = await asyncio.wait_for(inspector_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            sub_step = _inspector_event_to_step(event, next_step_num)
            steps.append(sub_step)
            yield _emit_step(sub_step)
            next_step_num += 1

        inspector_result = await inspector_task
        inspector_duration = time.perf_counter() - inspector_start
        inspector_step.duration_seconds = inspector_duration
        inspector_step.details = _llm_step_details(
            {
                "inspected_items": inspector_result.inspected_items if inspector_result else 0,
                "hits_count": len(inspector_result.hits) if inspector_result else 0,
                "inspected_docs": _summarize_inspected_docs(inspector_result.inspected_docs) if inspector_result else None,
            },
            inspector_result,
        )
        inspector_step.error = (
            inspector_result.error
            if inspector_result and not inspector_result.success
            else None
        )
        if inspector_result is not None:
            if inspector_result.success:
                inspector_found = bool(inspector_result.hits)
            else:
                inspector_found = None
        yield _emit_step(inspector_step)
        step_num = next_step_num

        if inspector_result and inspector_result.success and inspector_result.hits:
            inspector_evidence = _inspector_hits_to_evidence(inspector_result.hits)
            _assign_citation_ids(inspector_evidence)
            yield _emit_step(AgenticStep(step_num, "Compose Answer (Inspector)", "compose"), "started")
            compose_start = time.perf_counter()
            answer_parts: List[str] = []
            compose_stream = await compose_answer(
                query,
                inspector_evidence,
                llm_client,
                model,
                stream=True,
                output_preferences=decomposition.get("output_preferences"),
            )
            answer_generator = compose_stream.stream
            async for token in answer_generator:
                answer_parts.append(token)
                yield json.dumps({"type": "token", "content": token}) + "\n"
            compose_duration = time.perf_counter() - compose_start
            full_answer = "".join(answer_parts)
            verified_answer = verify_citations(full_answer, inspector_evidence)
            step = AgenticStep(
                step_number=step_num,
                name="Compose Answer (Inspector)",
                kind="compose",
                duration_seconds=compose_duration,
                details=_llm_step_details({}, {
                    "prompt": compose_stream.prompt,
                    "prompt_messages": compose_stream.prompt_messages,
                    "raw_response": full_answer,
                }),
            )
            steps.append(step)
            yield _emit_step(step)
            yield json.dumps({
                "type": "final",
                "answer": verified_answer,
                "needs_clarification": False,
                "sources": _build_sources(inspector_evidence),
                "conversation_id": conversation_id,
                "steps": [_step_to_dict(s) for s in steps],
                "total_tool_calls": total_tool_calls,
                "evidence_count": len(evidence),
                "finish_reason": "inspector",
                "inspector_found": True,
            }) + "\n"
            return
    
    # Prune evidence
    evidence = _prune_evidence(evidence, max_items=15)
    _assign_citation_ids(evidence)
    composer_evidence = [] if inspector_found is False else evidence
    
    # Step 4: Compose answer with streaming
    yield _emit_step(AgenticStep(step_num, "Compose Answer", "compose"), "started")
    compose_start = time.perf_counter()
    
    answer_parts: List[str] = []
    compose_stream = await compose_answer(
        query,
        composer_evidence,
        llm_client,
        model,
        stream=True,
        output_preferences=decomposition.get("output_preferences"),
    )
    answer_generator = compose_stream.stream
    
    async for token in answer_generator:
        answer_parts.append(token)
        yield json.dumps({"type": "token", "content": token}) + "\n"
    
    full_answer = "".join(answer_parts)
    verified_answer = verify_citations(full_answer, composer_evidence)
    
    compose_duration = time.perf_counter() - compose_start
    compose_meta = {
        "prompt": compose_stream.prompt,
        "prompt_messages": compose_stream.prompt_messages,
        "raw_response": full_answer,
    }
    step = AgenticStep(
        step_number=step_num,
        name="Compose Answer",
        kind="compose",
        duration_seconds=compose_duration,
        details=_llm_step_details({}, compose_meta),
    )
    steps.append(step)
    yield _emit_step(step)
    
    # Final result
    yield json.dumps({
        "type": "final",
        "answer": verified_answer,
        "needs_clarification": False,
        "sources": _build_sources(composer_evidence),
        "conversation_id": conversation_id,
        "steps": [_step_to_dict(s) for s in steps],
        "total_tool_calls": total_tool_calls,
        "evidence_count": len(composer_evidence),
        "finish_reason": "complete",
        "inspector_found": inspector_found,
    }) + "\n"




def _deduplicate_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate evidence items by chunk_id."""
    seen = set()
    result = []
    for item in evidence:
        chunk_id = item.get("chunk_id")
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            result.append(item)
        elif not chunk_id:
            result.append(item)
    return result


def _prune_evidence(
    evidence: List[Dict[str, Any]],
    max_items: int = 15,
) -> List[Dict[str, Any]]:
    """Prune evidence to top items by score."""
    # Sort by score descending
    sorted_evidence = sorted(
        evidence,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )
    return sorted_evidence[:max_items]


def _assign_citation_ids(evidence: List[Dict[str, Any]]) -> None:
    """Assign 1-indexed citation IDs to evidence for stable referencing."""
    for idx, item in enumerate(evidence, 1):
        item["citation_id"] = str(idx)


def _build_sources(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build source list for response."""
    sources = []
    for item in evidence:
        sources.append({
            "citation_id": item.get("citation_id"),
            "doc_hash": item.get("doc_hash"),
            "chunk_id": item.get("chunk_id"),
            "document_name": item.get("document_name", "Unknown"),
            "score": item.get("score", 0),
            "text_preview": item.get("text", "")[:200],
            "match_type": item.get("match_type", "unknown"),
        })
    return sources


def _filter_sources_by_ids(sources: List[Dict[str, Any]], ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not ids:
        return sources
    wanted = {i for i in ids if i}
    if not wanted:
        return sources
    filtered = [
        src for src in sources
        if src.get("doc_hash") in wanted or src.get("chunk_id") in wanted
    ]
    return filtered or sources


def _inspector_hits_to_evidence(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert inspector hits into evidence entries for the composer."""
    evidence: List[Dict[str, Any]] = []
    seen_docs: set[str] = set()
    for idx, hit in enumerate(hits, 1):
        citations = hit.get("citations") or []
        doc_hash = hit.get("doc_hash") or (citations[0] if citations else None)
        doc_name = hit.get("doc_name") or "Unknown Document"
        dedupe_key = doc_hash or doc_name
        if dedupe_key and dedupe_key in seen_docs:
            continue
        if dedupe_key:
            seen_docs.add(dedupe_key)
        parts = []
        quote = hit.get("quote") or ""
        text = f"Quote: {quote}" if quote else ""
        evidence.append({
            "doc_hash": doc_hash,
            "chunk_id": f"inspector_{idx}",
            "order_index": idx - 1,
            "text": text,
            "document_name": doc_name,
            "match_type": "inspector",
        })
    return evidence


def _prioritize_full_doc_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not evidence:
        return evidence
    full_docs: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for item in evidence:
        expanded = item.get("expanded_context") or {}
        is_full = expanded.get("type") == "full_document" or str(item.get("chunk_id", "")).endswith("::full_doc")
        if is_full:
            full_docs.append(item)
        else:
            rest.append(item)
    return full_docs + rest


def _summarize_inspected_docs(
    inspected_docs: Optional[List[Dict[str, Any]]],
    limit: int = 10,
) -> Optional[List[Dict[str, Any]]]:
    if not inspected_docs:
        return None
    summary: List[Dict[str, Any]] = []
    for entry in inspected_docs[:limit]:
        summary.append({
            "doc_hash": entry.get("doc_hash"),
            "doc_name": entry.get("doc_name"),
            "order": entry.get("order"),
        })
    return summary


def _step_to_dict(step: AgenticStep) -> Dict[str, Any]:
    """Convert AgenticStep to dict."""
    result = {
        "name": step.name,
        "kind": step.kind,
        "order": step.step_number,
        "duration_seconds": step.duration_seconds,
        "state": step.state,
    }
    if step.details:
        result["details"] = step.details
    if step.error:
        result["error"] = step.error
    return result


def _llm_step_details(base_details: Optional[Dict[str, Any]], result: Any) -> Optional[Dict[str, Any]]:
    """Augment step details with prompt/response metadata."""
    if result is None and not base_details:
        return None
    details: Dict[str, Any] = dict(base_details or {})
    def _pick(field: str):
        if result is None:
            return None
        if isinstance(result, dict):
            return result.get(field)
        return getattr(result, field, None)

    prompt = _pick("prompt")
    prompt_messages = _pick("prompt_messages")
    raw_response = _pick("raw_response")
    if prompt_messages:
        details["llm_prompt_messages"] = prompt_messages
    if raw_response:
        details["llm_response"] = raw_response
    return details or None


def _summarize_tool_results(results: Optional[List[Dict[str, Any]]], limit: int = 3) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    if not results:
        return summary
    for item in results[:limit]:
        if not item:
            continue
        preview_source = item.get("text") or item.get("chunk_text") or ""
        summary.append({
            "document_name": item.get("document_name", "Unknown"),
            "chunk_id": item.get("chunk_id"),
            "order_index": item.get("order_index"),
            "score": item.get("score"),
            "match_type": item.get("match_type"),
            "preview": preview_source[:240],
        })
    return summary


def _inspector_event_to_step(event: Dict[str, Any], step_number: int) -> AgenticStep:
    """Convert a per-document inspector callback into a detailed step entry."""
    doc_name = event.get("doc_name") or "Document"
    doc_hash = event.get("doc_hash")
    index = event.get("index")
    duration = float(event.get("duration") or 0.0)
    found = event.get("found")
    status = "error" if event.get("error") else ("found" if found else "not_found")
    if index is not None:
        label = f"Inspect Evidence · Item {index} ({doc_name})"
    else:
        label = f"Inspect Evidence · {doc_name}"
    
    detail_payload: Dict[str, Any] = {
        "doc_name": doc_name,
        "doc_hash": doc_hash,
        "status": status,
        "match_type": event.get("match_type"),
    }
    if index is not None:
        detail_payload["inspected_index"] = index
    if "parsed_result" in event and event.get("parsed_result") is not None:
        detail_payload["parsed_result"] = event.get("parsed_result")
    if found is not None:
        detail_payload["found"] = bool(found)
    base_llm_result = {
        "prompt": event.get("prompt"),
        "prompt_messages": event.get("prompt_messages"),
        "raw_response": event.get("raw_response"),
    }
    details = _llm_step_details(detail_payload, base_llm_result) or detail_payload
    
    return AgenticStep(
        step_number=step_number,
        name=label,
        kind="inspect",
        duration_seconds=duration,
        details=details,
        error=event.get("error"),
    )


def _tool_step_details(tool_name: str, tool_args: Dict[str, Any], result: ToolResult) -> Dict[str, Any]:
    """Normalize tool metadata for UI display."""
    details: Dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_return_count": result.total_found,
    }
    summary = _summarize_tool_results(result.results)
    if summary:
        details["tool_results"] = summary
    if result.error:
        details["tool_error"] = result.error
    return details
def _augmented_initial_queries(
    original_query: str,
    initial_queries: List[str],
    decomposition: Dict[str, Any],
    max_queries: int = 6,
) -> List[str]:
    """Expand initial queries to cover per-entity subproblems."""
    queries: List[str] = []
    seen = set()
    
    def _add(query: str):
        normalized = (query or "").strip()
        if not normalized:
            return
        if normalized.lower() in seen:
            return
        seen.add(normalized.lower())
        queries.append(normalized)
    
    for q in initial_queries:
        _add(q)
    
    for subq in decomposition.get("subqueries", []):
        if isinstance(subq, dict):
            _add(subq.get("query", ""))
        elif isinstance(subq, str):
            _add(subq)
    
    entity_terms = _extract_entity_terms(decomposition, original_query)
    attribute_terms = _extract_attribute_terms(original_query)
    for entity in entity_terms:
        if attribute_terms:
            for attr in attribute_terms:
                _add(f"{entity} {attr}")
        else:
            _add(f"{entity} specifications")
    
    return queries[:max_queries]


def _extract_entity_terms(decomposition: Dict[str, Any], query: str) -> List[str]:
    terms: List[str] = []
    entities = decomposition.get("entities", []) or []
    for entity in entities:
        name = ""
        if isinstance(entity, dict):
            name = entity.get("name") or ""
        elif isinstance(entity, str):
            name = entity
        name = (name or "").strip()
        if name:
            terms.append(name)
    number_matches = re.findall(r'\b\d+(?:\.\d+)?\s*(?:w|kw)\b', query.lower())
    for match in number_matches:
        terms.append(match.strip())
    return terms


def _extract_attribute_terms(query: str) -> List[str]:
    lowered = query.lower()
    attributes = {
        "dimensions": ["dimensions", "size"],
        "width height": ["width", "height"],
        "max amperage": ["amperage", "max amperage", "current"],
        "operating voltage": ["voltage", "operating voltage"],
    }
    results: List[str] = []
    for label, keywords in attributes.items():
        if any(keyword in lowered for keyword in keywords):
            results.append(label)
    return results
