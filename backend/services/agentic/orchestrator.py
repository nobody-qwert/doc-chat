"""
Agentic RAG Orchestrator.

Main entry point for the agentic RAG system. Implements:
- The main agentic_answer() function
- The streaming version stream_agentic_answer()
- Evidence collection loop with tool calls
- Context pruning to manage token budget
- Citation verification
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
    plan_search,
    review_evidence,
    compose_answer,
    inspect_evidence,
    verify_citations,
    build_clarification_response,
    rewrite_semantic_query,
    PlanResult,
    SubqueryPlan,
    InspectorResult,
)
from .tools import execute_tool, ToolResult

logger = logging.getLogger(__name__)

INSPECTOR_MAX_ITEMS = 10


@dataclass
class AgenticStep:
    """Record of a single step in the agentic loop."""
    step_number: int
    name: str
    kind: str  # "decompose", "plan", "review", "tool", "compose"
    duration_seconds: float = 0.0
    state: str = "done"  # "started", "done", "error"
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgenticResult:
    """Result from the agentic RAG pipeline."""
    answer: str
    success: bool
    sources: List[Dict[str, Any]] = field(default_factory=list)
    conversation_id: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    steps: List[AgenticStep] = field(default_factory=list)
    total_tool_calls: int = 0
    evidence_count: int = 0
    finish_reason: Optional[str] = None
    inspector_found: Optional[bool] = None


async def agentic_answer(
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    llm_client: Any,
    settings: Any,
    conversation_id: Optional[str] = None,
    max_tool_calls: int = 5,
) -> AgenticResult:
    """
    Main agentic RAG function.
    
    Orchestrates the full pipeline:
    1. Decompose query
    2. Plan search
    3. Iterative search loop (review → tool call → review ...)
    4. Compose answer
    5. Verify citations
    
    Args:
        query: User question
        document_store: Database access
        embedding_client: For generating embeddings
        embedding_cache: Precomputed embeddings
        llm_client: OpenAI-compatible async client
        settings: Application settings
        conversation_id: Optional conversation ID for context
        max_tool_calls: Maximum tool calls allowed
    
    Returns:
        AgenticResult with answer and metadata
    """
    steps: List[AgenticStep] = []
    evidence: List[Dict[str, Any]] = []
    total_tool_calls = 0
    inspector_found: Optional[bool] = None
    inspector_found_flag: Optional[bool] = None
    
    model = settings.llm_model
    
    async def _run_tool(
        tool_name: str,
        tool_args: Dict[str, Any],
        display_name: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> Optional[ToolResult]:
        """Execute a tool, record the step, and optionally trigger fallback search."""
        nonlocal total_tool_calls, evidence, steps
        if total_tool_calls >= effective_max_calls:
            return None
        args = dict(tool_args)
        if tool_name == "search_semantic" and (args.get("query") or "").strip():
            rewrite_label = "Enhance Semantic Query"
            rewrite_start = time.perf_counter()
            rewrite_result = await rewrite_semantic_query(
                original_query=args.get("query", ""),
                user_query=query,
                llm_client=llm_client,
                model=model,
            )
            rewrite_duration = time.perf_counter() - rewrite_start
            rewritten_query = rewrite_result.rewritten_query or args.get("query", "")
            args["query"] = rewritten_query
            steps.append(AgenticStep(
                step_number=len(steps),
                name=rewrite_label,
                kind="rewrite",
                duration_seconds=rewrite_duration,
                details=_llm_step_details(
                    {
                        "original_query": tool_args.get("query"),
                        "rewritten_query": rewritten_query,
                    },
                    rewrite_result,
                ),
                error=rewrite_result.error if not rewrite_result.success else None,
            ))
        tool_start = time.perf_counter()
        result = await execute_tool(
            tool_name=tool_name,
            args=args,
            document_store=document_store,
            embedding_client=embedding_client,
            embedding_cache=embedding_cache,
            settings=settings,
        )
        total_tool_calls += 1
        tool_duration = time.perf_counter() - tool_start
        
        label = display_name or tool_name
        steps.append(AgenticStep(
            step_number=len(steps),
            name=label,
            kind="tool",
            duration_seconds=tool_duration,
            details=_tool_step_details(tool_name, args, result),
            error=result.error if not result.success else None,
        ))
        
        if result.success:
            evidence.extend(result.results)
        
        if allow_fallback:
            await _run_fallback_if_needed(tool_name, args, label, result)
        return result
    
    async def _run_fallback_if_needed(
        primary_tool: str,
        primary_args: Dict[str, Any],
        primary_label: str,
        primary_result: Optional[ToolResult],
    ) -> None:
        """Automatically invoke the complementary search mode when no hits were found."""
        nonlocal total_tool_calls
        if not _should_trigger_fallback(primary_result):
            return
        fallback_tool = _fallback_tool_name(primary_tool)
        if not fallback_tool or total_tool_calls >= effective_max_calls:
            return
        fallback_args = dict(primary_args)
        _ensure_context_chars(fallback_tool, fallback_args)
        fallback_label = f"{primary_label} -> {fallback_tool}"
        await _run_tool(fallback_tool, fallback_args, display_name=fallback_label, allow_fallback=False)
    
    # Step 0: Decompose query
    decomp_start = time.perf_counter()
    decomp_result = await decompose_query(query, llm_client, model)
    decomp_duration = time.perf_counter() - decomp_start
    
    steps.append(AgenticStep(
        step_number=0,
        name="Decompose Query",
        kind="decompose",
        duration_seconds=decomp_duration,
        details=_llm_step_details(
            {"decomposition": decomp_result.data} if decomp_result.data else None,
            decomp_result,
        ),
        error=decomp_result.error if not decomp_result.success else None,
    ))
    
    if not decomp_result.success:
        decomposition = {"intent": "qa", "subqueries": [{"query": query}]}
    else:
        decomposition = decomp_result.data or {}

    # Step 1: Plan search
    plan_start = time.perf_counter()
    plan_result = await plan_search(query, decomposition, llm_client, model)
    plan_duration = time.perf_counter() - plan_start
    
    steps.append(AgenticStep(
        step_number=len(steps),
        name="Plan Search",
        kind="plan",
        duration_seconds=plan_duration,
        details=_llm_step_details(
            _plan_to_summary(plan_result) if plan_result.success else None,
            plan_result,
        ),
        error=plan_result.error if not plan_result.success else None,
    ))
    
    if not plan_result.success or not plan_result.subquery_plans:
        plan_result = PlanResult(
            success=True,
            subquery_plans=_fallback_subquery_plans(decomposition, query),
            max_tool_calls=max_tool_calls,
            prompt=plan_result.prompt,
            prompt_messages=plan_result.prompt_messages,
            raw_response=plan_result.raw_response,
            error=plan_result.error,
        )
    
    effective_max_calls = min(max_tool_calls, plan_result.max_tool_calls or 5)
    for subplan in plan_result.subquery_plans:
        base_queries = subplan.initial_queries or [subplan.subquery or query]
        subplan.initial_queries = _augmented_initial_queries(
            subplan.subquery or query,
            base_queries,
            decomposition,
            max_queries=6,
        )
    
    # Step 2: Initial searches based on plan
    for subplan in plan_result.subquery_plans[:2]:
        for search_query in subplan.initial_queries[:2]:
            if total_tool_calls >= effective_max_calls:
                break
            if subplan.strategy in ("keyword", "hybrid") and total_tool_calls < effective_max_calls:
                await _run_tool(
                    "search_text",
                    {"query": search_query, "top_k": 5},
                    display_name="search_text",
                )
            if subplan.strategy in ("semantic", "hybrid") and total_tool_calls < effective_max_calls:
                await _run_tool(
                    "search_semantic",
                    {"query": search_query, "top_k": 5},
                    display_name="search_semantic",
                )
    
    # Deduplicate evidence by chunk_id
    evidence = _deduplicate_evidence(evidence)
    
    # Step 3: Review and refine loop
    for iteration in range(effective_max_calls - total_tool_calls):
        review_start = time.perf_counter()
        review_result = await review_evidence(query, plan_result, evidence, llm_client, model)
        review_duration = time.perf_counter() - review_start
        
        steps.append(AgenticStep(
            step_number=len(steps),
            name="Review Evidence",
            kind="review",
            duration_seconds=review_duration,
            details=_llm_step_details(
                {
                    "status": review_result.status,
                    "reason": review_result.reason,
                    "evidence_count": len(evidence),
                    "proposed_tool_call": review_result.next_tool_call,
                    "clarification_details": review_result.clarification_details,
                },
                review_result,
            ),
            error=review_result.error if not review_result.success else None,
        ))
        
        if review_result.status == "enough":
            break
        
        if review_result.status == "clarify":
            # Need user clarification
            clarification_msg = build_clarification_response(
                review_result.clarification_details or {}
            )
            return AgenticResult(
                answer=clarification_msg,
                success=True,
                sources=_build_sources(evidence),
                conversation_id=conversation_id,
                needs_clarification=True,
                clarification_message=clarification_msg,
                steps=[_step_to_dict(s) for s in steps],
                total_tool_calls=total_tool_calls,
                evidence_count=len(evidence),
                finish_reason="clarify",
            )
        
        # status == "more" - execute next tool call
        if review_result.next_tool_call and total_tool_calls < effective_max_calls:
            tool_call = review_result.next_tool_call
            tool_name = tool_call.get("tool", "")
            original_tool_args = tool_call.get("args", {})
            tool_args = dict(original_tool_args)
            
            tool_result = await _run_tool(tool_name, tool_args, display_name=tool_name)
            
            if tool_result:
                evidence = _deduplicate_evidence(evidence)
        else:
            break
    
    # Inspector pass to extract direct answers from top evidence items
    inspector_result: Optional[InspectorResult] = None
    if evidence:
        inspector_input = _prioritize_full_doc_evidence(evidence)
        inspector_events: List[Dict[str, Any]] = []

        async def _capture_inspector_event(event: Dict[str, Any]) -> None:
            inspector_events.append(event)

        inspector_start = time.perf_counter()
        inspector_result = await inspect_evidence(
            query,
            inspector_input,
            llm_client,
            model,
            max_items=min(INSPECTOR_MAX_ITEMS, len(inspector_input)),
            max_hits=INSPECTOR_MAX_ITEMS,
            progress_callback=_capture_inspector_event,
        )
        inspector_duration = time.perf_counter() - inspector_start
        steps.append(AgenticStep(
            step_number=len(steps),
            name="Inspect Evidence",
            kind="inspect",
            duration_seconds=inspector_duration,
            details=_llm_step_details(
                {
                    "inspected_items": inspector_result.inspected_items if inspector_result else 0,
                    "hits_count": len(inspector_result.hits) if inspector_result else 0,
                    "inspected_docs": _summarize_inspected_docs(inspector_result.inspected_docs) if inspector_result else None,
                },
                inspector_result,
            ),
            error=inspector_result.error if inspector_result and not inspector_result.success else None,
        ))
        for event in inspector_events:
            sub_step = _inspector_event_to_step(event, len(steps))
            steps.append(sub_step)

        if inspector_result is not None:
            if inspector_result.success:
                inspector_found_flag = bool(inspector_result.hits)
            else:
                inspector_found_flag = None

        if inspector_result and inspector_result.success and inspector_result.hits:
            inspector_evidence = _inspector_hits_to_evidence(inspector_result.hits)
            _assign_citation_ids(inspector_evidence)
            compose_start = time.perf_counter()
            compose_response = await compose_answer(
                query,
                inspector_evidence,
                llm_client,
                model,
                stream=False,
                output_preferences=decomposition.get("output_preferences"),
            )
            answer = compose_response.answer
            compose_duration = time.perf_counter() - compose_start
            
            steps.append(AgenticStep(
                step_number=len(steps),
                name="Compose Answer (Inspector)",
                kind="compose",
                duration_seconds=compose_duration,
                details=_llm_step_details({}, compose_response),
            ))
            
            verified_answer = verify_citations(answer, inspector_evidence)
            sources = _build_sources(inspector_evidence)
            return AgenticResult(
                answer=verified_answer,
                success=True,
                sources=sources,
                conversation_id=conversation_id,
                needs_clarification=False,
                steps=[_step_to_dict(s) for s in steps],
                total_tool_calls=total_tool_calls,
                evidence_count=len(evidence),
                finish_reason="inspector",
                inspector_found=True,
            )

    # Step 4: Prune evidence to top items
    evidence = _prune_evidence(evidence, max_items=15)
    _assign_citation_ids(evidence)
    composer_evidence = [] if inspector_found_flag is False else evidence
    
    # Step 5: Compose final answer
    compose_start = time.perf_counter()
    compose_response = await compose_answer(
        query,
        composer_evidence,
        llm_client,
        model,
        stream=False,
        output_preferences=decomposition.get("output_preferences"),
    )
    answer = compose_response.answer
    compose_duration = time.perf_counter() - compose_start
    
    steps.append(AgenticStep(
        step_number=len(steps),
        name="Compose Answer",
        kind="compose",
        duration_seconds=compose_duration,
        details=_llm_step_details({}, compose_response),
    ))
    
    # Step 6: Verify citations
    verified_answer = verify_citations(answer, composer_evidence)
    
    return AgenticResult(
        answer=verified_answer,
        success=True,
        sources=_build_sources(composer_evidence),
        conversation_id=conversation_id,
        needs_clarification=False,
        steps=[_step_to_dict(s) for s in steps],
        total_tool_calls=total_tool_calls,
        evidence_count=len(composer_evidence),
        finish_reason="complete",
        inspector_found=inspector_found_flag,
    )


async def stream_agentic_answer(
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    llm_client: Any,
    settings: Any,
    conversation_id: Optional[str] = None,
    max_tool_calls: int = 5,
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
    
    model = settings.llm_model
    
    def _emit_step(step: AgenticStep, state: str = "done") -> str:
        step_dict = _step_to_dict(step)
        step_dict["state"] = state
        return json.dumps({"type": "step", "step": step_dict}) + "\n"
    
    async def _stream_fallback_if_needed(
        primary_tool: str,
        primary_args: Dict[str, Any],
        primary_label: str,
        primary_result: Optional[ToolResult],
        step_num_ref: List[int],
    ):
        """Async generator that yields fallback tool events when needed."""
        nonlocal total_tool_calls, evidence
        if not _should_trigger_fallback(primary_result):
            return
        fallback_tool = _fallback_tool_name(primary_tool)
        if not fallback_tool or total_tool_calls >= effective_max_calls:
            return
        args = dict(primary_args)
        _ensure_context_chars(fallback_tool, args)
        label = f"{primary_label} -> {fallback_tool}"
        if fallback_tool == "search_semantic" and (args.get("query") or "").strip():
            rewrite_label = "Enhance Semantic Query"
            yield _emit_step(AgenticStep(step_num_ref[0], rewrite_label, "rewrite"), "started")
            rewrite_start = time.perf_counter()
            rewrite_result = await rewrite_semantic_query(
                original_query=args.get("query", ""),
                user_query=query,
                llm_client=llm_client,
                model=model,
            )
            rewrite_duration = time.perf_counter() - rewrite_start
            rewritten_query = rewrite_result.rewritten_query or args.get("query", "")
            args["query"] = rewritten_query
            rewrite_step = AgenticStep(
                step_number=step_num_ref[0],
                name=rewrite_label,
                kind="rewrite",
                duration_seconds=rewrite_duration,
                details=_llm_step_details(
                    {
                        "original_query": primary_args.get("query"),
                        "rewritten_query": rewritten_query,
                    },
                    rewrite_result,
                ),
                error=rewrite_result.error if not rewrite_result.success else None,
            )
            steps.append(rewrite_step)
            yield _emit_step(rewrite_step)
            step_num_ref[0] += 1
        
        yield _emit_step(AgenticStep(step_num_ref[0], label, "tool"), "started")
        tool_start = time.perf_counter()
        result = await execute_tool(
            tool_name=fallback_tool,
            args=args,
            document_store=document_store,
            embedding_client=embedding_client,
            embedding_cache=embedding_cache,
            settings=settings,
        )
        total_tool_calls += 1
        tool_duration = time.perf_counter() - tool_start
        
        step = AgenticStep(
            step_number=step_num_ref[0],
            name=label,
            kind="tool",
            duration_seconds=tool_duration,
            details=_tool_step_details(fallback_tool, args, result),
        )
        steps.append(step)
        yield _emit_step(step)
        step_num_ref[0] += 1
        
        if result.success:
            evidence.extend(result.results)
    
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
        decomposition = {"intent": "qa", "subqueries": [{"query": query}]}
    else:
        decomposition = decomp_result.data or {}

    # Step 1: Plan search
    plan_step_num = len(steps)
    plan_start = time.perf_counter()
    yield _emit_step(AgenticStep(plan_step_num, "Plan Search", "plan"), "started")
    
    plan_result = await plan_search(query, decomposition, llm_client, model)
    plan_duration = time.perf_counter() - plan_start
    
    step = AgenticStep(
        step_number=plan_step_num,
        name="Plan Search",
        kind="plan",
        duration_seconds=plan_duration,
        details=_llm_step_details(
            _plan_to_summary(plan_result) if plan_result.success else None,
            plan_result,
        ),
    )
    steps.append(step)
    yield _emit_step(step)
    
    if not plan_result.success or not plan_result.subquery_plans:
        plan_result = PlanResult(
            success=True,
            subquery_plans=_fallback_subquery_plans(decomposition, query),
            max_tool_calls=max_tool_calls,
            prompt=plan_result.prompt,
            prompt_messages=plan_result.prompt_messages,
            raw_response=plan_result.raw_response,
            error=plan_result.error,
        )
    
    effective_max_calls = min(max_tool_calls, plan_result.max_tool_calls or 5)
    for subplan in plan_result.subquery_plans:
        base_queries = subplan.initial_queries or [subplan.subquery or query]
        subplan.initial_queries = _augmented_initial_queries(
            subplan.subquery or query,
            base_queries,
            decomposition,
            max_queries=6,
        )
    
    # Step 2: Initial searches
    step_num = len(steps)
    for subplan in plan_result.subquery_plans[:2]:
        for search_query in subplan.initial_queries[:2]:
            if total_tool_calls >= effective_max_calls:
                break
            
            if subplan.strategy in ("keyword", "hybrid") and total_tool_calls < effective_max_calls:
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
                
                counter = [step_num]
                async for event in _stream_fallback_if_needed("search_text", args, label, text_result, counter):
                    yield event
                step_num = counter[0]
            
            if subplan.strategy in ("semantic", "hybrid") and total_tool_calls < effective_max_calls:
                args = {"query": search_query, "top_k": 5}
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
                
                counter = [step_num]
                async for event in _stream_fallback_if_needed("search_semantic", args, label, semantic_result, counter):
                    yield event
                step_num = counter[0]
    
    evidence = _deduplicate_evidence(evidence)
    
    # Step 3: Review loop (simplified for streaming)
    for iteration in range(min(2, effective_max_calls - total_tool_calls)):
        yield _emit_step(AgenticStep(step_num, "Review Evidence", "review"), "started")
        review_start = time.perf_counter()
        
        review_result = await review_evidence(query, plan_result, evidence, llm_client, model)
        review_duration = time.perf_counter() - review_start
        
        step = AgenticStep(
            step_number=step_num,
            name="Review Evidence",
            kind="review",
            duration_seconds=review_duration,
            details=_llm_step_details(
                {
                    "status": review_result.status,
                    "reason": review_result.reason,
                    "evidence_count": len(evidence),
                    "proposed_tool_call": review_result.next_tool_call,
                    "clarification_details": review_result.clarification_details,
                },
                review_result,
            ),
        )
        steps.append(step)
        yield _emit_step(step)
        step_num += 1
        
        if review_result.status == "enough":
            break
        
        if review_result.status == "clarify":
            clarification_msg = build_clarification_response(review_result.clarification_details or {})
            yield json.dumps({"type": "token", "content": clarification_msg}) + "\n"
            yield json.dumps({
                "type": "final",
                "answer": clarification_msg,
                "needs_clarification": True,
                "sources": _build_sources(evidence),
                "steps": [_step_to_dict(s) for s in steps],
                "total_tool_calls": total_tool_calls,
                "evidence_count": len(evidence),
                "finish_reason": "clarify",
            }) + "\n"
            return
        
        if review_result.next_tool_call and total_tool_calls < effective_max_calls:
            tool_call = review_result.next_tool_call
            tool_name = tool_call.get("tool", "")
            original_tool_args = tool_call.get("args", {})
            tool_args = dict(original_tool_args)
            if tool_name == "search_semantic" and (tool_args.get("query") or "").strip():
                rewrite_label = "Enhance Semantic Query"
                yield _emit_step(AgenticStep(step_num, rewrite_label, "rewrite"), "started")
                rewrite_start = time.perf_counter()
                rewrite_result = await rewrite_semantic_query(
                    original_query=tool_args.get("query", ""),
                    user_query=query,
                    llm_client=llm_client,
                    model=model,
                )
                rewrite_duration = time.perf_counter() - rewrite_start
                rewritten_query = rewrite_result.rewritten_query or tool_args.get("query", "")
                tool_args["query"] = rewritten_query
                rewrite_step = AgenticStep(
                    step_number=step_num,
                    name=rewrite_label,
                    kind="rewrite",
                    duration_seconds=rewrite_duration,
                    details=_llm_step_details(
                        {
                            "original_query": original_tool_args.get("query"),
                            "rewritten_query": rewritten_query,
                        },
                        rewrite_result,
                    ),
                    error=rewrite_result.error if not rewrite_result.success else None,
                )
                steps.append(rewrite_step)
                yield _emit_step(rewrite_step)
                step_num += 1
            
            yield _emit_step(AgenticStep(step_num, tool_name, "tool"), "started")
            tool_start = time.perf_counter()
            
            tool_result = await execute_tool(
                tool_name=tool_name,
                args=tool_args,
                document_store=document_store,
                embedding_client=embedding_client,
                embedding_cache=embedding_cache,
                settings=settings,
            )
            total_tool_calls += 1
            tool_duration = time.perf_counter() - tool_start
            
            step = AgenticStep(
                step_number=step_num,
                name=tool_name,
                kind="tool",
                duration_seconds=tool_duration,
                details=_tool_step_details(tool_name, tool_args, tool_result),
            )
            steps.append(step)
            yield _emit_step(step)
            step_num += 1
            
            if tool_result.success:
                evidence.extend(tool_result.results)
            
            counter = [step_num]
            async for event in _stream_fallback_if_needed(tool_name, tool_args, tool_name, tool_result, counter):
                yield event
            step_num = counter[0]
            
            evidence = _deduplicate_evidence(evidence)
    
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


def _fallback_tool_name(primary_tool: str) -> Optional[str]:
    """Return the complementary search tool to try when no results were found."""
    mapping = {
        "search_text": "search_semantic",
        "search_semantic": "search_text",
    }
    return mapping.get(primary_tool)


def _should_trigger_fallback(result: Optional[ToolResult]) -> bool:
    """Determine if a fallback search should run."""
    if result is None:
        return False
    if not result.success:
        return False
    if result.total_found:
        return False
    return not result.results


def _ensure_context_chars(tool_name: str, args: Dict[str, Any]) -> None:
    """Set default context window sizes for fallback tool calls."""
    if "context_chars" in args:
        return
    if tool_name == "search_semantic":
        args["context_chars"] = 500
    elif tool_name == "search_text":
        args["context_chars"] = 400


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
    for idx, hit in enumerate(hits, 1):
        citations = hit.get("citations") or []
        doc_hash = hit.get("doc_hash") or (citations[0] if citations else None)
        doc_name = hit.get("doc_name") or "Unknown Document"
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


def _plan_to_summary(plan_result: Optional[PlanResult]) -> Optional[Dict[str, Any]]:
    if not plan_result or not plan_result.subquery_plans:
        return None
    return {
        "subquery_plans": [
            {
                "subquery": sp.subquery,
                "strategy": sp.strategy,
                "initial_queries": sp.initial_queries,
            }
            for sp in plan_result.subquery_plans
        ],
        "max_tool_calls": plan_result.max_tool_calls,
    }


def _decomposition_subqueries(
    decomposition: Optional[Dict[str, Any]],
    fallback_query: str,
) -> List[str]:
    if isinstance(decomposition, dict):
        items = decomposition.get("subqueries")
        if isinstance(items, list) and items:
            results: List[str] = []
            for entry in items:
                if isinstance(entry, dict):
                    text = entry.get("query")
                else:
                    text = entry
                text = (text or "").strip()
                if text:
                    results.append(text)
            if results:
                return results
    fallback = fallback_query.strip()
    return [fallback] if fallback else [fallback_query]


def _fallback_subquery_plans(
    decomposition: Optional[Dict[str, Any]],
    query: str,
) -> List[SubqueryPlan]:
    subqueries = _decomposition_subqueries(decomposition, query)
    plans: List[SubqueryPlan] = []
    for subquery in subqueries:
        text = subquery.strip() or query
        plans.append(SubqueryPlan(
            subquery=text,
            strategy="hybrid",
            initial_queries=[text],
        ))
    return plans


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
