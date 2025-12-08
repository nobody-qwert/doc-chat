"""
LLM modes for the agentic RAG system.

Implements the three conceptual modes:
- decompose_query: Parse user query into structured plan
- plan_search: Mode 1 - Decide search strategy
- review_evidence: Mode 2 - Decide if evidence is sufficient
- compose_answer: Mode 3 - Generate final answer with citations
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Awaitable, Callable

from .prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_USER_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    COMPOSER_SYSTEM_PROMPT,
    COMPOSER_USER_TEMPLATE,
    COMPOSER_NO_EVIDENCE_SYSTEM_PROMPT,
    COMPOSER_NO_EVIDENCE_USER_TEMPLATE,
    SEMANTIC_REWRITE_SYSTEM_PROMPT,
    SEMANTIC_REWRITE_USER_TEMPLATE,
    format_evidence_for_review,
    format_evidence_for_composer,
    INSPECTOR_SYSTEM_PROMPT,
    INSPECTOR_USER_TEMPLATE,
    format_evidence_for_inspector,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Result from query decomposition."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    prompt: Optional[str] = None
    prompt_messages: Optional[List[Dict[str, str]]] = None


@dataclass
class SubqueryPlan:
    subquery: str
    strategy: str = "hybrid"
    initial_queries: List[str] = field(default_factory=list)


@dataclass
class PlanResult:
    """Result from search planning."""
    success: bool
    subquery_plans: List[SubqueryPlan] = field(default_factory=list)
    max_tool_calls: int = 4
    error: Optional[str] = None
    raw_response: Optional[str] = None
    prompt: Optional[str] = None
    prompt_messages: Optional[List[Dict[str, str]]] = None


@dataclass
class ReviewResult:
    """Result from evidence review."""
    success: bool
    status: str = "more"  # "enough", "more", "clarify"
    reason: str = ""
    next_tool_call: Optional[Dict[str, Any]] = None
    clarification_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    prompt: Optional[str] = None
    prompt_messages: Optional[List[Dict[str, str]]] = None


@dataclass
class ComposeAnswerResult:
    """Non-streaming compose result including the prompt for auditing."""
    answer: str
    prompt_messages: List[Dict[str, str]]
    raw_response: str
    prompt: Optional[str] = None


@dataclass
class ComposeAnswerStreamResult:
    """Streaming compose result that still exposes the prompt."""
    stream: AsyncGenerator[str, None]
    prompt_messages: List[Dict[str, str]]
    prompt: Optional[str] = None


@dataclass
class QueryRewriteResult:
    """Outcome of the semantic query rewrite helper."""
    success: bool
    rewritten_query: str = ""
    error: Optional[str] = None
    raw_response: Optional[str] = None
    prompt: Optional[str] = None
    prompt_messages: Optional[List[Dict[str, str]]] = None




@dataclass
class InspectorResult:
    """Outcome of the evidence inspector."""
    success: bool
    hits: List[Dict[str, Any]] = None
    inspected_items: int = 0
    inspected_docs: List[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    prompt: Optional[str] = None
    prompt_messages: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        if self.hits is None:
            self.hits = []
        if self.inspected_docs is None:
            self.inspected_docs = []


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response text."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to parse the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def _extract_subqueries(decomposition: Optional[Dict[str, Any]], fallback_query: str) -> List[str]:
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
    return [fallback_query.strip()] if fallback_query.strip() else [fallback_query]


def _format_output_preferences(preferences: Optional[Dict[str, Any]]) -> str:
    if not preferences:
        return "No specific output preferences provided."
    try:
        return json.dumps(preferences, indent=2)
    except TypeError:
        return str(preferences)


def _build_subquery_plans(
    plans_data: Optional[List[Any]],
    fallback_subqueries: List[str],
) -> List[SubqueryPlan]:
    plans: List[SubqueryPlan] = []
    if isinstance(plans_data, list):
        for entry in plans_data:
            if not isinstance(entry, dict):
                continue
            subquery = (entry.get("subquery") or "").strip()
            if not subquery:
                continue
            strategy = entry.get("strategy", "hybrid")
            initial_queries = entry.get("initial_queries")
            if not isinstance(initial_queries, list) or not initial_queries:
                initial_queries = [subquery]
            plans.append(SubqueryPlan(
                subquery=subquery,
                strategy=str(strategy or "hybrid").lower(),
                initial_queries=[str(q).strip() for q in initial_queries if str(q).strip()],
            ))
    if not plans:
        plans = _default_subquery_plans(fallback_subqueries)
    return plans


def _default_subquery_plans(subqueries: List[str]) -> List[SubqueryPlan]:
    plans: List[SubqueryPlan] = []
    for subquery in subqueries:
        text = subquery.strip() or "general search"
        plans.append(SubqueryPlan(
            subquery=text,
            strategy="hybrid",
            initial_queries=[text],
        ))
    return plans


def _subplan_to_dict(plan: SubqueryPlan) -> Dict[str, Any]:
    return {
        "subquery": plan.subquery,
        "strategy": plan.strategy,
        "initial_queries": plan.initial_queries,
    }




async def decompose_query(
    query: str,
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> DecompositionResult:
    """
    Decompose a user query into a structured search plan.
    
    This is the first step that parses the natural language query
    into entities, constraints, and subqueries.
    """
    user_prompt = None
    prompt_messages: Optional[List[Dict[str, str]]] = None
    try:
        user_prompt = DECOMPOSER_USER_TEMPLATE.format(query=query)
        prompt_messages = [
            {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await llm_client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=1000,
        )
        
        if not response.choices:
            return DecompositionResult(
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data = _extract_json(raw_response)
        
        if not data:
            return DecompositionResult(
                success=False,
                error="Could not parse JSON from response",
                raw_response=raw_response,
            )
        
        return DecompositionResult(
            success=True,
            data=data,
            raw_response=raw_response,
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )
        
    except Exception as e:
        logger.exception(f"decompose_query failed: {e}")
        return DecompositionResult(
            success=False,
            error=str(e),
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )


async def plan_search(
    query: str,
    decomposition: Dict[str, Any],
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> PlanResult:
    """
    Generate retrieval instructions for each decomposed subquery.
    
    Mode 1: Selects search strategy, initial queries, and filters
    for every subquery produced by the decomposer.
    """
    user_prompt = None
    prompt_messages: Optional[List[Dict[str, str]]] = None
    subqueries = _extract_subqueries(decomposition, query)
    subquery_payload = decomposition.get("subqueries")
    if not isinstance(subquery_payload, list) or not subquery_payload:
        subquery_payload = [{"query": sq} for sq in subqueries]
    try:
        decomp_str = json.dumps(subquery_payload, indent=2)
        user_prompt = PLANNER_USER_TEMPLATE.format(
            query=query,
            decomposition=decomp_str,
        )
        prompt_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await llm_client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=500,
        )
        
        if not response.choices:
            return PlanResult(
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data = _extract_json(raw_response)
        
        if not data:
            # Fall back to default behavior
            return PlanResult(
                success=True,
                subquery_plans=_default_subquery_plans(subqueries),
                max_tool_calls=4,
                raw_response=raw_response,
                prompt=user_prompt,
                prompt_messages=prompt_messages,
            )
        
        plans = _build_subquery_plans(data.get("subquery_plans"), subqueries)
        return PlanResult(
            success=True,
            subquery_plans=plans,
            max_tool_calls=data.get("max_tool_calls", 4),
            raw_response=raw_response,
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )

    except Exception as e:
        logger.exception(f"plan_search failed: {e}")
        return PlanResult(
            success=False,
            error=str(e),
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )


async def rewrite_semantic_query(
    original_query: str,
    user_query: str,
    llm_client: Any,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 200,
) -> QueryRewriteResult:
    """Generate a high-definition semantic search prompt."""
    normalized_original = (original_query or "").strip()
    user_prompt = SEMANTIC_REWRITE_USER_TEMPLATE.format(
        user_query=user_query.strip(),
        subquery=normalized_original or user_query.strip(),
    )
    prompt_messages = [
        {"role": "system", "content": SEMANTIC_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    def _normalize(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r'^```[a-zA-Z0-9_]*\s*', '', stripped)
            stripped = stripped.strip()
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")]
        stripped = stripped.strip('"')
        stripped = re.sub(r'\s+', ' ', stripped)
        return stripped.strip()

    try:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            return QueryRewriteResult(
                success=False,
                rewritten_query=normalized_original,
                error="LLM returned no choices",
                prompt=user_prompt,
                prompt_messages=prompt_messages,
            )
        raw_response = response.choices[0].message.content or ""
        rewritten = _normalize(raw_response) or normalized_original
        success = bool(rewritten.strip())
        return QueryRewriteResult(
            success=success,
            rewritten_query=rewritten,
            raw_response=raw_response,
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )
    except Exception as e:
        logger.exception(f"rewrite_semantic_query failed: {e}")
        return QueryRewriteResult(
            success=False,
            rewritten_query=normalized_original,
            error=str(e),
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )


async def review_evidence(
    query: str,
    plan: PlanResult,
    evidence: List[Dict[str, Any]],
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> ReviewResult:
    """
    Review collected evidence and decide next step.
    
    Mode 2: Decides if we have enough evidence, need more searches,
    or need to ask for clarification.
    """
    user_prompt = None
    prompt_messages: Optional[List[Dict[str, str]]] = None
    try:
        plan_str = json.dumps({
            "subquery_plans": [_subplan_to_dict(p) for p in plan.subquery_plans],
            "max_tool_calls": plan.max_tool_calls,
        }, indent=2)
        
        evidence_summary = format_evidence_for_review(evidence)
        
        user_prompt = REVIEWER_USER_TEMPLATE.format(
            query=query,
            plan=plan_str,
            evidence_count=len(evidence),
            evidence_summary=evidence_summary,
        )
        prompt_messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await llm_client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=500,
        )
        
        if not response.choices:
            return ReviewResult(
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data = _extract_json(raw_response)
        
        if not data:
            # Default to "enough" if we have evidence, otherwise "more"
            return ReviewResult(
                success=True,
                status="enough" if evidence else "more",
                reason="Could not parse reviewer response",
                raw_response=raw_response,
                prompt=user_prompt,
                prompt_messages=prompt_messages,
            )
        
        return ReviewResult(
            success=True,
            status=data.get("status", "enough"),
            reason=data.get("reason", ""),
            next_tool_call=data.get("next_tool_call"),
            clarification_details=data.get("clarification_details"),
            raw_response=raw_response,
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )
        
    except Exception as e:
        logger.exception(f"review_evidence failed: {e}")
        return ReviewResult(
            success=False,
            error=str(e),
            prompt=user_prompt,
            prompt_messages=prompt_messages,
        )


async def compose_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    llm_client: Any,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    stream: bool = False,
    output_preferences: Optional[Dict[str, Any]] = None,
) -> ComposeAnswerResult | ComposeAnswerStreamResult:
    """
    Compose final answer from evidence.
    
    Mode 3: Generates the final answer with proper citations.
    
    Args:
        query: User query
        evidence: List of evidence items
        llm_client: OpenAI-compatible client
        model: Model name
        temperature: LLM temperature
        max_tokens: Max response tokens
        stream: If True, returns async generator for streaming
    
    Returns:
        ComposeAnswerResult when stream=False or ComposeAnswerStreamResult when stream=True
    """
    evidence_str = format_evidence_for_composer(evidence)
    use_no_evidence_prompt = len(evidence) == 0
    
    if use_no_evidence_prompt:
        user_prompt = COMPOSER_NO_EVIDENCE_USER_TEMPLATE.format(
            query=query,
            output_preferences=_format_output_preferences(output_preferences),
        )
        system_prompt = COMPOSER_NO_EVIDENCE_SYSTEM_PROMPT
    else:
        user_prompt = COMPOSER_USER_TEMPLATE.format(
            query=query,
            evidence=evidence_str,
            output_preferences=_format_output_preferences(output_preferences),
        )
        system_prompt = COMPOSER_SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    if stream:
        async def stream_generator():
            try:
                response = await llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.exception(f"compose_answer streaming failed: {e}")
                yield f"\n\n[Error generating answer: {e}]"
        
        return ComposeAnswerStreamResult(stream=stream_generator(), prompt_messages=messages, prompt=user_prompt)
    
    else:
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if not response.choices:
                fallback = "I was unable to generate an answer."
                return ComposeAnswerResult(answer=fallback, prompt_messages=messages, raw_response=fallback, prompt=user_prompt)
            
            answer_text = response.choices[0].message.content or ""
            return ComposeAnswerResult(answer=answer_text, prompt_messages=messages, raw_response=answer_text, prompt=user_prompt)
            
        except Exception as e:
            logger.exception(f"compose_answer failed: {e}")
            fallback = f"Error generating answer: {e}"
            return ComposeAnswerResult(answer=fallback, prompt_messages=messages, raw_response=fallback, prompt=user_prompt)


async def inspect_evidence(
    query: str,
    evidence: List[Dict[str, Any]],
    llm_client: Any,
    model: str,
    max_items: int = 6,
    max_hits: int = 4,
    temperature: float = 0.0,
    progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> InspectorResult:
    """
    Sequentially inspect top evidence items to see if any single snippet fully answers the query.
    
    progress_callback (optional) receives a dict after each LLM inspection with fields:
        - doc_name/doc_hash/index metadata
        - duration_seconds
        - prompt / prompt_messages / raw_response
        - parsed_result (if any) and found flag
        - error (if the inspection failed)
    """
    if not evidence:
        return InspectorResult(success=True, hits=[], inspected_items=0)
    
    items_to_check = evidence[:max_items]
    inspected = 0
    last_response = None
    last_prompt: Optional[str] = None
    last_messages: Optional[List[Dict[str, str]]] = None
    hits: List[Dict[str, Any]] = []
    inspected_docs: List[Dict[str, Any]] = []
    
    for item in items_to_check:
        inspected += 1
        doc_hash = item.get("doc_hash", item.get("doc_id"))
        doc_name = item.get("document_name", item.get("original_name", "Unknown Document"))
        match_type = item.get("match_type")
        inspected_docs.append({
            "doc_hash": doc_hash,
            "doc_name": doc_name,
            "order": inspected,
        })
        evidence_text = format_evidence_for_inspector(item)
        user_prompt = INSPECTOR_USER_TEMPLATE.format(
            query=query,
            doc_name=doc_name,
            doc_hash=doc_hash or "unknown",
            evidence=evidence_text,
        )
        prompt_messages = [
            {"role": "system", "content": INSPECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_prompt = user_prompt
        last_messages = prompt_messages
        
        start_time = time.perf_counter()
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=600,
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            if progress_callback:
                await progress_callback({
                    "doc_name": doc_name,
                    "doc_hash": doc_hash,
                    "index": inspected,
                    "match_type": match_type,
                    "prompt": user_prompt,
                    "prompt_messages": prompt_messages,
                    "duration": duration,
                    "error": str(e),
                })
            logger.exception(f"inspect_evidence failed: {e}")
            return InspectorResult(
                success=False,
                hits=[],
                inspected_items=inspected,
                inspected_docs=inspected_docs,
                error=str(e),
                raw_response=last_response,
                prompt=user_prompt,
                prompt_messages=prompt_messages,
            )
        
        if not response.choices:
            duration = time.perf_counter() - start_time
            if progress_callback:
                await progress_callback({
                    "doc_name": doc_name,
                    "doc_hash": doc_hash,
                    "index": inspected,
                    "match_type": match_type,
                    "prompt": user_prompt,
                    "prompt_messages": prompt_messages,
                    "duration": duration,
                    "raw_response": "",
                    "parsed_result": None,
                    "found": False,
                })
            continue
        
        raw_response = response.choices[0].message.content or ""
        last_response = raw_response
        data = _extract_json(raw_response)
        duration = time.perf_counter() - start_time
        found = bool(data.get("found")) if data else False
        
        if progress_callback:
            await progress_callback({
                "doc_name": doc_name,
                "doc_hash": doc_hash,
                "index": inspected,
                "match_type": match_type,
                "prompt": user_prompt,
                "prompt_messages": prompt_messages,
                "raw_response": raw_response,
                "parsed_result": data,
                "found": found,
                "duration": duration,
            })
        
        if not data or not found:
            continue
        
        hits.append({
            "quote": data.get("quote"),
            "doc_hash": doc_hash,
            "doc_name": doc_name,
        })
        last_response = raw_response
        last_prompt = user_prompt
        last_messages = prompt_messages
        
        if len(hits) >= max_hits:
            break
    
    return InspectorResult(
        success=True,
        hits=hits,
        inspected_items=inspected,
        inspected_docs=inspected_docs,
        raw_response=last_response,
        prompt=last_prompt,
        prompt_messages=last_messages,
    )


def verify_citations(answer: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Verify that all citations in the answer exist in the evidence.
    
    Removes or marks invalid citations to prevent hallucination.
    """
    # Extract all doc_ids from evidence
    valid_ids = set()
    for item in evidence:
        doc_id = item.get("doc_hash", item.get("doc_id"))
        if doc_id:
            valid_ids.add(doc_id)
            # Also add partial matches (first 8 chars)
            valid_ids.add(doc_id[:8])
        citation_id = item.get("citation_id")
        if citation_id:
            valid_ids.add(str(citation_id))
    
    # Find all citations in answer
    citation_pattern = r'\[([a-zA-Z0-9_-]+)\]'
    
    def check_citation(match):
        cited_id = match.group(1)
        # Check if citation is valid
        if cited_id in valid_ids:
            return match.group(0)
        # Check partial match
        for valid_id in valid_ids:
            if valid_id.startswith(cited_id) or cited_id.startswith(valid_id):
                return match.group(0)
        # Invalid citation - mark it
        logger.warning(f"Invalid citation removed: [{cited_id}]")
        return f"[citation needed]"
    
    verified_answer = re.sub(citation_pattern, check_citation, answer)
    return verified_answer


def build_clarification_response(details: Dict[str, Any]) -> str:
    """Build a user-friendly clarification request."""
    clarify_type = details.get("type", "unknown")
    missing_info = details.get("missing_info", "")
    
    if clarify_type == "no_results":
        return (
            f"I couldn't find any documents matching your query. {missing_info}\n\n"
            "Could you please:\n"
            "- Check if the search terms are correct\n"
            "- Try broader search criteria\n"
            "- Specify a different document category"
        )
    
    elif clarify_type == "overload":
        return (
            f"I found too many results to process effectively. {missing_info}\n\n"
            "Could you please narrow your search by:\n"
            "- Adding specific dates or date ranges\n"
            "- Specifying particular companies or entities\n"
            "- Adding amount constraints (e.g., 'over $1000')\n"
            "- Being more specific about what you're looking for"
        )
    
    else:
        return (
            f"I need more information to answer your question. {missing_info}\n\n"
            "Please provide additional details or rephrase your query."
        )
