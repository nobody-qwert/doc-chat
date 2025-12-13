"""
Search tools for the agentic RAG system.

Implements:
- search_text: Keyword-based search with LIKE queries
- search_semantic: Vector similarity search
- get_document_metadata: Retrieve document metadata

These tools are called by the agent during the evidence collection loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    total_found: int = 0


async def search_text(
    query: str,
    document_store: Any,
    settings: Any,
    embedding_client: Any,
    embedding_cache: Any,
    top_k: int = 10,
    context_chars: int = 400,
    doc_id: Optional[str] = None,
) -> ToolResult:
    """
    Keyword/phrase search across all processed documents using LIKE queries.
    
    Args:
        query: Keyword search string
        document_store: Database access object
        top_k: Number of results to return
        context_chars: Characters to include in snippet
        doc_id: Optional - scope search to specific document
    
    Returns:
        ToolResult with matching chunks
    """
    try:
        matching_doc_hashes, doc_info_map = await _collect_documents(
            document_store=document_store,
            doc_id=doc_id,
        )
        
        doc_token_map = _build_doc_token_map(doc_info_map)
        doc_chunks_cache: Dict[str, List[Dict[str, Any]]] = {}
        doc_text_cache: Dict[str, str] = {}
        search_terms = _prepare_search_terms(query)
        short_doc_chunk_limit = getattr(settings, "agentic_short_doc_chunk_limit")
        short_doc_token_limit = getattr(settings, "agentic_short_doc_token_limit")
        expansion_before = getattr(settings, "agentic_expansion_chunks_before")
        expansion_after = getattr(settings, "agentic_expansion_chunks_after")
        max_window_chunks = getattr(settings, "agentic_max_expanded_chunks")
        expanded_char_limit = _expansion_char_limit(context_chars, settings)
        combined_results: List[Dict[str, Any]] = []
        if matching_doc_hashes:
            keyword_threshold = getattr(settings, "min_keyword_score", 0.0)

            short_doc_results: List[Dict[str, Any]] = []
            long_doc_hits: List[Dict[str, Any]] = []
            
            for doc_hash in matching_doc_hashes:
                chunks = await document_store.fetch_chunks(doc_hash=doc_hash)
                doc_chunks_cache[doc_hash] = chunks
                doc_info = doc_info_map.get(doc_hash) or await document_store.get_document(doc_hash)
                doc_total_tokens = doc_token_map.get(doc_hash)
                if doc_total_tokens is None:
                    raise ValueError(f"Document {doc_hash} is missing token_count metadata")
                
                if doc_total_tokens <= short_doc_token_limit:
                    doc_text = await _load_full_document_text(
                        doc_hash,
                        document_store=document_store,
                        settings=settings,
                        doc_text_cache=doc_text_cache,
                        fallback_chunks=chunks,
                    )
                    doc_score = _keyword_score_text(doc_text.lower(), search_terms)
                    if doc_score >= keyword_threshold:
                        short_doc_results.append({
                            "doc_hash": doc_hash,
                            "chunk_id": f"{doc_hash}::full_doc",
                            "order_index": 0,
                            "text": doc_text,
                            "document_name": doc_info.get("original_name", "Unknown") if doc_info else "Unknown",
                            "score": doc_score,
                            "match_type": "keyword_full_doc",
                            "expanded_context": {
                                "type": "full_document",
                                "chunk_count": len(chunks),
                                "token_count": doc_total_tokens,
                            },
                        })
                    continue
                
                hits = _score_chunks_for_doc(
                    chunks=chunks,
                    search_terms=search_terms,
                    doc_hash=doc_hash,
                    doc_info=doc_info,
                    context_chars=context_chars,
                    min_score=keyword_threshold,
                )
                if hits:
                    long_doc_hits.extend(hits)
            
            long_doc_hits.sort(key=lambda x: x["score"], reverse=True)
            long_doc_hits = long_doc_hits[:max(top_k, 50)]
            
            expanded_long_hits = await _expand_evidence_chunks(
                long_doc_hits,
                document_store=document_store,
                doc_chunks_cache=doc_chunks_cache,
                settings=settings,
                doc_text_cache=doc_text_cache,
                doc_token_map=doc_token_map,
                short_doc_chunk_limit=short_doc_chunk_limit,
                short_doc_token_limit=short_doc_token_limit,
                expansion_before=expansion_before,
                expansion_after=expansion_after,
                max_window_chunks=max_window_chunks,
                max_chars=expanded_char_limit,
            )
            
            combined_results = short_doc_results + expanded_long_hits
            combined_results = [
                item for item in combined_results
                if float(item.get("score", 0.0)) >= keyword_threshold
            ]
            combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            combined_results = combined_results[:top_k]
        return ToolResult(
            tool_name="search_text",
            success=True,
            results=combined_results,
            total_found=len(combined_results),
        )
        
    except Exception as e:
        logger.exception(f"search_text failed: {e}")
        return ToolResult(
            tool_name="search_text",
            success=False,
            error=str(e),
        )


async def search_semantic(
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    settings: Any,
    top_k: int = 10,
    context_chars: int = 500,
    doc_id: Optional[str] = None,
) -> ToolResult:
    """
    Semantic/vector search across all processed documents.
    
    Args:
        query: Natural language query for embedding
        document_store: Database access object
        embedding_client: Client for generating embeddings
        embedding_cache: Cache with precomputed embeddings
        top_k: Number of results to return
        context_chars: Characters to include in snippet
        doc_id: Optional - scope search to specific document
    
    Returns:
        ToolResult with semantically similar chunks
    """
    try:
        # Embed the query
        query_vectors = await embedding_client.embed_batch([query])
        if not query_vectors or query_vectors[0] is None:
            return ToolResult(
                tool_name="search_semantic",
                success=False,
                error="Failed to embed query",
            )
        
        query_vec = np.asarray(query_vectors[0], dtype=np.float32)
        
        matching_doc_hashes, doc_info_map = await _collect_documents(
            document_store=document_store,
            doc_id=doc_id,
        )
        matching_doc_hashes_set: Set[str] = set(matching_doc_hashes)
        doc_token_map = _build_doc_token_map(doc_info_map)
        doc_chunks_cache: Dict[str, List[Dict[str, Any]]] = {}
        doc_text_cache: Dict[str, str] = {}
        short_doc_chunk_limit = getattr(settings, "agentic_short_doc_chunk_limit")
        short_doc_token_limit = getattr(settings, "agentic_short_doc_token_limit")
        expansion_before = getattr(settings, "agentic_expansion_chunks_before")
        expansion_after = getattr(settings, "agentic_expansion_chunks_after")
        max_window_chunks = getattr(settings, "agentic_max_expanded_chunks")
        expanded_char_limit = _expansion_char_limit(context_chars, settings)

        if not matching_doc_hashes_set:
            return ToolResult(
                tool_name="search_semantic",
                success=True,
                results=[],
                total_found=0,
            )
        
        # Get embedding snapshot
        snapshot = embedding_cache.snapshot()
        if snapshot.total == 0:
            return ToolResult(
                tool_name="search_semantic",
                success=True,
                results=[],
                total_found=0,
            )
        
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        # Compute similarities
        scores = snapshot.matrix @ query_vec

        requested_top_k = int(top_k or 1)
        max_hits = max(1, min(requested_top_k, 10))
        min_similarity = float(getattr(settings, "min_semantic_similarity", 0.0))
        
        # Get top candidates
        candidate_count = min(snapshot.total, max_hits * 5)  # Fetch extra for filtering
        if candidate_count > 0:
            top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = []
        
        # Build results
        results = []
        for idx in top_indices:
            chunk_id = snapshot.chunk_ids[idx]
            score = float(scores[idx])
            if score < min_similarity:
                break
            
            # Get chunk info
            chunks = await document_store.fetch_chunks_by_ids([chunk_id])
            if not chunks:
                continue
            
            chunk = chunks[0]
            doc_hash = chunk.get("doc_hash")
            
            # Scope to processed documents
            if doc_hash not in matching_doc_hashes_set:
                continue
            
            doc_info = doc_info_map.get(doc_hash) or await document_store.get_document(doc_hash)
            
            results.append({
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
                "order_index": chunk.get("order_index", 0),
                "text": chunk.get("text", "")[:context_chars],
                "document_name": doc_info.get("original_name", "Unknown") if doc_info else "Unknown",
                "score": score,
                "match_type": "semantic",
            })
            
            if len(results) >= max_hits:
                break
        
        results = await _expand_evidence_chunks(
            results,
            document_store=document_store,
            doc_chunks_cache=doc_chunks_cache,
            settings=settings,
            doc_text_cache=doc_text_cache,
            doc_token_map=doc_token_map,
            short_doc_chunk_limit=short_doc_chunk_limit,
            short_doc_token_limit=short_doc_token_limit,
            expansion_before=expansion_before,
            expansion_after=expansion_after,
            max_window_chunks=max_window_chunks,
            max_chars=expanded_char_limit,
        )
        return ToolResult(
            tool_name="search_semantic",
            success=True,
            results=results,
            total_found=len(results),
        )
        
    except Exception as e:
        logger.exception(f"search_semantic failed: {e}")
        return ToolResult(
            tool_name="search_semantic",
            success=False,
            error=str(e),
        )


async def get_document_metadata(
    doc_id: str,
    document_store: Any,
    settings: Any,
) -> ToolResult:
    """
    Get full metadata for a document.
    
    Args:
        doc_id: Document hash
        document_store: Database access object
        settings: Application settings
    
    Returns:
        ToolResult with document metadata
    """
    try:
        # Get document info
        doc = await document_store.get_document(doc_id)
        if not doc:
            return ToolResult(
                tool_name="get_document_metadata",
                success=False,
                error=f"Document not found: {doc_id}",
            )
        
        result = {
            "doc_hash": doc_id,
            "original_name": doc.get("original_name"),
            "status": doc.get("status"),
            "size": doc.get("size"),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "last_ingested_at": doc.get("last_ingested_at"),
        }
        
        return ToolResult(
            tool_name="get_document_metadata",
            success=True,
            results=[result],
            total_found=1,
        )
        
    except Exception as e:
        logger.exception(f"get_document_metadata failed: {e}")
        return ToolResult(
            tool_name="get_document_metadata",
            success=False,
            error=str(e),
        )


async def _load_full_document_text(
    doc_hash: str,
    *,
    document_store: Any,
    settings: Any,
    doc_text_cache: Dict[str, str],
    fallback_chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if doc_hash in doc_text_cache:
        return doc_text_cache[doc_hash]
    parser_key = getattr(settings, "ocr_parser_key", None)
    text: Optional[str] = None
    if parser_key:
        extraction = await document_store.get_extraction(doc_hash, parser_key)
        if extraction:
            text = extraction.get("text")
    if text is None and fallback_chunks is not None:
        max_chars = getattr(settings, "agentic_max_expanded_chars")
        text = _join_chunk_texts(fallback_chunks, max_chars * 2)
    if text is None:
        text = ""
    doc_text_cache[doc_hash] = text
    return text


def _document_token_count(doc_info: Optional[Dict[str, Any]], doc_hash: str) -> int:
    if not doc_info:
        raise ValueError(f"Missing document metadata for {doc_hash}")
    token_count = doc_info.get("token_count")
    if token_count is None:
        raise ValueError(
            f"Document {doc_hash} is missing token_count metadata. Reingest the document to compute it."
        )
    try:
        return int(token_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Document {doc_hash} has invalid token_count '{token_count}'") from exc


def _build_doc_token_map(doc_info_map: Dict[str, Any]) -> Dict[str, int]:
    token_map: Dict[str, int] = {}
    for doc_hash, info in doc_info_map.items():
        if not doc_hash:
            continue
        token_map[doc_hash] = _document_token_count(info, doc_hash)
    return token_map


async def _collect_documents(
    document_store: Any,
    doc_id: Optional[str] = None,
) -> tuple[List[str], Dict[str, Any]]:
    """Return doc hashes and metadata for processed documents."""
    docs = await document_store.list_documents()
    matches: List[str] = []
    doc_info_map: Dict[str, Any] = {}
    
    for doc in docs:
        if doc.get("status") != "processed":
            continue
        if doc_id and doc["doc_hash"] != doc_id:
            continue
        matches.append(doc["doc_hash"])
        doc_info_map[doc["doc_hash"]] = doc
    
    return matches, doc_info_map


async def _expand_evidence_chunks(
    results: List[Dict[str, Any]],
    *,
    document_store: Any,
    doc_chunks_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    settings: Any,
    doc_text_cache: Optional[Dict[str, str]] = None,
    doc_token_map: Optional[Dict[str, int]] = None,
    short_doc_chunk_limit: Optional[int] = None,
    short_doc_token_limit: Optional[int] = None,
    expansion_before: Optional[int] = None,
    expansion_after: Optional[int] = None,
    max_window_chunks: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Expand evidence snippets by including full documents or neighboring chunks."""
    if not results:
        return results

    doc_chunks_cache = doc_chunks_cache or {}
    doc_text_cache = doc_text_cache or {}
    doc_token_map = doc_token_map or {}
    doc_token_cache: Dict[str, int] = {}
    chunk_index_cache: Dict[str, Dict[str, int]] = {}
    expanded: List[Dict[str, Any]] = []
    seen_full_docs: Set[str] = set()

    short_doc_chunk_limit = short_doc_chunk_limit if short_doc_chunk_limit is not None else getattr(settings, "agentic_short_doc_chunk_limit")
    short_doc_token_limit = short_doc_token_limit if short_doc_token_limit is not None else getattr(settings, "agentic_short_doc_token_limit")
    expansion_before = expansion_before if expansion_before is not None else getattr(settings, "agentic_expansion_chunks_before")
    expansion_after = expansion_after if expansion_after is not None else getattr(settings, "agentic_expansion_chunks_after")
    max_window_chunks = max_window_chunks if max_window_chunks is not None else getattr(settings, "agentic_max_expanded_chunks")
    max_chars = max_chars if max_chars is not None else getattr(settings, "agentic_max_expanded_chars")

    for item in results:
        doc_hash = item.get("doc_hash")
        if not doc_hash:
            expanded.append(item)
            continue

        chunks = doc_chunks_cache.get(doc_hash)
        if chunks is None:
            chunks = await document_store.fetch_chunks(doc_hash=doc_hash)
            doc_chunks_cache[doc_hash] = chunks

        if not chunks:
            expanded.append(item)
            continue

        chunk_count = len(chunks)
        total_tokens = doc_token_cache.get(doc_hash)
        if total_tokens is None:
            if doc_hash not in doc_token_map:
                raise ValueError(f"Document {doc_hash} is missing token_count metadata")
            total_tokens = int(doc_token_map[doc_hash])
            doc_token_cache[doc_hash] = total_tokens

        is_short_doc = (
            (chunk_count <= short_doc_chunk_limit)
            or (total_tokens and total_tokens <= short_doc_token_limit)
        )

        if is_short_doc:
            if doc_hash in seen_full_docs:
                continue
            full_text = await _load_full_document_text(
                doc_hash,
                document_store=document_store,
                settings=settings,
                doc_text_cache=doc_text_cache,
                fallback_chunks=chunks,
            )
            new_item = dict(item)
            new_item["chunk_id"] = f"{doc_hash}::full_doc"
            new_item["order_index"] = 0
            new_item["text"] = full_text
            base_match = item.get("match_type") or "chunk"
            new_item["match_type"] = f"{base_match}_full_doc"
            new_item["expanded_context"] = {
                "type": "full_document",
                "chunk_count": chunk_count,
                "token_count": total_tokens,
            }
            expanded.append(new_item)
            seen_full_docs.add(doc_hash)
            continue

        chunk_index_map = chunk_index_cache.get(doc_hash)
        if chunk_index_map is None:
            chunk_index_map = {
                chunk.get("chunk_id"): idx
                for idx, chunk in enumerate(chunks)
                if chunk.get("chunk_id")
            }
            chunk_index_cache[doc_hash] = chunk_index_map

        focus_idx = chunk_index_map.get(item.get("chunk_id"))
        if focus_idx is None:
            focus_idx = min(max(0, item.get("order_index", 0)), chunk_count - 1)

        start_idx, end_idx = _select_chunk_window(
            chunk_count,
            focus_idx,
            expansion_before=expansion_before,
            expansion_after=expansion_after,
            max_window=max_window_chunks,
        )
        window_chunks = chunks[start_idx:end_idx]
        expanded_text = _join_chunk_texts(window_chunks, max_chars)

        new_item = dict(item)
        new_item["chunk_id"] = f"{item.get('chunk_id')}::window"
        new_item["order_index"] = window_chunks[0].get("order_index", item.get("order_index", 0)) if window_chunks else item.get("order_index", 0)
        new_item["text"] = expanded_text
        new_item["match_type"] = f"{(item.get('match_type') or 'chunk')}_expanded"
        new_item["expanded_context"] = {
            "type": "chunk_window",
            "start_index": start_idx,
            "end_index": end_idx,
            "chunk_count": len(window_chunks),
            "focus_chunk_id": item.get("chunk_id"),
        }
        expanded.append(new_item)

    return expanded


def _select_chunk_window(
    chunk_count: int,
    focus_idx: int,
    *,
    expansion_before: int,
    expansion_after: int,
    max_window: int,
) -> Tuple[int, int]:
    """Return slice indexes [start, end) for the expansion window."""
    start = max(0, focus_idx - expansion_before)
    end = min(chunk_count, focus_idx + expansion_after + 1)

    while (end - start) > max_window:
        # Trim the longer side while keeping focus in view
        left_span = focus_idx - start
        right_span = end - focus_idx - 1
        if right_span > left_span:
            end -= 1
        else:
            start += 1
    return start, end


def _prepare_search_terms(query: str) -> List[str]:
    terms = [term.strip() for term in (query or "").lower().split()]
    terms = [term for term in terms if term and len(term) > 2]
    if not terms:
        terms = [term for term in (query or "").lower().split() if term]
    return terms


def _keyword_score_text(text_lower: str, search_terms: List[str]) -> float:
    if not search_terms or not text_lower:
        return 0.0
    matches = sum(1 for term in search_terms if term in text_lower)
    return matches / len(search_terms) if search_terms else 0.0


def _score_chunks_for_doc(
    chunks: List[Dict[str, Any]],
    search_terms: List[str],
    doc_hash: str,
    doc_info: Optional[Dict[str, Any]],
    context_chars: int,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    if not search_terms:
        return []
    scores: List[Dict[str, Any]] = []
    doc_name = doc_info.get("original_name", "Unknown") if doc_info else "Unknown"
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        score = _keyword_score_text(text.lower(), search_terms)
        if score < min_score:
            continue
        scores.append({
            "doc_hash": doc_hash,
            "chunk_id": chunk.get("chunk_id"),
            "order_index": chunk.get("order_index", 0),
            "text": text[:context_chars],
            "document_name": doc_name,
            "score": score,
            "match_type": "keyword",
        })
    return scores


def _join_chunk_texts(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    """Concatenate chunk texts up to max_chars while trimming overlaps."""
    combined: str = ""
    for chunk in chunks:
        addition = chunk.get("text") or ""
        if not addition:
            continue
        combined = _append_with_overlap(combined, addition)
        if len(combined) >= max_chars:
            combined = combined[:max_chars]
            break
    return combined


def _append_with_overlap(existing: str, addition: str, max_overlap: int = 2000) -> str:
    if not existing:
        return addition
    if not addition:
        return existing
    overlap = _detect_overlap(existing, addition, max_overlap)
    if overlap > 0:
        return existing + addition[overlap:]
    separator = ""
    if not existing.endswith(("\n", " ")) and not addition.startswith(("\n", " ")):
        separator = "\n\n"
    return existing + separator + addition


def _detect_overlap(existing: str, addition: str, max_overlap: int) -> int:
    max_len = min(len(existing), len(addition), max_overlap)
    for size in range(max_len, 0, -1):
        if existing[-size:] == addition[:size]:
            return size
    return 0


def _expansion_char_limit(base_chars: int, settings: Any) -> int:
    """Compute the char cap for expanded context."""
    multiplier = getattr(settings, "agentic_expanded_char_multiplier")
    min_chars = getattr(settings, "agentic_min_expanded_chars")
    max_chars = getattr(settings, "agentic_max_expanded_chars")
    scaled = max(base_chars * multiplier, min_chars)
    return min(scaled, max_chars)


async def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    settings: Any,
) -> ToolResult:
    """
    Execute a tool by name with given arguments.
    
    This is the main dispatcher called by the orchestrator.
    """
    if tool_name == "search_text":
        return await search_text(
            query=args.get("query", ""),
            document_store=document_store,
            settings=settings,
            embedding_client=embedding_client,
            embedding_cache=embedding_cache,
            top_k=args.get("top_k", 10),
            context_chars=args.get("context_chars", 400),
            doc_id=args.get("doc_id"),
        )
    
    elif tool_name == "search_semantic":
        return await search_semantic(
            query=args.get("query", ""),
            document_store=document_store,
            embedding_client=embedding_client,
            embedding_cache=embedding_cache,
            settings=settings,
            top_k=args.get("top_k", 10),
            context_chars=args.get("context_chars", 500),
            doc_id=args.get("doc_id"),
        )
    
    elif tool_name == "get_document_metadata":
        return await get_document_metadata(
            doc_id=args.get("doc_id", ""),
            document_store=document_store,
            settings=settings,
        )
    
    else:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )
