"""
Agentic RAG System.

This module implements an agentic retrieval-augmented generation pattern
where the LLM actively plans, searches, inspects evidence, and composes answers.

Components:
- tools: Search tools exposed to the LLM agent
- decomposer: Two-phase query decomposition (router + schema planner)
- plan_builder: Deterministic hybrid search plan generator
- composer: Mode 3 - Compose final answer with citations
- orchestrator: Main agentic loop with context management
- prompts: System prompts for each mode
"""

from .orchestrator import stream_agentic_answer

__all__ = [
    "stream_agentic_answer",
]
