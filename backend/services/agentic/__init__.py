"""
Agentic RAG System.

This module implements an agentic retrieval-augmented generation pattern
where the LLM actively plans, searches, reviews evidence, and composes answers.

Components:
- tools: Search tools exposed to the LLM agent
- decomposer: Two-phase query decomposition (router + schema planner)
- planner: Mode 1 - Plan search strategy
- reviewer: Mode 2 - Review evidence and decide next steps
- composer: Mode 3 - Compose final answer with citations
- orchestrator: Main agentic loop with context management
- prompts: System prompts for each mode
"""

from .orchestrator import agentic_answer, stream_agentic_answer

__all__ = [
    "agentic_answer",
    "stream_agentic_answer",
]
