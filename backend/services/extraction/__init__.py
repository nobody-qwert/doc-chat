"""
Schema-driven metadata extraction system.

This module provides:
- ExtractionSchema registry for defining what to extract from document categories
- Generic CSV-based extraction engine using LLM
- Dynamic SQL table creation for metadata storage
- Batch extraction workflow
"""

from .schemas import ExtractionSchema, get_schema, list_schemas, EXTRACTION_SCHEMAS
from .engine import extract_metadata_batch

__all__ = [
    "ExtractionSchema",
    "get_schema",
    "list_schemas",
    "EXTRACTION_SCHEMAS",
    "extract_metadata_batch",
]
