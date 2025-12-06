"""
Generic CSV-based extraction engine.

Uses LLM to extract structured metadata from documents in CSV format
for token efficiency (~30-40% savings compared to JSON).
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .schemas import ExtractionSchema, FieldSpec

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extracting metadata from a single document."""
    doc_hash: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None


def build_extraction_prompt(
    schema: ExtractionSchema,
    document_text: str,
    max_text_chars: int = 8000,
) -> Tuple[str, str]:
    """
    Build system and user prompts for CSV-based extraction.
    
    Args:
        schema: The extraction schema
        document_text: Full text of the document
        max_text_chars: Max characters to include from document
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Truncate document if too long
    if len(document_text) > max_text_chars:
        document_text = document_text[:max_text_chars] + "\n...[TRUNCATED]..."
    
    headers = schema.get_csv_headers()
    field_descriptions = schema.get_field_descriptions()
    
    system_prompt = f"""You are a precise data extraction assistant. Your task is to extract structured metadata from documents and output it in CSV format.

EXTRACTION SCHEMA: {schema.display_name}
{schema.description}

FIELDS TO EXTRACT:
{field_descriptions}

OUTPUT FORMAT:
- Output ONLY a valid CSV block with headers on the first line
- Use the exact column names provided
- For missing/unknown values, leave the field empty (not "N/A" or "unknown")
- For dates, use ISO format: YYYY-MM-DD
- For numbers, output only the numeric value without currency symbols or commas
- Wrap values containing commas in double quotes
- Output exactly ONE data row (for one document)

EXAMPLE OUTPUT FORMAT:
```csv
{','.join(headers)}
value1,value2,value3,...
```"""

    user_prompt = f"""Extract metadata from the following document. Output ONLY the CSV block, nothing else.

DOCUMENT:
---
{document_text}
---

Output the CSV with headers and one data row:"""

    return system_prompt, user_prompt


def parse_csv_response(
    response_text: str,
    schema: ExtractionSchema,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse LLM CSV response into a dictionary.
    
    Args:
        response_text: Raw LLM response
        schema: The extraction schema for validation
    
    Returns:
        Tuple of (parsed_data, error_message)
    """
    # Try to extract CSV block from response
    csv_text = response_text.strip()
    
    # Look for ```csv ... ``` block
    csv_match = re.search(r'```(?:csv)?\s*\n?(.*?)```', csv_text, re.DOTALL | re.IGNORECASE)
    if csv_match:
        csv_text = csv_match.group(1).strip()
    
    # Also try to find just the CSV part if no code block
    if not csv_match:
        # Look for lines that look like CSV (contain commas, start with header)
        lines = csv_text.split('\n')
        csv_lines = []
        in_csv = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if this looks like our header line
            if not in_csv and any(h in line for h in schema.get_csv_headers()[:3]):
                in_csv = True
            if in_csv:
                csv_lines.append(line)
                if len(csv_lines) >= 2:  # Header + 1 data row
                    break
        if csv_lines:
            csv_text = '\n'.join(csv_lines)
    
    if not csv_text:
        return None, "No CSV content found in response"
    
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        
        if not rows:
            return None, "CSV has no data rows"
        
        row = rows[0]  # Take first row
        
        # Map to schema fields and convert types
        data: Dict[str, Any] = {}
        expected_fields = {f.name for f in schema.fields}
        
        for field in schema.fields:
            value = row.get(field.name, "").strip()
            
            if not value:
                data[field.name] = None
                continue
            
            # Type conversion
            if field.sql_type == "REAL":
                try:
                    # Remove common formatting
                    clean_value = re.sub(r'[,$€£¥]', '', value)
                    clean_value = clean_value.replace(' ', '')
                    data[field.name] = float(clean_value)
                except ValueError:
                    data[field.name] = None
                    logger.warning(f"Could not convert '{value}' to float for {field.name}")
            elif field.sql_type == "INTEGER":
                try:
                    clean_value = re.sub(r'[,$]', '', value)
                    data[field.name] = int(float(clean_value))
                except ValueError:
                    data[field.name] = None
            else:
                # TEXT or DATE - keep as string
                data[field.name] = value
        
        return data, None
        
    except csv.Error as e:
        return None, f"CSV parsing error: {e}"
    except Exception as e:
        return None, f"Unexpected parsing error: {e}"


async def extract_single_document(
    doc_hash: str,
    document_text: str,
    schema: ExtractionSchema,
    llm_client: Any,  # OpenAI-compatible async client
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> ExtractionResult:
    """
    Extract metadata from a single document.
    
    Args:
        doc_hash: Document identifier
        document_text: Full text content
        schema: Extraction schema to use
        llm_client: OpenAI-compatible async client
        model: Model name to use
        temperature: LLM temperature (low for extraction)
        max_tokens: Max response tokens
    
    Returns:
        ExtractionResult with success status and data
    """
    if not document_text or not document_text.strip():
        return ExtractionResult(
            doc_hash=doc_hash,
            success=False,
            error="Empty document text",
        )
    
    system_prompt, user_prompt = build_extraction_prompt(schema, document_text)
    
    try:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if not response.choices:
            return ExtractionResult(
                doc_hash=doc_hash,
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data, parse_error = parse_csv_response(raw_response, schema)
        
        if parse_error:
            return ExtractionResult(
                doc_hash=doc_hash,
                success=False,
                error=parse_error,
                raw_response=raw_response,
            )
        
        return ExtractionResult(
            doc_hash=doc_hash,
            success=True,
            data=data,
            raw_response=raw_response,
        )
        
    except Exception as e:
        logger.exception(f"Extraction failed for {doc_hash}: {e}")
        return ExtractionResult(
            doc_hash=doc_hash,
            success=False,
            error=str(e),
        )


async def extract_metadata_batch(
    documents: List[Dict[str, Any]],  # List of {doc_hash, text}
    schema: ExtractionSchema,
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> List[ExtractionResult]:
    """
    Extract metadata from a batch of documents.
    
    Args:
        documents: List of dicts with 'doc_hash' and 'text' keys
        schema: Extraction schema to use
        llm_client: OpenAI-compatible async client
        model: Model name
        temperature: LLM temperature
        max_tokens: Max response tokens
    
    Returns:
        List of ExtractionResult objects
    """
    results = []
    
    for doc in documents:
        doc_hash = doc.get("doc_hash", "")
        text = doc.get("text", "")
        
        result = await extract_single_document(
            doc_hash=doc_hash,
            document_text=text,
            schema=schema,
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        results.append(result)
        
        if result.success:
            logger.info(f"Successfully extracted metadata for {doc_hash}")
        else:
            logger.warning(f"Failed to extract metadata for {doc_hash}: {result.error}")
    
    return results
