"""
Dynamic table management for extraction metadata.

Handles:
- Dynamic table creation based on extraction schemas
- Inserting/updating extracted metadata
- Querying metadata with filters for agentic RAG
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiosqlite

from .schemas import ExtractionSchema, get_schema, EXTRACTION_SCHEMAS

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class MetadataStore:
    """
    Manages dynamic metadata tables for extracted document information.
    
    Works alongside the main DocumentStore but handles the schema-driven
    extraction tables (meta_invoices, meta_contracts, etc.)
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._initialized_tables: set[str] = set()

    async def _conn(self) -> aiosqlite.Connection:
        """Get database connection with proper settings."""
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = aiosqlite.Row
        return conn

    async def ensure_table_exists(self, schema: ExtractionSchema) -> None:
        """
        Ensure the metadata table for a schema exists.
        Creates it if missing.
        """
        if schema.table_name in self._initialized_tables:
            return

        conn = await self._conn()
        try:
            # Generate and execute CREATE TABLE
            create_sql = schema.to_create_table_sql()
            await conn.execute(create_sql)
            
            # Create index on common filter columns
            for field in schema.fields:
                if field.sql_type in ("TEXT", "REAL", "INTEGER"):
                    idx_name = f"idx_{schema.table_name}_{field.name}"
                    try:
                        await conn.execute(
                            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {schema.table_name}({field.name})"
                        )
                    except Exception as e:
                        logger.debug(f"Index creation skipped for {field.name}: {e}")
            
            await conn.commit()
            self._initialized_tables.add(schema.table_name)
            logger.info(f"Ensured metadata table exists: {schema.table_name}")
        finally:
            await conn.close()

    async def ensure_all_tables_exist(self) -> None:
        """Ensure all registered schema tables exist."""
        for schema in EXTRACTION_SCHEMAS.values():
            await self.ensure_table_exists(schema)

    async def upsert_metadata(
        self,
        schema: ExtractionSchema,
        doc_hash: str,
        data: Dict[str, Any],
        model: Optional[str] = None,
    ) -> None:
        """
        Insert or update metadata for a document.
        
        Args:
            schema: The extraction schema
            doc_hash: Document hash (primary key)
            data: Dict of field_name -> value
            model: LLM model used for extraction
        """
        await self.ensure_table_exists(schema)
        
        conn = await self._conn()
        try:
            # Build column list and values
            columns = ["doc_hash", "extracted_at", "extraction_model"]
            values: List[Any] = [doc_hash, _utc_now(), model]
            
            for field in schema.fields:
                columns.append(field.name)
                values.append(data.get(field.name))
            
            placeholders = ", ".join("?" for _ in columns)
            col_list = ", ".join(columns)
            
            # Build ON CONFLICT update clause
            update_parts = [f"{c}=excluded.{c}" for c in columns if c != "doc_hash"]
            update_clause = ", ".join(update_parts)
            
            sql = f"""
                INSERT INTO {schema.table_name} ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(doc_hash) DO UPDATE SET {update_clause}
            """
            
            await conn.execute(sql, tuple(values))
            await conn.commit()
            logger.debug(f"Upserted metadata for {doc_hash} in {schema.table_name}")
        finally:
            await conn.close()

    async def get_metadata(
        self,
        schema: ExtractionSchema,
        doc_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a single document."""
        await self.ensure_table_exists(schema)
        
        conn = await self._conn()
        try:
            cur = await conn.execute(
                f"SELECT * FROM {schema.table_name} WHERE doc_hash = ?",
                (doc_hash,)
            )
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()

    async def query_metadata(
        self,
        schema: ExtractionSchema,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query metadata table with optional filters.
        
        Args:
            schema: The extraction schema to query
            filters: Dict of field_name -> filter_spec
                     Simple: {"vendor_name": "ACME"} -> exact match
                     Operators: {"total_amount": {">": 1000}} -> comparison
                     Like: {"vendor_name": {"like": "%Corp%"}}
            limit: Max results to return
        
        Returns:
            List of matching metadata records
        """
        await self.ensure_table_exists(schema)
        
        conn = await self._conn()
        try:
            where_clauses: List[str] = []
            params: List[Any] = []
            
            if filters:
                for field_name, filter_spec in filters.items():
                    # Validate field exists in schema
                    valid_fields = {f.name for f in schema.fields}
                    valid_fields.add("doc_hash")
                    if field_name not in valid_fields:
                        continue
                    
                    if isinstance(filter_spec, dict):
                        # Operator-based filter
                        for op, value in filter_spec.items():
                            op_lower = op.lower()
                            if op_lower == "like":
                                where_clauses.append(f"{field_name} LIKE ?")
                                params.append(value)
                            elif op_lower in (">", ">=", "<", "<=", "=", "!=", "<>"):
                                where_clauses.append(f"{field_name} {op} ?")
                                params.append(value)
                    else:
                        # Exact match
                        where_clauses.append(f"{field_name} = ?")
                        params.append(filter_spec)
            
            sql = f"SELECT * FROM {schema.table_name}"
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            sql += f" LIMIT {limit}"
            
            cur = await conn.execute(sql, tuple(params))
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def delete_metadata(self, schema: ExtractionSchema, doc_hash: str) -> bool:
        """Delete metadata for a document."""
        await self.ensure_table_exists(schema)
        
        conn = await self._conn()
        try:
            cur = await conn.execute(
                f"DELETE FROM {schema.table_name} WHERE doc_hash = ?",
                (doc_hash,)
            )
            await conn.commit()
            return cur.rowcount > 0
        finally:
            await conn.close()

    async def count_extracted(self, schema: ExtractionSchema) -> int:
        """Count documents with extracted metadata for a schema."""
        await self.ensure_table_exists(schema)
        
        conn = await self._conn()
        try:
            cur = await conn.execute(f"SELECT COUNT(*) FROM {schema.table_name}")
            row = await cur.fetchone()
            await cur.close()
            return int(row[0]) if row else 0
        finally:
            await conn.close()

    async def list_extracted_doc_hashes(
        self,
        schema: ExtractionSchema,
    ) -> List[str]:
        """Get all doc_hashes that have been extracted for a schema."""
        await self.ensure_table_exists(schema)
        
        conn = await self._conn()
        try:
            cur = await conn.execute(
                f"SELECT doc_hash FROM {schema.table_name}"
            )
            rows = await cur.fetchall()
            await cur.close()
            return [row["doc_hash"] for row in rows]
        finally:
            await conn.close()


async def query_chunks_with_metadata_filter(
    db_path: str,
    schema: ExtractionSchema,
    metadata_filters: Optional[Dict[str, Any]] = None,
    text_search: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Query chunks joined with metadata filters.
    
    This is used by the agentic RAG tools to filter chunks based on
    extracted metadata (e.g., "invoices with amount > 1000").
    
    Args:
        db_path: Path to SQLite database
        schema: Extraction schema for metadata table
        metadata_filters: Filters for the metadata table
        text_search: Optional text to search in chunk content (LIKE match)
        limit: Max results
    
    Returns:
        List of chunks with their metadata
    """
    meta_store = MetadataStore(db_path)
    await meta_store.ensure_table_exists(schema)
    
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = aiosqlite.Row
    
    try:
        # Build query with JOIN
        where_clauses: List[str] = []
        params: List[Any] = []
        
        # Metadata filters
        if metadata_filters:
            for field_name, filter_spec in metadata_filters.items():
                valid_fields = {f.name for f in schema.fields}
                valid_fields.add("doc_hash")
                if field_name not in valid_fields:
                    continue
                
                if isinstance(filter_spec, dict):
                    for op, value in filter_spec.items():
                        op_lower = op.lower()
                        if op_lower == "like":
                            where_clauses.append(f"m.{field_name} LIKE ?")
                            params.append(value)
                        elif op_lower in (">", ">=", "<", "<=", "=", "!=", "<>"):
                            where_clauses.append(f"m.{field_name} {op} ?")
                            params.append(value)
                else:
                    where_clauses.append(f"m.{field_name} = ?")
                    params.append(filter_spec)
        
        # Text search in chunks
        if text_search:
            where_clauses.append("c.text LIKE ?")
            params.append(f"%{text_search}%")
        
        sql = f"""
            SELECT 
                c.chunk_id,
                c.doc_hash,
                c.order_index,
                c.text,
                c.token_count,
                d.original_name,
                m.*
            FROM chunks c
            JOIN documents d ON d.doc_hash = c.doc_hash
            JOIN {schema.table_name} m ON m.doc_hash = c.doc_hash
        """
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        sql += f" ORDER BY c.doc_hash, c.order_index LIMIT {limit}"
        
        cur = await conn.execute(sql, tuple(params))
        rows = await cur.fetchall()
        await cur.close()
        return [dict(r) for r in rows]
    finally:
        await conn.close()
