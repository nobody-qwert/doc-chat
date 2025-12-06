"""Async SQLite persistence for documents, jobs, extractions, chunks, embeddings."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite
import numpy as np


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class EmbeddingRow:
    chunk_id: str
    doc_hash: str
    dim: int
    model: str
    vector: np.ndarray  # float32


if TYPE_CHECKING:  # pragma: no cover
    from .config import ChunkingConfigSpec


class DocumentStore:
    def __init__(self, db_path: Path, *, chunking_configs: Optional[Sequence["ChunkingConfigSpec"]] = None) -> None:
        self.db_path = Path(db_path)
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._chunking_configs: List["ChunkingConfigSpec"] = list(chunking_configs or [])

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self.db_path)
            try:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
                await conn.execute("PRAGMA foreign_keys=ON;")

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_hash TEXT PRIMARY KEY,
                        original_name TEXT NOT NULL,
                        stored_name TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        token_count INTEGER,
                        status TEXT NOT NULL,
                        error TEXT,
                        classification_status TEXT NOT NULL DEFAULT 'pending',
                        classification_error TEXT,
                        last_classified_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_ingested_at TEXT
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS extractions (
                        doc_hash TEXT NOT NULL,
                        parser TEXT NOT NULL,
                        text TEXT NOT NULL,
                        meta TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (doc_hash, parser),
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunking_configs (
                        config_id TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        description TEXT,
                        core_size INTEGER NOT NULL,
                        left_overlap INTEGER NOT NULL,
                        right_overlap INTEGER NOT NULL,
                        step_size INTEGER NOT NULL
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        chunk_config_id TEXT NOT NULL,
                        order_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE,
                        FOREIGN KEY(chunk_config_id) REFERENCES chunking_configs(config_id) ON DELETE RESTRICT
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        dim INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        doc_hash TEXT PRIMARY KEY,
                        ocr_time_sec REAL,
                        chunking_time_sec REAL,
                        embedding_time_sec REAL,
                        total_time_sec REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_classifications (
                        doc_hash TEXT PRIMARY KEY,
                        l1_id TEXT NOT NULL,
                        l1_name TEXT,
                        l2_id TEXT,
                        l2_name TEXT,
                        l1_confidence TEXT,
                        l2_confidence TEXT,
                        l1_reason TEXT,
                        l2_reason TEXT,
                        model TEXT,
                        raw_response TEXT,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        summary TEXT,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_ext_doc ON extractions(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_config ON chunks(chunk_config_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_doc ON embeddings(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_doc ON performance_metrics(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_msgs_conv ON conversation_messages(conversation_id, id)")

                if self._chunking_configs:
                    for spec in self._chunking_configs:
                        await conn.execute(
                            """
                            INSERT INTO chunking_configs (config_id, label, description, core_size, left_overlap, right_overlap, step_size)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(config_id) DO UPDATE SET
                                label=excluded.label,
                                description=excluded.description,
                                core_size=excluded.core_size,
                                left_overlap=excluded.left_overlap,
                                right_overlap=excluded.right_overlap,
                                step_size=excluded.step_size
                            """,
                            (
                                spec.config_id,
                                spec.label,
                                spec.description,
                                spec.core_size,
                                spec.left_overlap,
                                spec.right_overlap,
                                spec.step_size,
                            ),
                        )

                await _ensure_column(conn, "documents", "token_count", "INTEGER")

                await conn.commit()
            finally:
                await conn.close()
            self._initialized = True

    async def _conn(self) -> aiosqlite.Connection:
        await self.init()
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = aiosqlite.Row
        return conn

    # Documents
    async def upsert_document(self, *, doc_hash: str, original_name: str, stored_name: str, size: int) -> None:
        now = _utc_now()
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO documents (doc_hash, original_name, stored_name, size, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?)
                ON CONFLICT(doc_hash) DO UPDATE SET
                    original_name=excluded.original_name,
                    stored_name=excluded.stored_name,
                    size=excluded.size,
                    status=CASE WHEN documents.status='processed' THEN documents.status ELSE 'pending' END,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (doc_hash, original_name, stored_name, size, now, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def update_document_token_count(self, doc_hash: str, token_count: int) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE documents SET token_count=?, updated_at=? WHERE doc_hash=?",
                (int(token_count), _utc_now(), doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def update_document_status(self, doc_hash: str, status: str, *, error: Optional[str] = None) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE documents SET status=?, error=?, updated_at=? WHERE doc_hash=?",
                (status, error, _utc_now(), doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def mark_document_processed(self, doc_hash: str) -> None:
        conn = await self._conn()
        now = _utc_now()
        try:
            await conn.execute("DELETE FROM document_classifications WHERE doc_hash=?", (doc_hash,))
            await conn.execute(
                """
                UPDATE documents
                SET status='processed',
                    error=NULL,
                    classification_status='pending',
                    classification_error=NULL,
                    last_classified_at=NULL,
                    updated_at=?,
                    last_ingested_at=?
                WHERE doc_hash=?
                """,
                (now, now, doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_document(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute("SELECT * FROM documents WHERE doc_hash=?", (doc_hash,))
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()

    async def list_documents(self) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute("SELECT * FROM documents ORDER BY updated_at DESC")
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def count_documents(self, status: Optional[str] = None) -> int:
        conn = await self._conn()
        try:
            if status is None:
                cur = await conn.execute("SELECT COUNT(*) FROM documents")
            else:
                cur = await conn.execute("SELECT COUNT(*) FROM documents WHERE status=?", (status,))
            row = await cur.fetchone()
            await cur.close()
            return int(row[0] if row else 0)
        finally:
            await conn.close()

    async def delete_document(self, doc_hash: str) -> bool:
        conn = await self._conn()
        try:
            await conn.execute("BEGIN")
            for table in ("performance_metrics", "embeddings", "chunks", "extractions", "document_classifications", "jobs"):
                await conn.execute(f"DELETE FROM {table} WHERE doc_hash=?", (doc_hash,))
            cur = await conn.execute("DELETE FROM documents WHERE doc_hash=?", (doc_hash,))
            await conn.commit()
            return cur.rowcount > 0
        finally:
            await conn.close()

    # Jobs
    async def create_job(self, job_id: str, doc_hash: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "INSERT INTO jobs (job_id, doc_hash, status, error, created_at) VALUES (?, ?, 'queued', NULL, ?)",
                (job_id, doc_hash, _utc_now()),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def mark_job_started(self, job_id: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE jobs SET status='running', started_at=?, error=NULL WHERE job_id=?",
                (_utc_now(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def finish_job(self, job_id: str, status: str, *, error: Optional[str] = None) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE job_id=?",
                (status, error, _utc_now(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                """
                SELECT j.job_id, j.doc_hash, j.status, j.error, j.created_at, j.started_at, j.finished_at,
                       d.original_name as file
                FROM jobs j LEFT JOIN documents d ON d.doc_hash=j.doc_hash
                ORDER BY j.created_at DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def list_active_jobs(self) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT job_id, doc_hash, status FROM jobs WHERE status IN ('queued','running') ORDER BY created_at ASC"
            )
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    # Extractions
    async def upsert_extraction(self, doc_hash: str, parser: str, *, text: str, meta_json: Optional[str]) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO extractions (doc_hash, parser, text, meta, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash, parser) DO UPDATE SET text=excluded.text, meta=excluded.meta, created_at=excluded.created_at
                """,
                (doc_hash, parser, text, meta_json, _utc_now()),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_extraction(self, doc_hash: str, parser: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT doc_hash, parser, text, meta, created_at FROM extractions WHERE doc_hash=? AND parser=?",
                (doc_hash, parser),
            )
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()

    # Chunks
    async def replace_chunks(
        self,
        doc_hash: str,
        chunk_config_id: str,
        items: Sequence[Tuple[str, int, str, int]],
    ) -> None:
        """Replace chunk rows for a document and chunking configuration.

        items: iterable of (chunk_id, order_index, text, token_count)
        """
        conn = await self._conn()
        try:
            await conn.execute(
                "DELETE FROM chunks WHERE doc_hash=? AND chunk_config_id=?",
                (doc_hash, chunk_config_id),
            )
            await conn.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_hash, chunk_config_id, order_index, text, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(cid, doc_hash, chunk_config_id, idx, txt, tok) for (cid, idx, txt, tok) in items],
            )
            await conn.commit()
        finally:
            await conn.close()

    async def fetch_chunks(
        self,
        doc_hash: Optional[str] = None,
        chunk_config_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            base = "SELECT * FROM chunks"
            conditions: List[str] = []
            params: List[Any] = []
            if doc_hash:
                conditions.append("doc_hash=?")
                params.append(doc_hash)
            if chunk_config_id:
                conditions.append("chunk_config_id=?")
                params.append(chunk_config_id)
            if conditions:
                base += " WHERE " + " AND ".join(conditions)
            base += " ORDER BY doc_hash, order_index ASC"
            cur = await conn.execute(base, tuple(params))
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def fetch_chunks_by_ids(self, chunk_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        conn = await self._conn()
        try:
            placeholders = ",".join("?" for _ in chunk_ids)
            cur = await conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids))
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    # Embeddings
    @staticmethod
    def _pack_vector(v: np.ndarray) -> bytes:
        v = np.asarray(v, dtype=np.float32)
        return v.tobytes()

    @staticmethod
    def _unpack_vector(b: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32, count=dim)

    async def replace_embeddings(self, rows: Iterable[EmbeddingRow]) -> None:
        conn = await self._conn()
        try:
            # It's acceptable to delete by doc scope per batch; group rows
            rows_list = list(rows)
            if not rows_list:
                return
            doc_hashes = {r.doc_hash for r in rows_list}
            for dh in doc_hashes:
                await conn.execute("DELETE FROM embeddings WHERE doc_hash=?", (dh,))
            await conn.executemany(
                "INSERT INTO embeddings (chunk_id, doc_hash, dim, model, vector) VALUES (?, ?, ?, ?, ?)",
                [
                    (r.chunk_id, r.doc_hash, r.dim, r.model, self._pack_vector(r.vector))
                    for r in rows_list
                ],
            )
            await conn.commit()
        finally:
            await conn.close()

    async def fetch_embeddings_for_chunks(self, chunk_ids: Sequence[str]) -> List[EmbeddingRow]:
        if not chunk_ids:
            return []
        conn = await self._conn()
        try:
            # Build query with variable placeholders
            qs = ",".join(["?"] * len(chunk_ids))
            cur = await conn.execute(
                f"SELECT chunk_id, doc_hash, dim, model, vector FROM embeddings WHERE chunk_id IN ({qs})",
                tuple(chunk_ids),
            )
            rows = await cur.fetchall()
            await cur.close()
            out: List[EmbeddingRow] = []
            for r in rows:
                dim = int(r["dim"])  # type: ignore
                out.append(
                    EmbeddingRow(
                        chunk_id=r["chunk_id"],
                        doc_hash=r["doc_hash"],
                        dim=dim,
                        model=r["model"],
                        vector=self._unpack_vector(r["vector"], dim),
                    )
                )
            return out
        finally:
            await conn.close()

    async def fetch_all_embeddings(self) -> List[EmbeddingRow]:
        conn = await self._conn()
        try:
            cur = await conn.execute("SELECT chunk_id, doc_hash, dim, model, vector FROM embeddings")
            rows = await cur.fetchall()
            await cur.close()
            out: List[EmbeddingRow] = []
            for r in rows:
                dim = int(r["dim"])  # type: ignore[index]
                out.append(
                    EmbeddingRow(
                        chunk_id=r["chunk_id"],
                        doc_hash=r["doc_hash"],
                        dim=dim,
                        model=r["model"],
                        vector=self._unpack_vector(r["vector"], dim),
                    )
                )
            return out
        finally:
            await conn.close()

    async def count_embeddings_by_config(self, doc_hash: str) -> Dict[str, int]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                """
                SELECT c.chunk_config_id, COUNT(e.chunk_id) as total
                FROM embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                WHERE e.doc_hash=? AND c.doc_hash=?
                GROUP BY c.chunk_config_id
                """,
                (doc_hash, doc_hash),
            )
            rows = await cur.fetchall()
            await cur.close()
            return {str(row["chunk_config_id"]): int(row["total"]) for row in rows}
        finally:
            await conn.close()

    # Performance Metrics
    async def save_performance_metrics(
        self,
        doc_hash: str,
        *,
        ocr_time_sec: Optional[float] = None,
        chunking_time_sec: Optional[float] = None,
        embedding_time_sec: Optional[float] = None,
        total_time_sec: Optional[float] = None,
    ) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO performance_metrics (
                    doc_hash, ocr_time_sec, chunking_time_sec,
                    embedding_time_sec, total_time_sec, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash) DO UPDATE SET
                    ocr_time_sec=excluded.ocr_time_sec,
                    chunking_time_sec=excluded.chunking_time_sec,
                    embedding_time_sec=excluded.embedding_time_sec,
                    total_time_sec=excluded.total_time_sec,
                    created_at=excluded.created_at
                """,
                (
                    doc_hash,
                    ocr_time_sec,
                    chunking_time_sec,
                    embedding_time_sec,
                    total_time_sec,
                    _utc_now(),
                ),
            )
            await conn.commit()
        finally:
            await conn.close()

    # Classification
    async def update_classification_status(self, doc_hash: str, status: str, *, error: Optional[str] = None) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                UPDATE documents
                SET classification_status=?,
                    classification_error=?,
                    updated_at=?
                WHERE doc_hash=?
                """,
                (status, error, _utc_now(), doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def clear_classification(self, doc_hash: str) -> None:
        conn = await self._conn()
        now = _utc_now()
        try:
            await conn.execute("DELETE FROM document_classifications WHERE doc_hash=?", (doc_hash,))
            await conn.execute(
                """
                UPDATE documents
                SET classification_status='pending',
                    classification_error=NULL,
                    last_classified_at=NULL,
                    updated_at=?
                WHERE doc_hash=?
                """,
                (now, doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def save_document_classification(
        self,
        doc_hash: str,
        *,
        l1_id: str,
        l1_name: Optional[str],
        l2_id: Optional[str],
        l2_name: Optional[str],
        l1_confidence: Optional[str],
        l2_confidence: Optional[str],
        l1_reason: Optional[str],
        l2_reason: Optional[str],
        model: Optional[str],
        raw_response: Optional[Dict[str, Any]],
    ) -> None:
        conn = await self._conn()
        now = _utc_now()
        raw_json = json.dumps(raw_response or {}, ensure_ascii=False)
        try:
            await conn.execute(
                """
                INSERT INTO document_classifications (
                    doc_hash, l1_id, l1_name, l2_id, l2_name,
                    l1_confidence, l2_confidence, l1_reason, l2_reason,
                    model, raw_response, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash) DO UPDATE SET
                    l1_id=excluded.l1_id,
                    l1_name=excluded.l1_name,
                    l2_id=excluded.l2_id,
                    l2_name=excluded.l2_name,
                    l1_confidence=excluded.l1_confidence,
                    l2_confidence=excluded.l2_confidence,
                    l1_reason=excluded.l1_reason,
                    l2_reason=excluded.l2_reason,
                    model=excluded.model,
                    raw_response=excluded.raw_response,
                    updated_at=excluded.updated_at
                """,
                (
                    doc_hash,
                    l1_id,
                    l1_name,
                    l2_id,
                    l2_name,
                    l1_confidence,
                    l2_confidence,
                    l1_reason,
                    l2_reason,
                    model,
                    raw_json,
                    now,
                ),
            )
            await conn.execute(
                """
                UPDATE documents
                SET classification_status='classified',
                    classification_error=NULL,
                    last_classified_at=?,
                    updated_at=?
                WHERE doc_hash=?
                """,
                (now, now, doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_classification(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                """
                SELECT doc_hash, l1_id, l1_name, l2_id, l2_name,
                       l1_confidence, l2_confidence, l1_reason, l2_reason,
                       model, raw_response, updated_at
                FROM document_classifications
                WHERE doc_hash=?
                """,
                (doc_hash,),
            )
            row = await cur.fetchone()
            await cur.close()
            if not row:
                return None
            data = dict(row)
            raw = data.get("raw_response")
            if isinstance(raw, str):
                try:
                    data["raw_response"] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            return data
        finally:
            await conn.close()

    # Conversations
    async def create_conversation(self, conversation_id: str) -> None:
        now = _utc_now()
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO conversations (conversation_id, summary, total_tokens, created_at, updated_at)
                VALUES (?, '', 0, ?, ?)
                ON CONFLICT(conversation_id) DO NOTHING
                """,
                (conversation_id, now, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def delete_conversation(self, conversation_id: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
            await conn.commit()
        finally:
            await conn.close()

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await conn.close()

    async def append_conversation_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        token_count: int,
    ) -> None:
        now = _utc_now()
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO conversation_messages (conversation_id, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, token_count, now),
            )
            await conn.execute(
                "UPDATE conversations SET updated_at=?, total_tokens=total_tokens + ? WHERE conversation_id=?",
                (now, token_count, conversation_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def fetch_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                """
                SELECT conversation_id, role, content, token_count, created_at
                FROM conversation_messages
                WHERE conversation_id=?
                ORDER BY id ASC
                """,
                (conversation_id,),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def update_conversation_summary(self, conversation_id: str, summary: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE conversations SET summary=?, updated_at=? WHERE conversation_id=?",
                (summary, _utc_now(), conversation_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_performance_metrics(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT * FROM performance_metrics WHERE doc_hash=?",
                (doc_hash,),
            )
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()
async def _ensure_column(
    conn: aiosqlite.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Ensure a column exists on a table, creating it if missing."""
    cur = await conn.execute(f"PRAGMA table_info({table})")
    rows = await cur.fetchall()
    await cur.close()
    existing = {row[1] for row in rows}
    if column not in existing:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
