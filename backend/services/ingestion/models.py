from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

JobPhase = str


@dataclass
class QueuedBatchDoc:
    doc_path: Path
    doc_hash: str
    display_name: str
    job_record_id: Optional[str] = None


@dataclass
class QueuedBatchJob:
    job_id: str
    docs: List[QueuedBatchDoc]
    phases: Sequence[JobPhase] = field(default_factory=list)
