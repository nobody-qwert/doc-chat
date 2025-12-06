from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__:
    from .dependencies import lifespan, settings
    from .routes import agentic, chat, documents, extraction, ingest, system
else:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parent))
    from dependencies import lifespan, settings  # type: ignore
    from routes import agentic, chat, documents, extraction, ingest, system  # type: ignore

app = FastAPI(title="RAG Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(extraction.router)
app.include_router(agentic.router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
