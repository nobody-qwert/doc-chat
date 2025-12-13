from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def _configure_logging() -> None:
    level_name = (os.getenv("APP_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger.addHandler(handler)


_configure_logging()

if __package__:
    from .dependencies import lifespan, settings
    from .routes import agentic, chat, documents, ingest, system
else:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parent))
    from dependencies import lifespan, settings  # type: ignore
    from routes import agentic, chat, documents, ingest, system  # type: ignore

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
app.include_router(agentic.router)


if __name__ == "__main__":
    import uvicorn

    raw_port = os.environ.get("BACKEND_PORT")
    if raw_port is None or not raw_port.strip():
        raise RuntimeError("BACKEND_PORT environment variable must be set")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"BACKEND_PORT must be an integer (got {raw_port!r})") from exc
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
