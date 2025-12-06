from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


LLM_SERVER_CMD = os.environ.get("LLM_SERVER_CMD")
if not LLM_SERVER_CMD:
    raise RuntimeError("LLM_SERVER_CMD must be set (e.g., 'python3 -m llama_cpp.server ...')")

CONTROL_PORT = int(os.environ.get("LLM_CONTROL_PORT", "9000"))
AUTO_LOAD = _env_bool("LLM_AUTO_LOAD", True)
SHUTDOWN_TIMEOUT = float(os.environ.get("LLM_SERVER_SHUTDOWN_TIMEOUT", "30"))

logger = logging.getLogger("llm_controller")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(title="LLM Controller", version="0.1.0")

_process: Optional[subprocess.Popen[Any]] = None
_process_lock = asyncio.Lock()
_watcher_tasks: set[asyncio.Task[Any]] = set()
_last_status: Dict[str, Any] = {
    "status": "stopped",
    "pid": None,
    "last_start": None,
    "last_stop": None,
    "last_exit_code": None,
}


def _status_payload() -> Dict[str, Any]:
    payload = dict(_last_status)
    payload["running"] = _process is not None and _process.poll() is None
    if payload["running"]:
        payload["pid"] = _process.pid
    else:
        payload["pid"] = None
    return payload


async def _start_llm(reason: str) -> Dict[str, Any]:
    global _process
    async with _process_lock:
        if _process is not None and _process.poll() is None:
            logger.info("LLM already running (reason=%s)", reason)
            return _status_payload()

        cmd = shlex.split(LLM_SERVER_CMD)
        logger.info("Starting LLM process: %s", " ".join(cmd))

        def _spawn() -> subprocess.Popen[Any]:
            return subprocess.Popen(cmd, stdout=None, stderr=None, preexec_fn=os.setsid)

        proc = await asyncio.to_thread(_spawn)
        _process = proc
        _last_status.update({
            "status": "running",
            "last_start": datetime.utcnow().isoformat(),
            "last_exit_code": None,
        })
        task = asyncio.create_task(_monitor_process(proc))
        _watcher_tasks.add(task)
        task.add_done_callback(_watcher_tasks.discard)
        return _status_payload()


async def _stop_llm(reason: str) -> Dict[str, Any]:
    global _process
    async with _process_lock:
        proc = _process
        if proc is None or proc.poll() is not None:
            _process = None
            _last_status.update({"status": "stopped", "last_stop": datetime.utcnow().isoformat()})
            return _status_payload()

        logger.info("Stopping LLM process pid=%s (reason=%s)", proc.pid, reason)

        def _terminate() -> None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("LLM process did not exit in %.1fs, killing", SHUTDOWN_TIMEOUT)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=10)

        await asyncio.to_thread(_terminate)
        _last_status.update({
            "status": "stopped",
            "last_stop": datetime.utcnow().isoformat(),
            "last_exit_code": proc.returncode,
        })
        _process = None
        return _status_payload()


async def _monitor_process(proc: subprocess.Popen[Any]) -> None:
    try:
        await asyncio.to_thread(proc.wait)
    finally:
        async with _process_lock:
            if _process is proc:
                logger.warning("LLM process pid=%s exited unexpectedly with code %s", proc.pid, proc.returncode)
                _last_status.update({
                    "status": "stopped",
                    "last_stop": datetime.utcnow().isoformat(),
                    "last_exit_code": proc.returncode,
                })
                _process = None


@app.on_event("startup")
async def _startup_event() -> None:
    if AUTO_LOAD:
        try:
            await _start_llm("startup")
        except Exception as exc:  # pragma: no cover - startup guard
            logger.error("Failed to auto-load LLM: %s", exc)


@app.post("/control/load")
async def control_load() -> Dict[str, Any]:
    try:
        return await _start_llm("manual")
    except Exception as exc:
        logger.exception("Failed to load LLM")
        raise HTTPException(status_code=500, detail=f"Failed to load LLM: {exc}") from exc


@app.post("/control/unload")
async def control_unload() -> Dict[str, Any]:
    try:
        return await _stop_llm("manual")
    except Exception as exc:
        logger.exception("Failed to unload LLM")
        raise HTTPException(status_code=500, detail=f"Failed to unload LLM: {exc}") from exc


@app.get("/control/status")
async def control_status() -> Dict[str, Any]:
    return _status_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm_controller:app", host="0.0.0.0", port=CONTROL_PORT)
