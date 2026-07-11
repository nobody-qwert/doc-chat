from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"Missing required environment variable {name}")
    value = str(raw).strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} cannot be empty")
    return value


def _bool_env(name: str) -> bool:
    cleaned = _require_env(name).lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Environment variable {name} must be a boolean (got {cleaned!r})")


def _float_env(name: str) -> float:
    value = _require_env(name)
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float (got {value!r})") from exc


def _int_env(name: str) -> int:
    value = _require_env(name)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an int (got {value!r})") from exc


def _optional_float_env(name: str) -> Optional[float]:
    value = _optional_env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float (got {value!r})") from exc


def _optional_int_env(name: str) -> Optional[int]:
    value = _optional_env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an int (got {value!r})") from exc


def _optional_env(name: str) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


@dataclass
class QwenVLSettings:
    enabled: bool
    managed_server: bool
    api_base: str
    api_key: Optional[str]
    server_cmd: Optional[str]
    model_alias: str
    temperature: float
    top_k: int
    top_p: float
    repeat_penalty: float
    auto_load: bool
    shutdown_timeout: float
    request_timeout: float
    prompt: str

    @property
    def inference_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/chat/completions"

    @property
    def readiness_target(self) -> Optional[Tuple[str, int]]:
        cleaned = self.api_base.strip()
        if not cleaned:
            return None
        parsed = urlparse(cleaned if "://" in cleaned else f"http://{cleaned}")
        host = parsed.hostname
        if not host:
            return None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return host, port


def load_qwen_vl_settings() -> QwenVLSettings:
    stub = _bool_env("QWEN_VL_STUB")
    enabled = not stub
    managed_server = _bool_env("QWEN_VL_MANAGED_SERVER")

    api_base = _require_env("QWEN_VL_BASE_URL").rstrip("/")
    api_key = _optional_env("QWEN_VL_API_KEY")
    if stub or not managed_server:
        server_cmd = _optional_env("QWEN_VL_SERVER_CMD")
    else:
        # Default command for llama-server (with --parallel 1 to prevent memory slot exhaustion)
        default_cmd = (
            "fasz"
        )
        server_cmd = os.environ.get("QWEN_VL_SERVER_CMD")
        if not server_cmd:
            server_cmd = default_cmd
        else:
            server_cmd = server_cmd.strip()
        
        if not server_cmd:
             raise RuntimeError("QWEN_VL_SERVER_CMD must be set (or set QWEN_VL_STUB=1 to disable inference)")

    model_alias = _require_env("QWEN_VL_MODEL_ALIAS").strip()
    if not model_alias:
        raise RuntimeError("QWEN_VL_MODEL_ALIAS cannot be empty")

    temperature = _float_env("QWEN_VL_TEMPERATURE")
    top_k = _optional_int_env("QWEN_VL_TOP_K")
    top_p = _optional_float_env("QWEN_VL_TOP_P")
    repeat_penalty = _optional_float_env("QWEN_VL_REPEAT_PENALTY")
    auto_load = _bool_env("QWEN_VL_AUTO_LOAD")
    shutdown_timeout = _float_env("QWEN_VL_SERVER_SHUTDOWN_TIMEOUT")
    request_timeout = _float_env("QWEN_VL_REQUEST_TIMEOUT")
    prompt = _require_env("QWEN_VL_PROMPT")

    if enabled and managed_server and (not server_cmd or not server_cmd.strip()):
        raise RuntimeError("QWEN_VL_SERVER_CMD must be set (or set QWEN_VL_STUB=1 to disable inference)")

    return QwenVLSettings(
        enabled=enabled,
        managed_server=managed_server,
        api_base=api_base,
        api_key=api_key,
        server_cmd=server_cmd,
        model_alias=model_alias,
        temperature=float(temperature),
        top_k=int(top_k) if top_k is not None else 20,
        top_p=float(top_p) if top_p is not None else 0.95,
        repeat_penalty=float(repeat_penalty) if repeat_penalty is not None else 1.1,
        auto_load=bool(auto_load),
        shutdown_timeout=max(1.0, float(shutdown_timeout)),
        request_timeout=max(5.0, float(request_timeout)),
        prompt=prompt,
    )


class QwenVLServer:
    def __init__(self, settings: QwenVLSettings) -> None:
        self._settings = settings
        self._proc: Optional[subprocess.Popen[Any]] = None
        self._lock = asyncio.Lock()
        self._log_tail: deque[str] = deque(maxlen=250)
        self._last: Dict[str, Any] = {
            "status": "stopped",
            "pid": None,
            "last_start": None,
            "last_stop": None,
            "last_exit_code": None,
        }
        self._watchers: set[asyncio.Task[Any]] = set()

    def status(self) -> Dict[str, Any]:
        payload = dict(self._last)
        payload["running"] = self._proc is not None and self._proc.poll() is None
        payload["pid"] = self._proc.pid if payload["running"] else None
        payload["api_base"] = self._settings.api_base
        payload["enabled"] = self._settings.enabled
        payload["managed_server"] = self._settings.managed_server
        payload["recent_logs"] = list(self._log_tail)
        return payload

    async def start(self, reason: str) -> Dict[str, Any]:
        if not self._settings.enabled:
            return {"state": "stub", "running": False}
        if not self._settings.managed_server:
            await self._wait_ready()
            self._last.update({
                "status": "external",
                "last_start": time.time(),
                "last_exit_code": None,
            })
            return self.status()
        async with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                logger.info("Qwen3-VL server already running (reason=%s)", reason)
                return self.status()

            server_cmd = self._settings.server_cmd
            if not server_cmd:
                raise RuntimeError("QWEN_VL_SERVER_CMD is not configured")
            cmd = shlex.split(server_cmd)
            logger.info("Starting Qwen3-VL server: %s", " ".join(cmd))

            def _spawn() -> subprocess.Popen[Any]:
                return subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )

            proc = await asyncio.to_thread(_spawn)
            self._log_tail.clear()
            self._start_log_thread(proc)
            self._proc = proc
            self._last.update({
                "status": "running",
                "last_start": time.time(),
                "last_exit_code": None,
            })
            task = asyncio.create_task(self._monitor(proc))
            self._watchers.add(task)
            task.add_done_callback(self._watchers.discard)

        await self._wait_ready()
        return self.status()

    async def stop(self, reason: str) -> Dict[str, Any]:
        if not self._settings.managed_server:
            self._last.update({"status": "external", "last_stop": time.time()})
            return self.status()
        async with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._proc = None
                self._last.update({"status": "stopped", "last_stop": time.time()})
                return self.status()

            logger.info("Stopping Qwen3-VL server pid=%s (reason=%s)", proc.pid, reason)

            def _terminate() -> None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=self._settings.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Qwen3-VL server did not exit in %.1fs, killing",
                        self._settings.shutdown_timeout,
                    )
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                    proc.wait(timeout=2)

            await asyncio.to_thread(_terminate)
            self._last.update({
                "status": "stopped",
                "last_stop": time.time(),
                "last_exit_code": proc.returncode,
            })
            self._proc = None
            return self.status()

    async def _monitor(self, proc: subprocess.Popen[Any]) -> None:
        try:
            await asyncio.to_thread(proc.wait)
        finally:
            async with self._lock:
                if self._proc is proc:
                    logger.warning("Qwen3-VL server pid=%s exited with code %s", proc.pid, proc.returncode)
                    self._last.update({
                        "status": "stopped",
                        "last_stop": time.time(),
                        "last_exit_code": proc.returncode,
                    })
                    self._proc = None

    async def _wait_ready(self) -> None:
        target = self._settings.readiness_target
        if not target:
            return
        host, port = target
        # Use a longer deadline for model loading - large VL models can take 60+ seconds
        deadline = time.perf_counter() + max(120.0, self._settings.request_timeout * 2)
        last_error: Optional[Exception] = None
        models_url = f"{self._settings.api_base.rstrip('/')}/models"

        # First wait for TCP port to be available
        port_ready = False
        while time.perf_counter() < deadline and not port_ready:
            if self._settings.managed_server and self._proc is not None:
                exit_code = self._proc.poll()
                if exit_code is not None:
                    tail = "\n".join(list(self._log_tail)[-80:]).strip()
                    hint = ""
                    if "unknown model architecture" in tail:
                        hint = (
                            "\nHint: llama.cpp in this image likely does not support this GGUF architecture yet. "
                            "Try rebuilding with a newer llama-cpp-python/llama.cpp (set LLAMA_CPP_PYTHON_REF and bump "
                            "LLAMA_CPP_PYTHON_CACHE_BUST), or use a model architecture that your current llama.cpp supports."
                        )
                    raise RuntimeError(
                        f"Qwen3-VL server exited before becoming ready (exit_code={exit_code})."
                        f"{hint}\n--- server log tail ---\n{tail}"
                    )
            try:
                connect = asyncio.open_connection(host, port)
                reader, writer = await asyncio.wait_for(connect, timeout=2.0)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                port_ready = True
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)
        
        if not port_ready:
            if last_error:
                raise RuntimeError(f"Qwen3-VL server port not ready on {host}:{port}: {last_error}") from last_error
            raise RuntimeError(f"Qwen3-VL server port not ready on {host}:{port}")
        
        # The shared llama-cpp server exposes the OpenAI-compatible /v1/models
        # endpoint, but not /health. A non-empty model list means its model is
        # loaded and it can accept OCR completion requests.
        logger.info("Port %s:%s is open, waiting for model to load (checking /models)...", host, port)
        timeout = httpx.Timeout(5.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while time.perf_counter() < deadline:
                if self._settings.managed_server and self._proc is not None:
                    exit_code = self._proc.poll()
                    if exit_code is not None:
                        tail = "\n".join(list(self._log_tail)[-80:]).strip()
                        raise RuntimeError(
                            f"Qwen3-VL server exited while loading model (exit_code={exit_code})."
                            f"\n--- server log tail ---\n{tail}"
                        )
                try:
                    resp = await client.get(models_url)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("data")
                        if isinstance(models, list) and models:
                            logger.info("Qwen3-VL server is ready (models=%s)", len(models))
                            return
                        logger.debug("Model list is empty or invalid, waiting...")
                    else:
                        logger.debug("Model readiness check returned %s, waiting...", resp.status_code)
                except Exception as exc:
                    last_error = exc
                    logger.debug("Model readiness check failed: %s, waiting...", exc)
                await asyncio.sleep(1.0)
        
        if last_error:
            raise RuntimeError(f"Qwen3-VL server not ready (model check failed): {last_error}") from last_error
        raise RuntimeError("Qwen3-VL server not ready (model check timed out)")

    def _start_log_thread(self, proc: subprocess.Popen[Any]) -> None:
        stdout = proc.stdout
        if stdout is None:
            return

        def _reader() -> None:
            try:
                for line in stdout:
                    cleaned = line.rstrip("\n")
                    self._log_tail.append(cleaned)
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except Exception:
                        pass
            finally:
                try:
                    stdout.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_reader, name="qwen3_vl_server_log", daemon=True)
        thread.start()


def _image_variants(image_bytes: bytes) -> list[dict[str, Any]]:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{image_b64}"
    # Use only the standard OpenAI-compatible format
    # llama-server expects: {"type": "image_url", "image_url": {"url": "..."}}
    return [
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


async def qwen_vl_ocr_image(
    *,
    settings: QwenVLSettings,
    image_bytes: bytes,
    prompt: Optional[str] = None,
) -> str:
    if not settings.enabled:
        return "[stubbed qwen3_vl ocr]\n"

    user_prompt = (prompt or settings.prompt).strip()
    if not user_prompt:
        raise RuntimeError("QWEN_VL_PROMPT cannot be empty")

    headers: Dict[str, str] = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    image_size_kb = len(image_bytes) / 1024.0

    timeout = httpx.Timeout(settings.request_timeout, connect=min(10.0, settings.request_timeout))
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        last_error: Optional[Exception] = None
        for idx, image_block in enumerate(_image_variants(image_bytes), start=1):
            payload = {
                "model": settings.model_alias,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            image_block,
                        ],
                    }
                ],
                "temperature": float(settings.temperature),
                "top_k": int(settings.top_k),
                "top_p": float(settings.top_p),
                "repeat_penalty": float(settings.repeat_penalty),
            }
            try:
                logger.debug(
                    "Qwen VL inference attempt %d/%d: image_size_kb=%.1f",
                    idx,
                    len(_image_variants(image_bytes)),
                    image_size_kb,
                )
                resp = await client.post(settings.inference_url, json=payload)
                if resp.status_code == 400:
                    error_text = resp.text
                    logger.warning("Qwen VL inference attempt %d failed with 400: %s", idx, error_text[:200])
                    last_error = RuntimeError(error_text)
                    continue
                if resp.status_code == 503:
                    error_text = resp.text
                    logger.warning("Qwen VL inference attempt %d failed with 503 (server busy): %s", idx, error_text[:200])
                    last_error = RuntimeError(f"Server busy (503): {error_text}")
                    continue
                resp.raise_for_status()
                data = resp.json()

                # Log token usage from the response
                usage = data.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", "N/A")
                completion_tokens = usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")
                logger.info(
                    "Qwen VL inference: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, image_size_kb=%.1f",
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    image_size_kb,
                )

                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("No choices in response")
                message = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                    if parts:
                        return "\n".join(parts)
                raise RuntimeError("Unrecognized chat completion response shape")
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Qwen3-VL inference failed: {last_error}") from last_error
