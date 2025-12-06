import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import psutil
from fastapi import FastAPI, HTTPException

try:
    import pynvml
except Exception:  # pragma: no cover - runtime only
    pynvml = None


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _gpu_snapshot() -> Dict[str, Any]:
    if pynvml is None:
        raise RuntimeError("pynvml unavailable")
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(f"nvml init failed: {exc}")

    gpus: List[Dict[str, Any]] = []
    processes: List[Dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8", errors="ignore") if isinstance(raw_name, bytes) else str(raw_name)
            gpus.append(
                {
                    "index": idx,
                    "uuid": uuid,
                    "name": name,
                    "memory_used_mb": mem.used / (1024 * 1024),
                    "memory_total_mb": mem.total / (1024 * 1024),
                    "utilization_gpu": util.gpu,
                    "utilization_memory": util.memory,
                }
            )

            proc_getters = [
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
            ]
            proc_list = None
            for getter in proc_getters:
                if getter is None:
                    continue
                try:
                    proc_list = getter(handle)
                    break
                except pynvml.NVMLError:  # type: ignore[attr-defined]
                    continue
            if not proc_list:
                continue
            for proc in proc_list:
                used_mem = getattr(proc, "usedGpuMemory", 0)
                pid = getattr(proc, "pid", None)
                process_name = getattr(proc, "name", "")
                if hasattr(pynvml, "nvmlSystemGetProcessName") and pid is not None and not process_name:
                    try:
                        buf = pynvml.nvmlSystemGetProcessName(pid)
                        process_name = buf.decode("utf-8", errors="ignore") if isinstance(buf, bytes) else str(buf)
                    except TypeError:
                        try:
                            buf = pynvml.nvmlSystemGetProcessName(pid, 256)  # legacy signature
                            process_name = buf.decode("utf-8", errors="ignore") if isinstance(buf, bytes) else str(buf)
                        except Exception:
                            process_name = ""
                    except pynvml.NVMLError:  # type: ignore[attr-defined]
                        process_name = ""
                processes.append(
                    {
                        "gpu_uuid": uuid,
                        "gpu_index": idx,
                        "pid": pid,
                        "name": process_name,
                        "used_memory_mb": (used_mem or 0) / (1024 * 1024),
                    }
                )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return {"gpus": gpus, "processes": processes}


def _system_snapshot() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "cpu_percent": cpu_percent,
        "cpu_load": load_avg,
        "memory_used_mb": vm.used / (1024 * 1024),
        "memory_total_mb": vm.total / (1024 * 1024),
    }


app = FastAPI(title="Diagnostics Service", version="0.1.0")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/gpu")
async def gpu_route() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"timestamp": _utc_now()}
    try:
        payload.update(_gpu_snapshot())
        payload["error"] = None
    except Exception as exc:
        payload.update({"gpus": [], "processes": [], "error": str(exc)})
    payload["system"] = _system_snapshot()
    return payload


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "9001"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
