from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import psutil

from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import RuntimeStatus, SystemLiveMetrics


_GPU_COUNTER_CACHE_TTL_SECONDS = 2.0
_GPU_COUNTER_CACHE: tuple[float, float | None, float | None, str | None, list[str]] | None = None


def build_runtime_status(
    paths: WorkspacePaths,
    gpu_names: list[str] | None = None,
    driver_version: str | None = None,
) -> RuntimeStatus:
    detected_warnings: list[str] = []
    names = gpu_names
    amd_driver_version = driver_version

    if names is None:
        names, detected_driver_version, detected_warnings = _detect_windows_gpus()
        amd_driver_version = detected_driver_version

    comfyui_path_exists = (paths.root / "ComfyUI").exists()
    warnings = list(detected_warnings)

    if not _has_amd_radeon_gpu(names):
        warnings.append("No AMD Radeon GPU detected.")

    if not comfyui_path_exists:
        warnings.append(f"ComfyUI was not found at {paths.root / 'ComfyUI'}.")

    return RuntimeStatus(
        os_name=platform.platform(),
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        cpu_name=_cpu_name(),
        total_ram_gb=round(psutil.virtual_memory().total / (1024**3), 1),
        gpu_names=names,
        amd_driver_version=amd_driver_version,
        comfyui_path_exists=comfyui_path_exists,
        warnings=warnings,
    )


def build_live_system_metrics() -> SystemLiveMetrics:
    memory = psutil.virtual_memory()
    process = psutil.Process()
    gpu_percent, dedicated_bytes, provider, warnings = _cached_windows_gpu_counters()
    dedicated_gb = (
        round(dedicated_bytes / (1024**3), 2)
        if dedicated_bytes is not None
        else None
    )

    return SystemLiveMetrics(
        cpu_percent=round(psutil.cpu_percent(interval=None), 1),
        ram_used_gb=round(memory.used / (1024**3), 1),
        ram_total_gb=round(memory.total / (1024**3), 1),
        ram_percent=round(memory.percent, 1),
        gpu_percent=round(gpu_percent, 1) if gpu_percent is not None else None,
        gpu_memory_used_gb=dedicated_gb,
        gpu_memory_total_gb=None,
        gpu_provider=provider,
        process_memory_mb=round(process.memory_info().rss / (1024**2), 1),
        warnings=warnings,
    )


def _detect_windows_gpus() -> tuple[list[str], str | None, list[str]]:
    if platform.system() != "Windows":
        return [], None, ["GPU detection is only available on Windows."]

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController "
            "| Select-Object Name,DriverVersion "
            "| ConvertTo-Json -Depth 3"
        ),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        controllers = _parse_gpu_json(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return [], None, [f"GPU detection failed: {exc}"]

    names = [controller["Name"] for controller in controllers if controller.get("Name")]
    amd_driver_version = next(
        (
            controller.get("DriverVersion")
            for controller in controllers
            if _has_amd_radeon_gpu([str(controller.get("Name", ""))])
            and controller.get("DriverVersion")
        ),
        None,
    )
    return names, amd_driver_version, []


def _cached_windows_gpu_counters() -> tuple[float | None, float | None, str | None, list[str]]:
    global _GPU_COUNTER_CACHE
    now = time.monotonic()
    if _GPU_COUNTER_CACHE and now - _GPU_COUNTER_CACHE[0] < _GPU_COUNTER_CACHE_TTL_SECONDS:
        _, gpu_percent, dedicated_bytes, provider, warnings = _GPU_COUNTER_CACHE
        return gpu_percent, dedicated_bytes, provider, list(warnings)

    gpu_percent, dedicated_bytes, provider, warnings = _probe_windows_gpu_counters()
    _GPU_COUNTER_CACHE = (now, gpu_percent, dedicated_bytes, provider, list(warnings))
    return gpu_percent, dedicated_bytes, provider, warnings


def _probe_windows_gpu_counters() -> tuple[float | None, float | None, str | None, list[str]]:
    if platform.system() != "Windows":
        return None, None, None, ["GPU live counters are only available on Windows."]

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$ErrorActionPreference = 'Stop';"
            "try {"
            "  $util = @((Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples "
            "    | Where-Object { $_.CookedValue -gt 0 } "
            "    | Measure-Object -Property CookedValue -Sum).Sum;"
            "} catch { $util = $null };"
            "try {"
            "  $dedicated = @((Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples "
            "    | Measure-Object -Property CookedValue -Sum).Sum;"
            "} catch { $dedicated = $null };"
            "[pscustomobject]@{"
            "  GpuPercent = $util;"
            "  DedicatedBytes = $dedicated;"
            "  Provider = 'Windows performance counters'"
            "} | ConvertTo-Json -Compress"
        ),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=6,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return None, None, None, [f"GPU live counter read failed: {exc}"]

    gpu_percent = _float_or_none(payload.get("GpuPercent"))
    dedicated_bytes = _float_or_none(payload.get("DedicatedBytes"))
    provider = payload.get("Provider") if isinstance(payload.get("Provider"), str) else None
    warnings: list[str] = []
    if gpu_percent is None:
        warnings.append("GPU utilization counter is unavailable.")
    if dedicated_bytes is None:
        warnings.append("Dedicated GPU memory counter is unavailable.")
    if gpu_percent is not None:
        gpu_percent = min(max(gpu_percent, 0.0), 100.0)
    return gpu_percent, dedicated_bytes, provider, warnings


def _parse_gpu_json(raw_json: str) -> list[dict[str, str]]:
    if not raw_json.strip():
        return []

    parsed = json.loads(raw_json)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    controllers: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        controllers.append(
            {
                "Name": _string_value(item.get("Name")),
                "DriverVersion": _string_value(item.get("DriverVersion")),
            }
        )
    return controllers


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_amd_radeon_gpu(gpu_names: list[str]) -> bool:
    return any(
        "amd" in name.lower() or "radeon" in name.lower() for name in gpu_names
    )


def _cpu_name() -> str:
    return (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER", "")
        or platform.uname().processor
        or platform.machine()
        or "Unknown CPU"
    )
