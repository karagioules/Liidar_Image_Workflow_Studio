from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any

import psutil

from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import RuntimeStatus


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
