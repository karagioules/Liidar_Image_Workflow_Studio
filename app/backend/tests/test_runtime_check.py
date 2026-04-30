from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio.runtime_check import build_runtime_status


def test_runtime_status_reports_missing_comfyui(tmp_path: Path) -> None:
    status = build_runtime_status(
        WorkspacePaths(tmp_path),
        gpu_names=["AMD Radeon RX 9070 XT"],
        driver_version="32.0",
    )

    assert status.comfyui_path_exists is False
    assert status.gpu_names == ["AMD Radeon RX 9070 XT"]
    assert any("ComfyUI" in warning for warning in status.warnings)


def test_runtime_status_warns_when_no_amd_gpu(tmp_path: Path) -> None:
    status = build_runtime_status(
        WorkspacePaths(tmp_path),
        gpu_names=["Microsoft Basic Display"],
        driver_version=None,
    )

    assert any("AMD Radeon" in warning for warning in status.warnings)
