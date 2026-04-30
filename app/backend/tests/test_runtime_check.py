from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio import runtime_check
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


def test_runtime_status_treats_radeon_gpu_name_as_amd_related(tmp_path: Path) -> None:
    status = build_runtime_status(
        WorkspacePaths(tmp_path),
        gpu_names=["Radeon RX 9070 XT"],
        driver_version="32.0",
    )

    assert not any("AMD Radeon" in warning for warning in status.warnings)


def test_parse_gpu_json_accepts_single_controller_object() -> None:
    controllers = runtime_check._parse_gpu_json(
        '{"Name":"AMD Radeon RX 9070 XT","DriverVersion":"32.0"}'
    )

    assert controllers == [{"Name": "AMD Radeon RX 9070 XT", "DriverVersion": "32.0"}]


def test_detect_windows_gpus_reports_failed_subprocess(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise FileNotFoundError("powershell missing")

    monkeypatch.setattr(runtime_check.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_check.subprocess, "run", fail_run)

    names, driver_version, warnings = runtime_check._detect_windows_gpus()

    assert names == []
    assert driver_version is None
    assert warnings == ["GPU detection failed: powershell missing"]
