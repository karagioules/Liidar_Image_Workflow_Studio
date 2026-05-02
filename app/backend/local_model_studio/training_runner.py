from __future__ import annotations

import subprocess
import sys
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_config import build_trainer_command
from local_model_studio.training_schemas import TrainingJobConfig, TrainingRunStatus
from local_model_studio.training_store import TrainingStore

PROGRESS_RE = re.compile(r"(?P<percent>\d{1,3})%\|.*?\|\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)")


class TrainingRunManager:
    def __init__(self, paths: WorkspacePaths, training: TrainingStore) -> None:
        self.paths = paths
        self.training = training
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, _TrainingRun] = {}
        self._lock = Lock()

    def start(self, job_id: str, trainer_entrypoint: str) -> TrainingRunStatus:
        job = self.training.prepare_job_dataset(job_id)
        build_trainer_command(_resolve_workspace_path(self.paths.root, trainer_entrypoint), job.config_path or "")
        run = _TrainingRun(self.paths, self.training, job, trainer_entrypoint)
        with self._lock:
            self._runs[run.run_id] = run
        status = run.snapshot()
        run.start()
        return status

    def list(self) -> list[TrainingRunStatus]:
        with self._lock:
            return [run.snapshot() for run in self._runs.values()]

    def get(self, run_id: str) -> TrainingRunStatus:
        return self._run(run_id).snapshot()

    def cancel(self, run_id: str) -> TrainingRunStatus:
        run = self._run(run_id)
        run.cancel()
        return run.snapshot()

    def _run(self, run_id: str) -> "_TrainingRun":
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise KeyError("Training run not found.") from exc


class _TrainingRun:
    def __init__(self, paths: WorkspacePaths, training: TrainingStore, job: TrainingJobConfig, trainer_entrypoint: str) -> None:
        self.paths = paths
        self.training = training
        self.job = job
        self.trainer_entrypoint = trainer_entrypoint
        self.run_id = uuid4().hex
        self.status = "queued"
        self.process: subprocess.Popen | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self.started_at = datetime.now(UTC)
        self.updated_at = self.started_at
        self.log_path = self.paths.logs_dir / f"training-{self.run_id}.log"
        self.output_lora_path: Path | None = _expected_lora_path(paths.root, job)
        self._lock = Lock()

    def start(self) -> None:
        thread = Thread(target=self._run, name=f"training-run-{self.run_id[:8]}", daemon=True)
        thread.start()

    def cancel(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                _terminate_process_tree(self.process)
            self.status = "cancelled"
            self.updated_at = datetime.now(UTC)

    def snapshot(self) -> TrainingRunStatus:
        with self._lock:
            tail = _read_tail(self.log_path)
            progress = _parse_training_progress(tail)
            return TrainingRunStatus(
                run_id=self.run_id,
                job_id=self.job.job_id,
                status=self.status,
                trainer_entrypoint=self.trainer_entrypoint,
                config_path=self.job.config_path or "",
                output_lora_path=str(self.output_lora_path) if self.output_lora_path else None,
                process_id=self.process.pid if self.process else None,
                exit_code=self.exit_code,
                log_path=str(self.log_path),
                progress_current=progress[0],
                progress_total=progress[1],
                progress_percent=progress[2],
                tail=tail,
                error=self.error,
                started_at=self.started_at,
                updated_at=self.updated_at,
            )

    def _run(self) -> None:
        with self._lock:
            self.status = "running"
            self.updated_at = datetime.now(UTC)
        try:
            command = build_trainer_command(_resolve_workspace_path(self.paths.root, self.trainer_entrypoint), self.job.config_path or "")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write("Starting training command:\n")
                log.write(" ".join(command) + "\n\n")
                stopped_comfy_processes = _stop_workspace_comfyui(self.paths.root)
                if stopped_comfy_processes:
                    log.write(
                        f"Paused {stopped_comfy_processes} local ComfyUI process(es) so training can use GPU VRAM.\n\n"
                    )
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(self.paths.root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self._lock:
                    self.process = process
                    self.updated_at = datetime.now(UTC)
                exit_code = self._wait_for_process(process)
                if stopped_comfy_processes:
                    _start_workspace_comfyui(self.paths.root, self.paths.logs_dir)
            with self._lock:
                self.exit_code = exit_code
                if self.status == "cancelled":
                    self.updated_at = datetime.now(UTC)
                    return
                completed_lora = _find_completed_lora_artifact(self.paths.root, self.job, self.output_lora_path)
                if completed_lora:
                    self.training.register_completed_lora(self.job.job_id, completed_lora)
                    self.output_lora_path = completed_lora
                    self.status = "completed"
                elif exit_code == _NAN_LOSS_EXIT_CODE:
                    self.status = "failed"
                    self.error = (
                        "Training stopped because the loss became NaN. "
                        "The run was unstable, so no pack was saved. "
                        "Use the updated safer BF16 trainer settings and start again."
                    )
                elif exit_code != 0:
                    self.status = "failed"
                    self.error = f"Trainer exited with code {exit_code}. See log: {self.log_path}"
                else:
                    self.status = "failed"
                    self.error = f"Trainer finished but expected LoRA was not found: {self.output_lora_path}"
                self.updated_at = datetime.now(UTC)
        except Exception as exc:
            with self._lock:
                self.status = "failed"
                self.error = str(exc)
                self.updated_at = datetime.now(UTC)

    def _wait_for_process(self, process: subprocess.Popen) -> int:
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                return int(exit_code)
            if _tail_has_nan_loss(_read_tail(self.log_path, line_count=80)):
                _terminate_process_tree(process)
                return _NAN_LOSS_EXIT_CODE
            time.sleep(0.5)


def _resolve_workspace_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _expected_lora_path(root: Path, job: TrainingJobConfig) -> Path:
    output_dir = Path(job.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return output_dir / f"{job.lora_name}.safetensors"


_NAN_LOSS_EXIT_CODE = -9501


def _find_completed_lora_artifact(root: Path, job: TrainingJobConfig, expected_path: Path | None) -> Path | None:
    if expected_path and expected_path.is_file():
        return expected_path

    output_dir = Path(job.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not output_dir.is_dir():
        return None

    candidates = sorted(
        output_dir.glob(f"{job.lora_name}*.safetensors"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_tail(path: Path, line_count: int = 2000) -> list[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    except OSError:
        return []


def _parse_training_progress(lines: list[str]) -> tuple[int | None, int | None, float | None]:
    for line in reversed(lines):
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        current = int(match.group("current"))
        total = int(match.group("total"))
        if total <= 0:
            return current, total, None
        percent = min(100.0, max(0.0, (current / total) * 100))
        return current, total, round(percent, 1)
    return None, None, None


def _tail_has_nan_loss(lines: list[str]) -> bool:
    return any("avr_loss=nan" in line.lower() or "loss=nan" in line.lower() for line in lines)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    process.terminate()


def _stop_workspace_comfyui(root: Path) -> int:
    if sys.platform != "win32":
        return 0
    comfy_path = str((root / "ComfyUI").resolve())
    script = f"""
$comfyPath = '{comfy_path.replace("'", "''")}'
$portOwners = Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique
$processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
  Where-Object {{
    ($portOwners -contains $_.ProcessId) -or
    ($_.CommandLine -and $_.CommandLine.Contains($comfyPath) -and $_.CommandLine.Contains('main.py') -and $_.CommandLine.Contains('--port 8188'))
  }}
$count = 0
foreach ($process in $processes) {{
  Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  $count++
}}
Write-Output $count
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int((result.stdout or "0").strip().splitlines()[-1])
    except (IndexError, ValueError):
        return 0


def _start_workspace_comfyui(root: Path, logs_dir: Path) -> None:
    if sys.platform != "win32":
        return
    comfy_path = root / "ComfyUI"
    comfy_python = comfy_path / ".venv" / "Scripts" / "python.exe"
    if not comfy_python.is_file():
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "comfyui-after-training.log"
    comfy_dir = str(comfy_path).replace("'", "''")
    python_path = str(comfy_python).replace("'", "''")
    output_log = str(log_path).replace("'", "''")
    command = (
        f"Set-Location -LiteralPath '{comfy_dir}'; "
        f"& '{python_path}' main.py --listen 127.0.0.1 --port 8188 "
        f"*> '{output_log}'"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=str(root),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
