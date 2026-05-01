from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_config import build_trainer_command
from local_model_studio.training_schemas import TrainingJobConfig, TrainingRunStatus
from local_model_studio.training_store import TrainingStore


class TrainingRunManager:
    def __init__(self, paths: WorkspacePaths, training: TrainingStore) -> None:
        self.paths = paths
        self.training = training
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, _TrainingRun] = {}
        self._lock = Lock()

    def start(self, job_id: str, trainer_entrypoint: str) -> TrainingRunStatus:
        job = self.training.get(job_id)
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
                self.process.terminate()
            self.status = "cancelled"
            self.updated_at = datetime.now(UTC)

    def snapshot(self) -> TrainingRunStatus:
        with self._lock:
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
                tail=_read_tail(self.log_path),
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
                exit_code = process.wait()
            with self._lock:
                self.exit_code = exit_code
                if self.status == "cancelled":
                    self.updated_at = datetime.now(UTC)
                    return
                if exit_code != 0:
                    self.status = "failed"
                    self.error = f"Trainer exited with code {exit_code}. See log: {self.log_path}"
                elif self.output_lora_path and self.output_lora_path.is_file():
                    self.training.register_completed_lora(self.job.job_id, self.output_lora_path)
                    self.status = "completed"
                else:
                    self.status = "failed"
                    self.error = f"Trainer finished but expected LoRA was not found: {self.output_lora_path}"
                self.updated_at = datetime.now(UTC)
        except Exception as exc:
            with self._lock:
                self.status = "failed"
                self.error = str(exc)
                self.updated_at = datetime.now(UTC)


def _resolve_workspace_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _expected_lora_path(root: Path, job: TrainingJobConfig) -> Path:
    output_dir = Path(job.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return output_dir / f"{job.lora_name}.safetensors"


def _read_tail(path: Path, line_count: int = 30) -> list[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    except OSError:
        return []
