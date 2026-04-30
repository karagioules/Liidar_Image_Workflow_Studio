from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_schemas import TrainingJobConfig

SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class TrainingStore:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self.paths.training_dir.mkdir(parents=True, exist_ok=True)

    def save(self, config: TrainingJobConfig) -> TrainingJobConfig:
        saved = TrainingJobConfig.model_validate(config.model_dump())
        self._path_for(saved.job_id).write_text(saved.model_dump_json(indent=2), encoding="utf-8")
        return saved

    def list(self) -> list[TrainingJobConfig]:
        jobs = [self._load_path(path) for path in self.paths.training_dir.glob("*.json")]
        return sorted(jobs, key=lambda job: job.created_at)

    def get(self, job_id: str) -> TrainingJobConfig:
        path = self._path_for(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return self._load_path(path)

    def register_completed_lora(self, job_id: str, lora_path: str | Path) -> TrainingJobConfig:
        config = self.get(job_id)
        saved = config.model_copy(
            update={
                "completed_lora_path": str(lora_path),
                "updated_at": datetime.now(UTC),
            }
        )
        self._path_for(job_id).write_text(saved.model_dump_json(indent=2), encoding="utf-8")
        return saved

    def _path_for(self, job_id: str) -> Path:
        if not SAFE_JOB_ID_RE.fullmatch(job_id):
            raise ValueError(
                "Training job id must be non-empty and contain only letters, numbers, "
                "hyphen, or underscore."
            )
        return self.paths.training_dir / f"{job_id}.json"

    def _load_path(self, path: Path) -> TrainingJobConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TrainingJobConfig.model_validate(data)
