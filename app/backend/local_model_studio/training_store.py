from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from local_model_studio.file_browser import IMAGE_SUFFIXES
from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_schemas import DatasetType, TrainingJobConfig

SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
LORA_ARTIFACT_SUFFIXES = {".safetensors", ".pt", ".pth"}


class TrainingStore:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self.paths.training_dir.mkdir(parents=True, exist_ok=True)

    def save(self, config: TrainingJobConfig) -> TrainingJobConfig:
        config = TrainingJobConfig.model_validate(config.model_dump())
        now = datetime.now(UTC)
        existing_created_at = config.created_at
        path = self._path_for(config.job_id)
        if path.exists():
            existing_created_at = TrainingJobConfig.model_validate_json(
                path.read_text(encoding="utf-8")
            ).created_at
        prepared_dataset_path = self._prepare_captioned_dataset(config)
        saved = config.model_copy(
            update={
                "dataset_path": str(prepared_dataset_path),
                "config_path": str(path),
                "created_at": existing_created_at,
                "updated_at": now,
            }
        )
        path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
        return saved

    def list(self) -> list[TrainingJobConfig]:
        jobs = [self._load_path(path) for path in self.paths.training_dir.glob("*.json")]
        return sorted(jobs, key=lambda job: job.created_at)

    def clear_pending_global_jobs(self) -> int:
        removed_count = 0
        for path in self.paths.training_dir.glob("*.json"):
            job = self._load_path(path)
            if job.global_pack and not job.completed_lora_path:
                path.unlink()
                removed_count += 1
        return removed_count

    def get(self, job_id: str) -> TrainingJobConfig:
        path = self._path_for(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return self._load_path(path)

    def prepare_job_dataset(self, job_id: str) -> TrainingJobConfig:
        config = self.get(job_id)
        prepared_dataset_path = self._prepare_captioned_dataset(config)
        if str(prepared_dataset_path) == config.dataset_path:
            return config
        return self.save(config.model_copy(update={"dataset_path": str(prepared_dataset_path)}))

    def register_completed_lora(self, job_id: str, lora_path: str | Path) -> TrainingJobConfig:
        config = self.get(job_id)
        resolved_lora_path = Path(lora_path)
        if resolved_lora_path.suffix.lower() not in LORA_ARTIFACT_SUFFIXES:
            allowed_suffixes = ", ".join(sorted(LORA_ARTIFACT_SUFFIXES))
            raise ValueError(f"LoRA artifact file must end with one of: {allowed_suffixes}.")
        if not resolved_lora_path.is_file():
            raise FileNotFoundError(f"LoRA artifact file does not exist: {resolved_lora_path}")

        saved = config.model_copy(
            update={
                "completed_lora_path": str(resolved_lora_path),
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
        return TrainingJobConfig.model_validate({**data, "config_path": str(path)})

    def _prepare_captioned_dataset(self, config: TrainingJobConfig) -> Path:
        source = Path(config.dataset_path)
        target = self.paths.root / "config" / "training_datasets" / config.job_id
        if source.resolve() == target.resolve():
            _normalize_prepared_dataset(target, config)
            _ensure_captions(target, config.dataset_type)
            return target

        if not source.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {source}")

        if target.exists():
            shutil.rmtree(target)
        image_dir = target / f"{config.repeats}_{_slug(config.lora_name)}"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        for index, image_path in enumerate(image_paths, start=1):
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._") or "image"
            target_image = image_dir / f"{index:05d}_{safe_stem}{image_path.suffix.lower()}"
            shutil.copy2(image_path, target_image)
            target_image.with_suffix(".caption").write_text(_caption_for(config.dataset_type), encoding="utf-8")

        return target


def _ensure_captions(dataset_path: Path, dataset_type: DatasetType | None) -> None:
    for image_path in dataset_path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            caption_path = image_path.with_suffix(".caption")
            if not caption_path.exists():
                caption_path.write_text(_caption_for(dataset_type), encoding="utf-8")


def _normalize_prepared_dataset(dataset_path: Path, config: TrainingJobConfig) -> None:
    expected_dir = dataset_path / f"{config.repeats}_{_slug(config.lora_name)}"
    image_paths = sorted(path for path in dataset_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths or all(expected_dir in path.parents for path in image_paths):
        return

    tmp_path = dataset_path.with_name(f"{dataset_path.name}_tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_image_dir = tmp_path / expected_dir.name
    tmp_image_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._") or "image"
        target_image = tmp_image_dir / f"{index:05d}_{safe_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target_image)
        target_image.with_suffix(".caption").write_text(_caption_for(config.dataset_type), encoding="utf-8")

    shutil.rmtree(dataset_path)
    tmp_path.replace(dataset_path)


def _caption_for(dataset_type: DatasetType | None) -> str:
    captions: dict[DatasetType | None, str] = {
        "body_part": "adult body detail, realistic skin texture, natural anatomy",
        "body_shape": "adult body proportions, realistic anatomy, natural body shape",
        "pose": "adult pose reference, realistic body positioning, natural anatomy",
        "style": "realistic photo style, natural lighting, believable texture",
        "fictional_face_identity": "fictional adult face identity reference, consistent facial features",
        None: "adult body detail, realistic anatomy",
    }
    return captions.get(dataset_type, captions[None])


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "global_body_pack"
