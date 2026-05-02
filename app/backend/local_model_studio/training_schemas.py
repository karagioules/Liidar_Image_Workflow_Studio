from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from local_model_studio.schemas import DatasetPrepResponse, DatasetPrepScanMode, DatasetPrepTarget


DatasetType = Literal["body_part", "body_shape", "pose", "style", "fictional_face_identity"]
FacePolicy = Literal["reject_faces", "redact_faces", "body_part_crops_only"]
SourceRights = Literal["synthetic", "owned", "licensed", "consented"]
TrainingRunState = Literal["queued", "running", "cancelled", "completed", "failed"]


class DatasetScanRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_folder: Path
    dataset_type: DatasetType
    face_policy: FacePolicy = "reject_faces"
    character_id: str | None = None
    source_rights: SourceRights | None = None
    tags: list[str] = Field(default_factory=list)


class FaceScanResult(BaseModel):
    face_count: int = Field(ge=0)
    detector_available: bool
    warning: str | None = None


class ScannedImage(BaseModel):
    source_path: str
    stored_path: str | None = None
    sha256: str
    accepted: bool
    reason: str
    caption: str
    character_id: str | None = None
    source_rights: SourceRights | None = None


class DatasetScanReport(BaseModel):
    dataset_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    dataset_type: DatasetType
    accepted_count: int
    rejected_count: int
    duplicate_count: int
    ignored_count: int
    accepted: list[ScannedImage] = Field(default_factory=list)
    rejected: list[ScannedImage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrainingConfigRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    base_model_path: str
    lora_name: str = Field(min_length=1, max_length=120)
    dataset_type: DatasetType | None = None
    global_pack: bool = True
    resolution: int = Field(default=768, gt=0)
    repeats: int = Field(default=6, gt=0)
    batch_size: int = Field(default=1, gt=0)
    max_train_steps: int = Field(default=900, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    network_dim: int = Field(default=32, gt=0)
    network_alpha: int = Field(default=16, gt=0)


class TrainingJobConfig(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid4().hex)
    dataset_id: str
    dataset_path: str
    output_dir: str
    config_path: str | None = None
    base_model_path: str
    lora_name: str
    dataset_type: DatasetType | None = None
    global_pack: bool = True
    resolution: int = 768
    repeats: int = 6
    batch_size: int = 1
    max_train_steps: int = 900
    learning_rate: float = 1e-4
    network_dim: int = 32
    network_alpha: int = 16
    accepted_image_count: int = Field(gt=0)
    completed_lora_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ClearTrainingJobsResponse(BaseModel):
    removed_count: int = Field(ge=0)
    remaining_jobs: list[TrainingJobConfig] = Field(default_factory=list)


class TrainingRunStartRequest(BaseModel):
    trainer_entrypoint: str = Field(default="tools/train_global_lora.ps1", min_length=1)


class TrainingRunStatus(BaseModel):
    run_id: str
    job_id: str
    status: TrainingRunState
    trainer_entrypoint: str
    config_path: str
    output_lora_path: str | None = None
    process_id: int | None = None
    exit_code: int | None = None
    log_path: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: float | None = None
    tail: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TrainerStatus(BaseModel):
    trainer_entrypoint: str
    config_path: str
    trainer_entrypoint_exists: bool
    config_path_exists: bool
    warnings: list[str] = Field(default_factory=list)


class GlobalLearningRequest(BaseModel):
    pack_name: str = Field(default="global-body-pack", min_length=1, max_length=120)
    source_folder: Path
    dataset_type: DatasetType = "body_shape"
    target: DatasetPrepTarget = "chest_detail"
    recursive: bool = True
    scan_mode: DatasetPrepScanMode = "local"
    preset: Literal["fast", "balanced", "high_quality"] = "balanced"
    base_model_path: str = Field(default="models/sdxl_base_1.0.safetensors", min_length=1)
    output_dir: str = Field(default="outputs/global_lora", min_length=1)


class GlobalLearningResponse(BaseModel):
    prep: DatasetPrepResponse
    training_job: TrainingJobConfig
    version: int = Field(gt=0)
    warnings: list[str] = Field(default_factory=list)
