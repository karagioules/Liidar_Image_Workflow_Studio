from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


DatasetType = Literal["body_part", "body_shape", "pose", "style", "fictional_face_identity"]
FacePolicy = Literal["reject_faces", "redact_faces", "body_part_crops_only"]
SourceRights = Literal["synthetic", "owned", "licensed", "consented"]


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
    resolution: int = Field(default=1024, gt=0)
    repeats: int = Field(default=10, gt=0)
    batch_size: int = Field(default=1, gt=0)
    max_train_steps: int = Field(default=1200, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    network_dim: int = Field(default=32, gt=0)
    network_alpha: int = Field(default=16, gt=0)


class TrainingJobConfig(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid4().hex)
    dataset_id: str
    dataset_path: str
    output_dir: str
    base_model_path: str
    lora_name: str
    resolution: int = 1024
    repeats: int = 10
    batch_size: int = 1
    max_train_steps: int = 1200
    learning_rate: float = 1e-4
    network_dim: int = 32
    network_alpha: int = 16
    accepted_image_count: int = Field(gt=0)
    completed_lora_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TrainerStatus(BaseModel):
    trainer_entrypoint: str
    config_path: str
    trainer_entrypoint_exists: bool
    config_path_exists: bool
    warnings: list[str] = Field(default_factory=list)
