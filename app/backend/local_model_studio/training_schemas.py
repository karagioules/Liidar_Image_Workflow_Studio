from __future__ import annotations

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
