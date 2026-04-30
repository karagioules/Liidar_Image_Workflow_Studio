from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


AdultAgeCategory = Literal["adult_18_plus", "adult_21_plus", "adult_25_plus", "adult_30_plus"]
QualityPreset = Literal["fast", "balanced", "high", "ultra"]
GenerationMode = Literal["portrait", "full_body", "lifestyle_post", "studio", "reference_match"]
SeedStrategy = Literal["locked", "vary", "reuse_last"]


class CharacterProfile(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    display_name: str = Field(min_length=1, max_length=80)
    age_category: AdultAgeCategory = "adult_25_plus"
    face_summary: str = Field(default="", max_length=600)
    hair: str = Field(default="", max_length=240)
    eyes: str = Field(default="", max_length=160)
    skin_tone: str = Field(default="", max_length=160)
    body_shape: str = Field(default="", max_length=400)
    chest: str = Field(default="", max_length=240)
    grooming: str = Field(default="", max_length=240)
    style_notes: str = Field(default="", max_length=600)
    negative_notes: str = Field(default="", max_length=600)
    reference_images: list[str] = Field(default_factory=list)
    lora_files: list[str] = Field(default_factory=list)
    seed_strategy: SeedStrategy = "vary"
    locked_seed: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("age_category")
    @classmethod
    def require_adult_age_category(cls, value: str) -> str:
        if not value.startswith("adult_"):
            raise ValueError("Character profiles must use an adult age category.")
        return value


class ReferenceAnalysisRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    display_name: str = Field(default="", max_length=80)
    age_category: AdultAgeCategory = "adult_25_plus"
    reference_images: list[str] = Field(min_length=1)
    lora_files: list[str] = Field(default_factory=list)
    seed_strategy: SeedStrategy = "vary"
    locked_seed: int | None = None


class BodyAttributeCues(BaseModel):
    coverage: str = Field(max_length=80)
    chest_visibility: str = Field(max_length=120)
    pose_framing: str = Field(max_length=120)
    confidence: int = Field(ge=0, le=100)
    evidence: str = Field(max_length=240)


class ReferenceAnalysisImage(BaseModel):
    path: str
    file_name: str
    width: int
    height: int
    orientation: str
    brightness: str
    tone: str
    texture: str
    caption: str | None = None
    body_attributes: BodyAttributeCues


class ReferenceAnalysisResponse(CharacterProfile):
    consistency_score: int = Field(ge=0, le=100)
    analysis_warnings: list[str] = Field(default_factory=list)
    analysis_images: list[ReferenceAnalysisImage] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    character_id: str
    mode: GenerationMode = "portrait"
    quality: QualityPreset = "balanced"
    scene_prompt: str = Field(default="", max_length=1000)
    extra_negative: str = Field(default="", max_length=1000)
    seed: int | None = None


class PromptRecipe(BaseModel):
    positive: str
    negative: str
    seed: int
    width: int
    height: int
    steps: int
    cfg: float


class RuntimeStatus(BaseModel):
    os_name: str
    python_version: str
    cpu_name: str
    total_ram_gb: float
    gpu_names: list[str]
    amd_driver_version: str | None
    comfyui_path_exists: bool
    warnings: list[str] = Field(default_factory=list)


PathEntryKind = Literal["directory", "file", "drive"]


class PathBrowserEntry(BaseModel):
    name: str
    path: str
    kind: PathEntryKind


class PathBrowserResponse(BaseModel):
    current_path: str | None
    parent_path: str | None
    entries: list[PathBrowserEntry]


class SelectedReferenceImagesResponse(BaseModel):
    selected_paths: list[str]


class GenerationJobResponse(BaseModel):
    prompt_id: str
    recipe: PromptRecipe


class OutputMetadata(BaseModel):
    file_name: str
    character_id: str
    character_display_name: str
    request: GenerationRequest
    recipe: PromptRecipe
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
