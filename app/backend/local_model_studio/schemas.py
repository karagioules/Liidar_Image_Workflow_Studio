from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


AdultAgeCategory = Literal["adult_18_plus", "adult_21_plus", "adult_25_plus", "adult_30_plus"]
QualityPreset = Literal["fast", "balanced", "high", "ultra"]
GenerationMode = Literal["portrait", "full_body", "lifestyle_post", "studio", "reference_match"]
SeedStrategy = Literal["locked", "vary", "reuse_last"]
VisibilityLevel = Literal["clear", "partial", "covered", "not_visible", "unclear"]
DatasetPrepTarget = Literal["chest_detail", "upper_torso", "full_body_context"]
DatasetPrepScanMode = Literal["local", "claude", "off"]
DatasetPrepJobState = Literal["queued", "running", "cancelling", "cancelled", "completed", "failed"]


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


class AttributeDetail(BaseModel):
    visibility: VisibilityLevel
    confidence: int = Field(ge=0, le=100)
    summary: str = Field(max_length=260)
    evidence: str = Field(max_length=260)


class ReferenceImageDetails(BaseModel):
    face: AttributeDetail
    hair: AttributeDetail
    skin: AttributeDetail
    body_shape: AttributeDetail
    chest: AttributeDetail
    waist_hips: AttributeDetail
    pose: AttributeDetail
    clothing: AttributeDetail
    lighting: AttributeDetail
    camera: AttributeDetail
    background: AttributeDetail
    quality: AttributeDetail


class AdultContentSignals(BaseModel):
    nudity_level: str = Field(max_length=120)
    breast_visibility: str = Field(max_length=120)
    nipple_areola_visibility: str = Field(max_length=120)
    genital_visibility: str = Field(max_length=120)
    buttocks_visibility: str = Field(max_length=120)
    sexual_activity: str = Field(max_length=160)
    confidence: int = Field(ge=0, le=100)
    evidence: str = Field(max_length=320)


class VisionTag(BaseModel):
    name: str = Field(max_length=80)
    confidence: int = Field(ge=0, le=100)
    source: str = Field(max_length=120)


class AggregateReferenceIntelligence(ReferenceImageDetails):
    adult_content: AdultContentSignals
    strong_tags: list[VisionTag] = Field(default_factory=list)
    prompt_summary: str = Field(max_length=900)
    uncertainty_notes: list[str] = Field(default_factory=list)


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
    image_details: ReferenceImageDetails
    adult_content: AdultContentSignals
    vision_tags: list[VisionTag] = Field(default_factory=list)


class ReferenceAnalysisResponse(CharacterProfile):
    consistency_score: int = Field(ge=0, le=100)
    analysis_warnings: list[str] = Field(default_factory=list)
    analysis_images: list[ReferenceAnalysisImage] = Field(default_factory=list)
    aggregate_intelligence: AggregateReferenceIntelligence


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
    lora_files: list[str] = Field(default_factory=list)


class RuntimeStatus(BaseModel):
    os_name: str
    python_version: str
    cpu_name: str
    total_ram_gb: float
    gpu_names: list[str]
    amd_driver_version: str | None
    comfyui_path_exists: bool
    warnings: list[str] = Field(default_factory=list)


class SystemLiveMetrics(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_percent: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_total_gb: float | None = None
    gpu_provider: str | None = None
    process_memory_mb: float
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


class SelectedFolderResponse(BaseModel):
    folder_path: str | None


class DatasetPrepRequest(BaseModel):
    source_folder: Path
    output_folder: Path | None = None
    target: DatasetPrepTarget = "chest_detail"
    recursive: bool = True
    scan_mode: DatasetPrepScanMode | None = None
    use_ai: bool = False
    ai_max_images: int = Field(default=100, ge=0, le=500)
    ai_model: str = Field(default="claude-haiku-4-5-20251001", max_length=120)

    def effective_scan_mode(self) -> DatasetPrepScanMode:
        if self.scan_mode is not None:
            return self.scan_mode
        if self.use_ai:
            return "claude"
        return "local"


class DatasetPrepImage(BaseModel):
    source_path: str
    output_path: str | None = None
    width: int
    height: int
    face_count: int = Field(ge=0)
    accepted: bool
    reason: str
    method: str
    crop_box: list[int] | None = None


class DatasetPrepResponse(BaseModel):
    output_folder: str
    processed_count: int
    cropped_count: int
    skipped_count: int
    ai_attempted_count: int = 0
    ai_guided_count: int = 0
    ai_failed_count: int = 0
    face_guided_count: int
    fallback_count: int
    warnings: list[str] = Field(default_factory=list)
    images: list[DatasetPrepImage] = Field(default_factory=list)


class DatasetPrepJobStatus(BaseModel):
    job_id: str
    status: DatasetPrepJobState
    total_count: int = 0
    processed_count: int = 0
    cropped_count: int = 0
    skipped_count: int = 0
    ai_attempted_count: int = 0
    ai_guided_count: int = 0
    ai_failed_count: int = 0
    face_guided_count: int = 0
    fallback_count: int = 0
    active_file: str | None = None
    output_folder: str | None = None
    scan_mode: DatasetPrepScanMode = "local"
    use_ai: bool = False
    ai_max_images: int = 0
    cancel_requested: bool = False
    error: str | None = None
    result: DatasetPrepResponse | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApiKeyStatus(BaseModel):
    provider: str
    saved: bool
    model: str


class ApiKeyTestResponse(BaseModel):
    provider: str
    ok: bool
    status_code: int | None = None
    model: str
    message: str


class ApiKeyPayload(BaseModel):
    api_key: str = Field(min_length=1, max_length=400)
    model: str = Field(default="claude-haiku-4-5-20251001", max_length=120)


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
