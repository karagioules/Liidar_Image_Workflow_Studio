from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from local_model_studio.comfy_client import ComfyClient
from local_model_studio.dataset_scanner import scan_dataset
from local_model_studio.file_browser import IMAGE_SUFFIXES, browse_filesystem, render_thumbnail
from local_model_studio.local_vision_captioner import local_captioner_from_workspace
from local_model_studio.paths import WorkspacePaths, default_workspace_root
from local_model_studio.profile_store import ProfileStore
from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.reference_analyzer import analyze_references
from local_model_studio.runtime_check import build_runtime_status
from local_model_studio.schemas import (
    CharacterProfile,
    GenerationJobResponse,
    GenerationRequest,
    ImportReferenceImagesResponse,
    PathBrowserResponse,
    PromptRecipe,
    ReferenceAnalysisRequest,
    ReferenceAnalysisResponse,
    RuntimeStatus,
)
from local_model_studio.training_config import build_training_config, check_trainer_status
from local_model_studio.training_schemas import (
    DatasetScanReport,
    DatasetScanRequest,
    TrainerStatus,
    TrainingConfigRequest,
    TrainingJobConfig,
)
from local_model_studio.training_store import TrainingStore
from local_model_studio.workflow_templates import render_sdxl_workflow


CHECKPOINT_NAME = "sdxl_base_1.0.safetensors"


class TrainingConfigPayload(BaseModel):
    request: TrainingConfigRequest
    accepted_image_count: int = Field(gt=0)


class RegisterLoraPayload(BaseModel):
    lora_path: Path


def create_app(
    paths: WorkspacePaths | None = None,
    comfy_client: ComfyClient | None = None,
) -> FastAPI:
    workspace_paths = paths or WorkspacePaths(default_workspace_root())
    profiles = ProfileStore(workspace_paths)
    training = TrainingStore(workspace_paths)
    comfy = comfy_client or ComfyClient()

    app = FastAPI(title="Local Model Studio API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5274",
            "http://127.0.0.1:5274",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/runtime", response_model=RuntimeStatus)
    def runtime_status() -> RuntimeStatus:
        return build_runtime_status(workspace_paths)

    @app.get("/api/filesystem/browse", response_model=PathBrowserResponse)
    def browse_local_filesystem(path: str | None = None) -> PathBrowserResponse:
        try:
            return browse_filesystem(path, workspace_paths.root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/filesystem/thumbnail")
    def thumbnail(path: str) -> Response:
        try:
            return Response(content=render_thumbnail(path), media_type="image/jpeg")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/filesystem/import-references", response_model=ImportReferenceImagesResponse)
    async def import_reference_images(files: list[UploadFile] = File(...)) -> ImportReferenceImagesResponse:
        imported_paths: list[str] = []
        skipped_files: list[str] = []
        batch_dir = _reference_import_dir(workspace_paths.root)
        batch_dir.mkdir(parents=True, exist_ok=True)

        for upload in files:
            original_name = upload.filename or "reference"
            if Path(original_name).suffix.lower() not in IMAGE_SUFFIXES:
                skipped_files.append(original_name)
                continue

            destination = _unique_upload_path(batch_dir, original_name)
            try:
                content = await upload.read()
                destination.write_bytes(content)
                _validate_uploaded_image(destination)
            except ValueError:
                destination.unlink(missing_ok=True)
                skipped_files.append(original_name)
                continue
            imported_paths.append(str(destination))

        if not imported_paths and skipped_files:
            raise HTTPException(status_code=400, detail="No supported image files were selected.")

        return ImportReferenceImagesResponse(imported_paths=imported_paths, skipped_files=skipped_files)

    @app.get("/api/characters", response_model=list[CharacterProfile])
    def list_characters() -> list[CharacterProfile]:
        return profiles.list()

    @app.post("/api/characters", response_model=CharacterProfile)
    def save_character(profile: CharacterProfile) -> CharacterProfile:
        try:
            return profiles.save(profile)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/characters/analyze-references", response_model=ReferenceAnalysisResponse)
    def analyze_character_references(request: ReferenceAnalysisRequest, vision: bool = False) -> ReferenceAnalysisResponse:
        try:
            captioner = getattr(app.state, "captioner", None)
            if captioner is None and vision:
                captioner = local_captioner_from_workspace(workspace_paths.root)
                app.state.captioner = captioner
            captions = captioner([Path(path) for path in request.reference_images]) if captioner else None
            return analyze_references(request, captions=captions)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/characters/{profile_id}", response_model=CharacterProfile)
    def get_character(profile_id: str) -> CharacterProfile:
        return _get_profile_or_404(profiles, profile_id)

    @app.delete("/api/characters/{profile_id}", status_code=204)
    def delete_character(profile_id: str) -> Response:
        try:
            profiles.delete(profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Character profile not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.post("/api/generate/preview", response_model=PromptRecipe)
    def preview_generation(request: GenerationRequest) -> PromptRecipe:
        profile = _get_profile_or_404(profiles, request.character_id)
        return build_prompt_recipe(profile, request)

    @app.post("/api/generate", response_model=GenerationJobResponse)
    def generate(request: GenerationRequest) -> GenerationJobResponse:
        profile = _get_profile_or_404(profiles, request.character_id)
        recipe = build_prompt_recipe(profile, request)
        workflow = render_sdxl_workflow(recipe, CHECKPOINT_NAME)
        try:
            prompt_id = comfy.queue_prompt(workflow)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"ComfyUI request failed: {exc}",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return GenerationJobResponse(prompt_id=prompt_id, recipe=recipe)

    @app.post("/api/training/scan", response_model=DatasetScanReport)
    def scan_training_dataset(request: DatasetScanRequest) -> DatasetScanReport:
        try:
            return scan_dataset(request, output_root=workspace_paths.root / "datasets")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/training/config", response_model=TrainingJobConfig)
    def create_training_config(payload: TrainingConfigPayload) -> TrainingJobConfig:
        try:
            config = build_training_config(payload.request, payload.accepted_image_count)
            return training.save(config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/training/status", response_model=TrainerStatus)
    def training_status(
        trainer_entrypoint: str = Query(min_length=1),
        config_path: str | None = None,
    ) -> TrainerStatus:
        resolved_config_path = config_path or str(workspace_paths.training_dir)
        return check_trainer_status(trainer_entrypoint, resolved_config_path)

    @app.post("/api/training/{job_id}/register-lora", response_model=TrainingJobConfig)
    def register_lora(job_id: str, payload: RegisterLoraPayload) -> TrainingJobConfig:
        try:
            return training.register_completed_lora(job_id, payload.lora_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training job not found.") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _get_profile_or_404(store: ProfileStore, profile_id: str) -> CharacterProfile:
    try:
        return store.get(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Character profile not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _reference_import_dir(workspace_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return workspace_root / "inputs" / "reference_images" / f"{timestamp}-{uuid4().hex[:8]}"


def _unique_upload_path(root: Path, original_name: str) -> Path:
    safe_parts = [_safe_filename(part) for part in Path(original_name.replace("\\", "/")).parts]
    filename = safe_parts[-1] if safe_parts else f"reference-{uuid4().hex}.png"
    if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
        filename = f"{Path(filename).stem}.png"
    candidate = root / filename
    counter = 2
    while candidate.exists():
        candidate = root / f"{candidate.stem}-{counter}{candidate.suffix}"
        counter += 1
    return candidate


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._- " else "_" for char in value).strip(" .")
    return safe or f"reference-{uuid4().hex}"


def _validate_uploaded_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"Unsupported image file: {path.name}") from exc


app = create_app()
