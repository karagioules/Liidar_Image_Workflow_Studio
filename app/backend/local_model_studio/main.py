from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from local_model_studio.comfy_client import ComfyClient
from local_model_studio.dataset_scanner import scan_dataset
from local_model_studio.paths import WorkspacePaths, default_workspace_root
from local_model_studio.profile_store import ProfileStore
from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.runtime_check import build_runtime_status
from local_model_studio.schemas import (
    CharacterProfile,
    GenerationJobResponse,
    GenerationRequest,
    PromptRecipe,
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

    @app.get("/api/characters", response_model=list[CharacterProfile])
    def list_characters() -> list[CharacterProfile]:
        return profiles.list()

    @app.post("/api/characters", response_model=CharacterProfile)
    def save_character(profile: CharacterProfile) -> CharacterProfile:
        try:
            return profiles.save(profile)
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


app = create_app()
