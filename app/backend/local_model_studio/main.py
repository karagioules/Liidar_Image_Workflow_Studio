from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from local_model_studio.api_key_store import ApiKeyStore
from local_model_studio.comfy_client import ComfyClient
from local_model_studio.dataset_scanner import scan_dataset
from local_model_studio.dataset_prep_jobs import DatasetPrepJobManager
from local_model_studio.dataset_prepper import prepare_dataset_crops
from local_model_studio.file_browser import IMAGE_SUFFIXES, browse_filesystem, render_thumbnail
from local_model_studio.local_image_tagger import local_tagger_from_workspace
from local_model_studio.local_vision_captioner import local_captioner_from_workspace
from local_model_studio.paths import WorkspacePaths, default_workspace_root
from local_model_studio.profile_store import ProfileStore
from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.reference_analyzer import analyze_references
from local_model_studio.runtime_check import build_live_system_metrics, build_runtime_status
from local_model_studio.schemas import (
    CharacterProfile,
    ApiKeyPayload,
    ApiKeyStatus,
    DatasetPrepJobStatus,
    DatasetPrepRequest,
    DatasetPrepResponse,
    GenerationJobResponse,
    GenerationRequest,
    PathBrowserResponse,
    PromptRecipe,
    ReferenceAnalysisRequest,
    ReferenceAnalysisResponse,
    RuntimeStatus,
    SelectedFolderResponse,
    SelectedReferenceImagesResponse,
    SystemLiveMetrics,
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
    api_keys = ApiKeyStore(workspace_paths.root)
    prep_jobs = DatasetPrepJobManager()
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

    @app.get("/api/system/live", response_model=SystemLiveMetrics)
    def live_system_metrics() -> SystemLiveMetrics:
        return build_live_system_metrics()

    @app.get("/api/settings/anthropic-key", response_model=ApiKeyStatus)
    def anthropic_key_status() -> ApiKeyStatus:
        return api_keys.status()

    @app.post("/api/settings/anthropic-key", response_model=ApiKeyStatus)
    def save_anthropic_key(payload: ApiKeyPayload) -> ApiKeyStatus:
        return api_keys.save(payload)

    @app.delete("/api/settings/anthropic-key", response_model=ApiKeyStatus)
    def delete_anthropic_key() -> ApiKeyStatus:
        return api_keys.delete()

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

    @app.post("/api/filesystem/select-references", response_model=SelectedReferenceImagesResponse)
    def select_reference_images(mode: str = Query(pattern="^(files|folder)$")) -> SelectedReferenceImagesResponse:
        try:
            return SelectedReferenceImagesResponse(selected_paths=_select_reference_paths(mode))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/filesystem/select-folder", response_model=SelectedFolderResponse)
    def select_local_folder() -> SelectedFolderResponse:
        try:
            return SelectedFolderResponse(folder_path=_select_folder_path())
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
            tagger = getattr(app.state, "tagger", None)
            if tagger is None and vision:
                tagger = local_tagger_from_workspace(workspace_paths.root)
                app.state.tagger = tagger
            captions = captioner([Path(path) for path in request.reference_images]) if captioner else None
            tag_reports = tagger([Path(path) for path in request.reference_images]) if tagger else None
            return analyze_references(request, captions=captions, tag_reports=tag_reports)
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
        return build_prompt_recipe(profile, request, _active_global_lora_files(training))

    @app.post("/api/generate", response_model=GenerationJobResponse)
    def generate(request: GenerationRequest) -> GenerationJobResponse:
        profile = _get_profile_or_404(profiles, request.character_id)
        recipe = build_prompt_recipe(profile, request, _active_global_lora_files(training))
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

    @app.post("/api/dataset-prep/crop", response_model=DatasetPrepResponse)
    def prep_dataset_crops(request: DatasetPrepRequest) -> DatasetPrepResponse:
        try:
            return prepare_dataset_crops(request, anthropic_api_key=api_keys.api_key() if request.use_ai else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dataset-prep/jobs", response_model=DatasetPrepJobStatus)
    def start_dataset_prep_job(request: DatasetPrepRequest) -> DatasetPrepJobStatus:
        try:
            return prep_jobs.start(request, anthropic_api_key=api_keys.api_key() if request.use_ai else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dataset-prep/jobs/{job_id}", response_model=DatasetPrepJobStatus)
    def get_dataset_prep_job(job_id: str) -> DatasetPrepJobStatus:
        try:
            return prep_jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dataset-prep/jobs/{job_id}/cancel", response_model=DatasetPrepJobStatus)
    def cancel_dataset_prep_job(job_id: str) -> DatasetPrepJobStatus:
        try:
            return prep_jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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


def _active_global_lora_files(training: TrainingStore) -> list[str]:
    return [
        job.completed_lora_path
        for job in training.list()
        if job.global_pack and job.completed_lora_path
    ]


def _select_reference_paths(mode: str) -> list[str]:
    try:
        from tkinter import Tk, filedialog
    except Exception as exc:
        raise RuntimeError("Windows file picker is not available in this Python environment.") from exc

    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
    except Exception as exc:
        raise RuntimeError(f"Unable to open Windows file picker: {exc}") from exc

    try:
        if mode == "files":
            selected = filedialog.askopenfilenames(
                title="Select reference images",
                filetypes=[
                    ("Image files", "*.avif *.bmp *.jpeg *.jpg *.png *.webp"),
                    ("All files", "*.*"),
                ],
            )
            return _supported_image_paths(Path(path) for path in selected)

        folder = filedialog.askdirectory(title="Select reference image folder")
        if not folder:
            return []
        return _supported_image_paths(path for path in Path(folder).rglob("*") if path.is_file())
    except Exception as exc:
        raise RuntimeError(f"Unable to open Windows file picker: {exc}") from exc
    finally:
        root.destroy()


def _select_folder_path() -> str | None:
    try:
        from tkinter import Tk, filedialog
    except Exception as exc:
        raise RuntimeError("Windows folder picker is not available in this Python environment.") from exc

    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
    except Exception as exc:
        raise RuntimeError(f"Unable to open Windows folder picker: {exc}") from exc

    try:
        folder = filedialog.askdirectory(title="Select training data folder")
        return folder or None
    except Exception as exc:
        raise RuntimeError(f"Unable to open Windows folder picker: {exc}") from exc
    finally:
        root.destroy()


def _supported_image_paths(paths: Iterable[Path]) -> list[str]:
    selected: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            selected.append(str(candidate))
    return sorted(dict.fromkeys(selected))


app = create_app()
