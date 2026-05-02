from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from local_model_studio.api_key_store import ApiKeyStore
from local_model_studio.comfy_client import ComfyClient
from local_model_studio.dataset_scanner import scan_dataset
from local_model_studio.dataset_prep_jobs import DatasetPrepJobManager
from local_model_studio.dataset_prepper import _anthropic_error_message, prepare_dataset_crops
from local_model_studio.file_browser import IMAGE_SUFFIXES, browse_filesystem, render_thumbnail
from local_model_studio.generation_settings import GenerationSettingsStore, comfy_checkpoint_path, comfy_lora_path
from local_model_studio.global_learning import create_global_learning_job
from local_model_studio.local_image_tagger import local_tagger_from_workspace
from local_model_studio.local_vision_captioner import local_captioner_from_workspace
from local_model_studio.paths import WorkspacePaths, default_workspace_root
from local_model_studio.prompt_enhancer import enhance_photo_brief
from local_model_studio.profile_store import ProfileStore
from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.reference_analyzer import analyze_references
from local_model_studio.runtime_check import build_live_system_metrics, build_runtime_status
from local_model_studio.schemas import (
    CharacterProfile,
    ApiKeyPayload,
    ApiKeyStatus,
    ApiKeyTestResponse,
    DatasetPrepJobStatus,
    DatasetPrepRequest,
    DatasetPrepResponse,
    GenerationJobResponse,
    GenerationJobStatus,
    GenerationOutputImage,
    GenerationPreflightItem,
    GenerationPreflightResponse,
    GenerationRequest,
    GenerationSettings,
    PathBrowserResponse,
    PromptRecipe,
    PromptEnhanceRequest,
    PromptEnhanceResponse,
    ReferenceAnalysisRequest,
    ReferenceAnalysisResponse,
    RuntimeStatus,
    SelectedFolderResponse,
    SelectedReferenceImagesResponse,
    SystemLiveMetrics,
)
from local_model_studio.training_config import build_training_config, check_trainer_status
from local_model_studio.training_runner import TrainingRunManager
from local_model_studio.training_schemas import (
    ClearTrainingJobsResponse,
    DatasetScanReport,
    DatasetScanRequest,
    GlobalLearningRequest,
    GlobalLearningResponse,
    TrainerStatus,
    TrainingConfigRequest,
    TrainingJobConfig,
    TrainingRunStartRequest,
    TrainingRunStatus,
)
from local_model_studio.training_store import TrainingStore
from local_model_studio.workflow_templates import render_sdxl_workflow


class TrainingConfigPayload(BaseModel):
    request: TrainingConfigRequest
    accepted_image_count: int = Field(gt=0)


class ImageCountResponse(BaseModel):
    path: str
    image_count: int = Field(ge=0)


class RegisterLoraPayload(BaseModel):
    lora_path: Path


def create_app(
    paths: WorkspacePaths | None = None,
    comfy_client: ComfyClient | None = None,
) -> FastAPI:
    workspace_paths = paths or WorkspacePaths(default_workspace_root())
    profiles = ProfileStore(workspace_paths)
    training = TrainingStore(workspace_paths)
    training_runs = TrainingRunManager(workspace_paths, training)
    api_keys = ApiKeyStore(workspace_paths.root)
    generation_settings = GenerationSettingsStore(workspace_paths)
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

    @app.post("/api/settings/anthropic-key/test", response_model=ApiKeyTestResponse)
    def test_anthropic_key() -> ApiKeyTestResponse:
        key = api_keys.api_key()
        model = api_keys.model()
        if not key:
            return ApiKeyTestResponse(provider="anthropic", ok=False, model=model, message="No Claude API key saved.")
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Reply OK"}],
                },
                timeout=20,
            )
        except httpx.HTTPError as exc:
            return ApiKeyTestResponse(provider="anthropic", ok=False, model=model, message=f"Connection failed: {exc}")
        if response.status_code >= 400:
            return ApiKeyTestResponse(
                provider="anthropic",
                ok=False,
                status_code=response.status_code,
                model=model,
                message=_anthropic_error_message(response),
            )
        return ApiKeyTestResponse(provider="anthropic", ok=True, status_code=response.status_code, model=model, message="Claude API connection works.")

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

    @app.get("/api/filesystem/image-count", response_model=ImageCountResponse)
    def image_count(path: str) -> ImageCountResponse:
        folder = Path(path)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")
        count = sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        return ImageCountResponse(path=str(folder), image_count=count)

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
        return build_prompt_recipe(profile, request, _routed_global_lora_files(training, request, profile))

    @app.post("/api/generate/enhance-prompt", response_model=PromptEnhanceResponse)
    def enhance_generation_prompt(request: PromptEnhanceRequest) -> PromptEnhanceResponse:
        profile = _optional_profile_for_enhancement(profiles, request.character_id)
        return enhance_photo_brief(profile, request.brief, request.mode)

    @app.get("/api/generate/settings", response_model=GenerationSettings)
    def get_generation_settings() -> GenerationSettings:
        try:
            return generation_settings.load()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/generate/settings", response_model=GenerationSettings)
    def save_generation_settings(settings: GenerationSettings) -> GenerationSettings:
        return generation_settings.save(settings)

    @app.post("/api/generate/preflight", response_model=GenerationPreflightResponse)
    def generation_preflight(request: GenerationRequest) -> GenerationPreflightResponse:
        profile = _get_profile_or_404(profiles, request.character_id)
        settings = generation_settings.load()
        recipe = build_prompt_recipe(profile, request, _routed_global_lora_files(training, request, profile))
        return _generation_preflight(workspace_paths, comfy, settings, recipe)

    @app.post("/api/generate", response_model=GenerationJobResponse)
    def generate(request: GenerationRequest) -> GenerationJobResponse:
        profile = _get_profile_or_404(profiles, request.character_id)
        settings = generation_settings.load()
        recipe = build_prompt_recipe(profile, request, _routed_global_lora_files(training, request, profile))
        preflight = _generation_preflight(workspace_paths, comfy, settings, recipe)
        if not preflight.ready:
            failed = [item.detail for item in preflight.items if not item.ok]
            raise HTTPException(status_code=409, detail="Generation is not ready: " + " ".join(failed))
        workflow = render_sdxl_workflow(recipe, settings.checkpoint_name)
        try:
            prompt_id = comfy.queue_prompt(workflow)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"ComfyUI request failed: {exc}",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=_friendly_comfy_error(str(exc))) from exc
        return GenerationJobResponse(prompt_id=prompt_id, recipe=recipe)

    @app.get("/api/generate/jobs/{prompt_id}", response_model=GenerationJobStatus)
    def generation_status(prompt_id: str) -> GenerationJobStatus:
        try:
            return _generation_status_from_comfy(comfy, prompt_id)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"ComfyUI request failed: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/generate/image")
    def generated_image(
        filename: str = Query(min_length=1),
        subfolder: str = "",
        type: str = "output",
    ) -> Response:
        try:
            content, media_type = comfy.image_bytes(filename=filename, subfolder=subfolder, image_type=type)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"ComfyUI image request failed: {exc}") from exc
        return Response(content=content, media_type=media_type)

    @app.post("/api/training/scan", response_model=DatasetScanReport)
    def scan_training_dataset(request: DatasetScanRequest) -> DatasetScanReport:
        try:
            return scan_dataset(request, output_root=workspace_paths.root / "datasets")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/training/jobs", response_model=list[TrainingJobConfig])
    def list_training_jobs() -> list[TrainingJobConfig]:
        return training.list()

    @app.delete("/api/training/jobs/pending", response_model=ClearTrainingJobsResponse)
    def clear_pending_training_jobs() -> ClearTrainingJobsResponse:
        removed_count = training.clear_pending_global_jobs()
        return ClearTrainingJobsResponse(
            removed_count=removed_count,
            remaining_jobs=training.list(),
        )

    @app.get("/api/training/runs", response_model=list[TrainingRunStatus])
    def list_training_runs() -> list[TrainingRunStatus]:
        return training_runs.list()

    @app.post("/api/learning/global-job", response_model=GlobalLearningResponse)
    def create_learning_job(request: GlobalLearningRequest) -> GlobalLearningResponse:
        try:
            return create_global_learning_job(request, training)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dataset-prep/crop", response_model=DatasetPrepResponse)
    def prep_dataset_crops(request: DatasetPrepRequest) -> DatasetPrepResponse:
        try:
            prepared_request = _server_ai_request(request, api_keys)
            return prepare_dataset_crops(prepared_request, anthropic_api_key=api_keys.api_key() if prepared_request.effective_scan_mode() == "claude" else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dataset-prep/jobs", response_model=DatasetPrepJobStatus)
    def start_dataset_prep_job(request: DatasetPrepRequest) -> DatasetPrepJobStatus:
        try:
            prepared_request = _server_ai_request(request, api_keys)
            return prep_jobs.start(prepared_request, anthropic_api_key=api_keys.api_key() if prepared_request.effective_scan_mode() == "claude" else None)
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
        return check_trainer_status(
            _resolve_workspace_path(workspace_paths.root, trainer_entrypoint),
            _resolve_workspace_path(workspace_paths.root, resolved_config_path),
        )

    @app.post("/api/training/{job_id}/runs", response_model=TrainingRunStatus)
    def start_training_run(job_id: str, payload: TrainingRunStartRequest) -> TrainingRunStatus:
        try:
            return training_runs.start(job_id, payload.trainer_entrypoint)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training job not found.") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/training/runs/{run_id}", response_model=TrainingRunStatus)
    def get_training_run(run_id: str) -> TrainingRunStatus:
        try:
            return training_runs.get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/training/runs/{run_id}/cancel", response_model=TrainingRunStatus)
    def cancel_training_run(run_id: str) -> TrainingRunStatus:
        try:
            return training_runs.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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


def _optional_profile_for_enhancement(store: ProfileStore, profile_id: str) -> CharacterProfile:
    if not profile_id:
        return CharacterProfile(id="draft", display_name="Draft character")
    try:
        return store.get(profile_id)
    except (KeyError, ValueError):
        return CharacterProfile(id="draft", display_name="Draft character")


def _server_ai_request(request: DatasetPrepRequest, api_keys: ApiKeyStore) -> DatasetPrepRequest:
    if request.effective_scan_mode() != "claude":
        return request
    return request.model_copy(update={"ai_model": api_keys.model()})


def _routed_global_lora_files(
    training: TrainingStore,
    request: GenerationRequest,
    profile: CharacterProfile,
) -> list[str]:
    if request.global_lora_files is not None:
        return request.global_lora_files
    prompt_text = " ".join(
        [
            request.scene_prompt,
            request.extra_negative,
            profile.display_name,
            profile.id,
            profile.body_shape,
            profile.chest,
            profile.style_notes,
        ]
    ).lower()
    scored: list[tuple[int, str]] = []
    for job in training.list():
        if not job.global_pack or not job.completed_lora_path:
            continue
        score = _lora_route_score(job.dataset_type, job.lora_name, prompt_text, profile)
        if score > 0:
            scored.append((score, job.completed_lora_path))
    return [path for _, path in sorted(scored, key=lambda item: item[0], reverse=True)[:4]]


def _lora_route_score(
    dataset_type: str | None,
    lora_name: str,
    prompt_text: str,
    profile: CharacterProfile,
) -> int:
    name = lora_name.lower().replace("_", " ")
    text = f"{prompt_text} {name}"
    score = 0
    if dataset_type in {"body_part", "body_shape"}:
        score += _contains_any(prompt_text, ["nude", "body", "figure", "torso", "waist", "hips", "chest", "breast", "curves", "full body", "bending"])
    elif dataset_type == "pose":
        score += _contains_any(prompt_text, ["pose", "standing", "sitting", "lying", "kneeling", "bending", "walking", "looking back"])
    elif dataset_type == "style":
        score += _contains_any(prompt_text, ["film", "studio", "selfie", "cinematic", "candid", "polaroid", "flash", "low light", "lighting"])
    elif dataset_type == "fictional_face_identity":
        character_terms = [profile.id.lower(), profile.display_name.lower()]
        score += 3 if any(term and term in name for term in character_terms) else 0
    if score > 0:
        score += _contains_any(name, ["body", "pose", "style", "realism", "chest", "breast", "identity"])
    return score


def _contains_any(value: str, needles: list[str]) -> int:
    return sum(1 for needle in needles if needle in value)


def _generation_preflight(
    paths: WorkspacePaths,
    comfy: ComfyClient,
    settings: GenerationSettings,
    recipe: PromptRecipe,
) -> GenerationPreflightResponse:
    checkpoint_path = comfy_checkpoint_path(paths, settings.checkpoint_name)
    lora_paths = [comfy_lora_path(paths, lora_file) for lora_file in recipe.lora_files]
    comfy_available = _comfy_is_available(comfy)
    checkpoint_exists = checkpoint_path.is_file()
    items = [
        GenerationPreflightItem(
            id="comfyui",
            label="ComfyUI API",
            ok=comfy_available,
            detail="ComfyUI is responding on 127.0.0.1:8188." if comfy_available else "ComfyUI is not responding on 127.0.0.1:8188.",
        ),
        GenerationPreflightItem(
            id="checkpoint",
            label="Checkpoint",
            ok=checkpoint_exists,
            detail=f"{settings.checkpoint_name} found." if checkpoint_exists else f"Missing {checkpoint_path}.",
        ),
        GenerationPreflightItem(
            id="loras",
            label="LoRA files",
            ok=all(path.is_file() for path in lora_paths),
            detail=_lora_preflight_detail(lora_paths),
        ),
    ]
    return GenerationPreflightResponse(
        ready=all(item.ok for item in items),
        settings=settings,
        items=items,
        warnings=[],
    )


def _comfy_is_available(comfy: ComfyClient) -> bool:
    checker = getattr(comfy, "is_available", None)
    if not callable(checker):
        return False
    return bool(checker())


def _lora_preflight_detail(lora_paths: list[Path]) -> str:
    if not lora_paths:
        return "No LoRA files selected."
    missing = [path for path in lora_paths if not path.is_file()]
    if missing:
        return "Missing LoRA file: " + ", ".join(str(path) for path in missing)
    return f"{len(lora_paths)} LoRA file{'s' if len(lora_paths) != 1 else ''} found."


def _friendly_comfy_error(message: str) -> str:
    lowered = message.lower()
    if "checkpoint" in lowered or "ckpt" in lowered:
        return f"{message} Check that the configured checkpoint exists in ComfyUI\\models\\checkpoints."
    if "lora" in lowered:
        return f"{message} Check that selected LoRA files exist in ComfyUI\\models\\loras or use absolute paths."
    if "connect" in lowered or "connection" in lowered:
        return "ComfyUI is not reachable at 127.0.0.1:8188. Start ComfyUI before queuing generation."
    return message


def _resolve_workspace_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _generation_status_from_comfy(comfy: ComfyClient, prompt_id: str) -> GenerationJobStatus:
    history = comfy.history(prompt_id)
    history_item = history.get(prompt_id)
    if isinstance(history_item, dict):
        images = _images_from_history(history_item)
        if images:
            return GenerationJobStatus(prompt_id=prompt_id, status="completed", images=images)
        status = history_item.get("status")
        if isinstance(status, dict) and status.get("status_str") == "error":
            messages = status.get("messages")
            return GenerationJobStatus(prompt_id=prompt_id, status="failed", error=str(messages or "ComfyUI reported an error."))
        return GenerationJobStatus(prompt_id=prompt_id, status="completed", images=[])

    queue = comfy.queue()
    running_ids = _prompt_ids_from_queue_items(queue.get("queue_running"))
    if prompt_id in running_ids:
        return GenerationJobStatus(prompt_id=prompt_id, status="running", queue_position=0)

    pending_ids = _prompt_ids_from_queue_items(queue.get("queue_pending"))
    if prompt_id in pending_ids:
        return GenerationJobStatus(prompt_id=prompt_id, status="queued", queue_position=pending_ids.index(prompt_id) + 1)

    return GenerationJobStatus(prompt_id=prompt_id, status="unknown", error="Job is not in ComfyUI queue or history yet.")


def _images_from_history(history_item: dict[str, Any]) -> list[GenerationOutputImage]:
    outputs = history_item.get("outputs")
    if not isinstance(outputs, dict):
        return []

    images: list[GenerationOutputImage] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        output_images = output.get("images")
        if not isinstance(output_images, list):
            continue
        for image in output_images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            images.append(
                GenerationOutputImage(
                    filename=filename,
                    subfolder=str(image.get("subfolder") or ""),
                    type=str(image.get("type") or "output"),
                )
            )
    return images


def _prompt_ids_from_queue_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    prompt_ids: list[str] = []
    for item in value:
        if isinstance(item, list) and len(item) > 1 and isinstance(item[1], str):
            prompt_ids.append(item[1])
        elif isinstance(item, dict):
            prompt_id = item.get("prompt_id")
            if isinstance(prompt_id, str):
                prompt_ids.append(prompt_id)
    return prompt_ids


def _select_reference_paths(mode: str) -> list[str]:
    if mode == "files":
        return _supported_image_paths(Path(path) for path in _select_windows_files())

    folder = _select_folder_path("Select reference image folder")
    if not folder:
        return []
    return _supported_image_paths(path for path in Path(folder).rglob("*") if path.is_file())


def _select_folder_path(title: str = "Select training data folder") -> str | None:
    if sys.platform != "win32":
        raise RuntimeError("Windows folder picker is only available on Windows.")
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
$OutputPath = '{_ps_quote("__OUTPUT_PATH__")}'
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{_ps_quote(title)}'
$dialog.ShowNewFolderButton = $true
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = 'CenterScreen'
$owner.Width = 1
$owner.Height = 1
$owner.ShowInTaskbar = $false
$owner.Show()
$owner.Activate()
$result = $dialog.ShowDialog($owner)
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
  Set-Content -LiteralPath $OutputPath -Encoding UTF8 -Value $dialog.SelectedPath
}}
$owner.Close()
$owner.Dispose()
"""
    output = _run_sta_powershell_picker(script)
    return output[0] if output else None


def _select_windows_files() -> list[str]:
    if sys.platform != "win32":
        raise RuntimeError("Windows file picker is only available on Windows.")
    script = """
Add-Type -AssemblyName System.Windows.Forms
$OutputPath = '__OUTPUT_PATH__'
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Select reference images'
$dialog.Filter = 'Image files (*.avif;*.bmp;*.jpeg;*.jpg;*.png;*.webp)|*.avif;*.bmp;*.jpeg;*.jpg;*.png;*.webp|All files (*.*)|*.*'
$dialog.Multiselect = $true
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = 'CenterScreen'
$owner.Width = 1
$owner.Height = 1
$owner.ShowInTaskbar = $false
$owner.Show()
$owner.Activate()
$result = $dialog.ShowDialog($owner)
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  Set-Content -LiteralPath $OutputPath -Encoding UTF8 -Value $dialog.FileNames
}
$owner.Close()
$owner.Dispose()
"""
    return _run_sta_powershell_picker(script)


def _run_sta_powershell_picker(script: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="liidar-picker-") as temp_dir:
        temp_path = Path(temp_dir)
        output_path = temp_path / "selected.txt"
        error_path = temp_path / "picker.err.txt"
        script_path = temp_path / "picker.ps1"
        script_path.write_text(
            script.replace("__OUTPUT_PATH__", str(output_path).replace("'", "''")),
            encoding="utf-8",
        )
        escaped_script_path = str(script_path).replace("'", "''")

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    "$process = Start-Process powershell.exe "
                    f"-ArgumentList @('-NoProfile','-STA','-ExecutionPolicy','Bypass','-File','{escaped_script_path}') "
                    "-WindowStyle Normal -Wait -PassThru; "
                    "exit $process.ExitCode"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=None,
            check=False,
        )
        if result.returncode != 0:
            detail = (error_path.read_text(encoding="utf-8", errors="replace") if error_path.exists() else result.stderr or result.stdout or "Unknown picker error.").strip()
            raise RuntimeError(f"Unable to open Windows picker: {detail}")
        if not output_path.exists():
            return []
        return [line.strip().lstrip("\ufeff") for line in output_path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.strip()]


def _ps_quote(value: str) -> str:
    return value.replace("'", "''")


def _supported_image_paths(paths: Iterable[Path]) -> list[str]:
    selected: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            selected.append(str(candidate))
    return sorted(dict.fromkeys(selected))


app = create_app()
