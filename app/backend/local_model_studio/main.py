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
from local_model_studio.global_learning import create_global_learning_job
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
    ApiKeyTestResponse,
    DatasetPrepJobStatus,
    DatasetPrepRequest,
    DatasetPrepResponse,
    GenerationJobResponse,
    GenerationJobStatus,
    GenerationOutputImage,
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


CHECKPOINT_NAME = "sdxl_base_1.0.safetensors"


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


def _server_ai_request(request: DatasetPrepRequest, api_keys: ApiKeyStore) -> DatasetPrepRequest:
    if request.effective_scan_mode() != "claude":
        return request
    return request.model_copy(update={"ai_model": api_keys.model()})


def _active_global_lora_files(training: TrainingStore) -> list[str]:
    return [
        job.completed_lora_path
        for job in training.list()
        if job.global_pack and job.completed_lora_path
    ]


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
