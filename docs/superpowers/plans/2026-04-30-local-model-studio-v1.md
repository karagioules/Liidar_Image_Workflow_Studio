# Local Model Studio V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Windows app that manages fictional adult character profiles, renders simple generation recipes, and submits jobs to a local ComfyUI server without cloud token costs.

**Architecture:** Use a Python FastAPI backend for profile storage, runtime checks, prompt rendering, and ComfyUI API calls. Use a Vite React frontend for a simple browser UI. Keep ComfyUI as an external local engine managed by setup scripts rather than bundling it into the app code.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic, pytest, httpx, Vite, React, TypeScript, Vitest, PowerShell setup scripts, ComfyUI local API.

---

## File Structure

- Create `app/backend/pyproject.toml`: backend package metadata and test dependencies.
- Create `app/backend/local_model_studio/__init__.py`: package marker.
- Create `app/backend/local_model_studio/paths.py`: central workspace path helpers.
- Create `app/backend/local_model_studio/schemas.py`: character, generation, runtime, and output metadata schemas.
- Create `app/backend/local_model_studio/profile_store.py`: JSON profile persistence under `config/characters/`.
- Create `app/backend/local_model_studio/prompt_builder.py`: converts structured character profiles and generation requests into positive/negative prompts.
- Create `app/backend/local_model_studio/runtime_check.py`: detects OS, CPU, RAM, GPU, driver, Python, and ComfyUI folder.
- Create `app/backend/local_model_studio/comfy_client.py`: small ComfyUI HTTP API adapter.
- Create `app/backend/local_model_studio/workflow_templates.py`: renders a minimal SDXL-style ComfyUI workflow from app settings.
- Create `app/backend/local_model_studio/main.py`: FastAPI routes.
- Create `app/backend/tests/*.py`: focused backend tests.
- Create `app/frontend/package.json`: frontend scripts and dependencies.
- Create `app/frontend/index.html`: browser entry point.
- Create `app/frontend/src/main.tsx`: React bootstrap.
- Create `app/frontend/src/App.tsx`: app shell and page state.
- Create `app/frontend/src/api.ts`: backend API wrapper.
- Create `app/frontend/src/types.ts`: TypeScript types matching backend schemas.
- Create `app/frontend/src/App.test.tsx`: smoke tests for core UI states.
- Create `app/frontend/src/styles.css`: restrained app styling.
- Create `tools/setup/check-system.ps1`: Windows hardware/runtime detection helper.
- Create `tools/setup/install-comfyui.ps1`: official-first ComfyUI setup helper.
- Create `Start-Liidar.ps1`: launches backend and frontend locally.
- Create `config/presets/scene-presets.json`: initial generation mode presets.
- Create `config/workflows/README.md`: explains generated workflow templates.
- Modify `.gitignore`: ignore generated outputs, local virtualenvs, ComfyUI install, logs, and downloaded model files.
- Keep `Models/` untouched.

## Task 1: Repository Hygiene and Backend Skeleton

**Files:**
- Create: `.gitignore`
- Create: `app/backend/pyproject.toml`
- Create: `app/backend/local_model_studio/__init__.py`
- Create: `app/backend/local_model_studio/paths.py`
- Test: `app/backend/tests/test_paths.py`

- [ ] **Step 1: Write the failing path test**

Create `app/backend/tests/test_paths.py`:

```python
from pathlib import Path

from local_model_studio.paths import WorkspacePaths


def test_workspace_paths_are_rooted_in_repo(tmp_path: Path) -> None:
    paths = WorkspacePaths(tmp_path)

    assert paths.root == tmp_path
    assert paths.characters_dir == tmp_path / "config" / "characters"
    assert paths.outputs_dir == tmp_path / "outputs" / "images"
    assert paths.metadata_dir == tmp_path / "outputs" / "metadata"
    assert paths.logs_dir == tmp_path / "logs"
```

- [ ] **Step 2: Add backend project metadata**

Create `app/backend/pyproject.toml`:

```toml
[project]
name = "local-model-studio-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.8.0",
  "httpx>=0.27.0",
  "python-multipart>=0.0.9"
]

[project.optional-dependencies]
test = [
  "pytest>=8.2.0",
  "pytest-cov>=5.0.0",
  "respx>=0.21.0"
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Add ignore rules**

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
node_modules/
dist/
ComfyUI/
models/
*.safetensors
*.ckpt
*.pt
*.pth
outputs/
logs/
config/characters/*.json
!config/characters/.gitkeep
```

- [ ] **Step 4: Implement path helpers**

Create `app/backend/local_model_studio/__init__.py`:

```python
"""Local Model Studio backend package."""
```

Create `app/backend/local_model_studio/paths.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def characters_dir(self) -> Path:
        return self.root / "config" / "characters"

    @property
    def presets_dir(self) -> Path:
        return self.root / "config" / "presets"

    @property
    def workflows_dir(self) -> Path:
        return self.root / "config" / "workflows"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs" / "images"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "outputs" / "metadata"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


def default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]
```

- [ ] **Step 5: Run the backend path test**

Run:

```powershell
cd app\backend
python -m pip install -e .[test]
python -m pytest tests/test_paths.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore app/backend
git commit -m "chore: scaffold backend workspace"
```

## Task 2: Character Profile Schema and Store

**Files:**
- Create: `app/backend/local_model_studio/schemas.py`
- Create: `app/backend/local_model_studio/profile_store.py`
- Test: `app/backend/tests/test_profile_store.py`

- [ ] **Step 1: Write failing profile store tests**

Create `app/backend/tests/test_profile_store.py`:

```python
from pathlib import Path

import pytest

from local_model_studio.paths import WorkspacePaths
from local_model_studio.profile_store import ProfileStore
from local_model_studio.schemas import CharacterProfile


def make_profile(name: str = "Domna Test") -> CharacterProfile:
    return CharacterProfile(
        display_name=name,
        age_category="adult_25_plus",
        face_summary="soft oval face, brown eyes, warm smile",
        hair="long dark brunette hair",
        skin_tone="olive skin tone",
        body_shape="curvy natural build",
        chest="natural teardrop chest shape",
        grooming="natural adult grooming",
        style_notes="Mediterranean lifestyle photography",
        negative_notes="cartoon, distorted hands, low detail",
    )


def test_save_and_load_profile(tmp_path: Path) -> None:
    store = ProfileStore(WorkspacePaths(tmp_path))
    saved = store.save(make_profile())

    loaded = store.get(saved.id)

    assert loaded.id == saved.id
    assert loaded.display_name == "Domna Test"
    assert loaded.age_category == "adult_25_plus"


def test_rejects_non_adult_age_category(tmp_path: Path) -> None:
    store = ProfileStore(WorkspacePaths(tmp_path))

    with pytest.raises(ValueError, match="adult"):
        store.save(make_profile().model_copy(update={"age_category": "teen"}))


def test_list_profiles_sorts_by_display_name(tmp_path: Path) -> None:
    store = ProfileStore(WorkspacePaths(tmp_path))
    store.save(make_profile("Zeta"))
    store.save(make_profile("Alpha"))

    names = [profile.display_name for profile in store.list()]

    assert names == ["Alpha", "Zeta"]
```

- [ ] **Step 2: Implement schemas**

Create `app/backend/local_model_studio/schemas.py`:

```python
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
```

- [ ] **Step 3: Implement the JSON store**

Create `app/backend/local_model_studio/profile_store.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import CharacterProfile


class ProfileStore:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self.paths.characters_dir.mkdir(parents=True, exist_ok=True)

    def save(self, profile: CharacterProfile) -> CharacterProfile:
        now = datetime.now(UTC)
        existing_created_at = profile.created_at
        path = self._path_for(profile.id)
        if path.exists():
            existing_created_at = CharacterProfile.model_validate_json(path.read_text(encoding="utf-8")).created_at
        saved = profile.model_copy(update={"created_at": existing_created_at, "updated_at": now})
        path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
        return saved

    def list(self) -> list[CharacterProfile]:
        profiles = [self._load_path(path) for path in self.paths.characters_dir.glob("*.json")]
        return sorted(profiles, key=lambda profile: profile.display_name.casefold())

    def get(self, profile_id: str) -> CharacterProfile:
        path = self._path_for(profile_id)
        if not path.exists():
            raise KeyError(profile_id)
        return self._load_path(path)

    def delete(self, profile_id: str) -> None:
        path = self._path_for(profile_id)
        if not path.exists():
            raise KeyError(profile_id)
        path.unlink()

    def _path_for(self, profile_id: str) -> Path:
        safe_id = "".join(char for char in profile_id if char.isalnum() or char in {"-", "_"})
        if not safe_id:
            raise ValueError("Profile id must contain safe filename characters.")
        return self.paths.characters_dir / f"{safe_id}.json"

    def _load_path(self, path: Path) -> CharacterProfile:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CharacterProfile.model_validate(data)
```

- [ ] **Step 4: Run profile tests**

Run:

```powershell
cd app\backend
python -m pytest tests/test_profile_store.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```powershell
git add app/backend/local_model_studio/schemas.py app/backend/local_model_studio/profile_store.py app/backend/tests/test_profile_store.py
git commit -m "feat: add character profile storage"
```

## Task 3: Prompt Builder and Presets

**Files:**
- Create: `app/backend/local_model_studio/prompt_builder.py`
- Create: `config/presets/scene-presets.json`
- Test: `app/backend/tests/test_prompt_builder.py`

- [ ] **Step 1: Write failing prompt tests**

Create `app/backend/tests/test_prompt_builder.py`:

```python
from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.schemas import CharacterProfile, GenerationRequest


def profile() -> CharacterProfile:
    return CharacterProfile(
        display_name="Marianna",
        age_category="adult_25_plus",
        face_summary="consistent soft face, green eyes",
        hair="wavy black hair",
        skin_tone="warm olive skin",
        body_shape="petite natural build",
        chest="small natural chest",
        grooming="soft natural adult grooming",
        style_notes="realistic influencer photography",
        negative_notes="deformed anatomy, extra fingers",
        locked_seed=1234,
        seed_strategy="locked",
    )


def test_builds_locked_identity_prompt() -> None:
    recipe = build_prompt_recipe(
        profile(),
        GenerationRequest(character_id="abc", mode="lifestyle_post", scene_prompt="sunlit balcony selfie"),
    )

    assert "Marianna" not in recipe.positive
    assert "consistent soft face" in recipe.positive
    assert "small natural chest" in recipe.positive
    assert "sunlit balcony selfie" in recipe.positive
    assert "deformed anatomy" in recipe.negative
    assert recipe.seed == 1234


def test_quality_changes_dimensions_and_steps() -> None:
    fast = build_prompt_recipe(profile(), GenerationRequest(character_id="abc", quality="fast"))
    ultra = build_prompt_recipe(profile(), GenerationRequest(character_id="abc", quality="ultra"))

    assert fast.steps < ultra.steps
    assert fast.width <= ultra.width
    assert fast.height <= ultra.height
```

- [ ] **Step 2: Implement prompt builder**

Create `app/backend/local_model_studio/prompt_builder.py`:

```python
from __future__ import annotations

import secrets

from local_model_studio.schemas import CharacterProfile, GenerationRequest, PromptRecipe


QUALITY_SETTINGS = {
    "fast": {"width": 832, "height": 1216, "steps": 18, "cfg": 4.5},
    "balanced": {"width": 896, "height": 1344, "steps": 26, "cfg": 5.0},
    "high": {"width": 1024, "height": 1536, "steps": 34, "cfg": 5.5},
    "ultra": {"width": 1024, "height": 1536, "steps": 42, "cfg": 6.0},
}

MODE_PROMPTS = {
    "portrait": "professional realistic portrait, sharp eyes, natural skin texture",
    "full_body": "realistic full body photo, coherent anatomy, natural proportions",
    "lifestyle_post": "realistic lifestyle social media photo, candid composition",
    "studio": "professional studio photography, clean lighting, high detail",
    "reference_match": "realistic reference-guided composition, consistent identity",
}

BASE_NEGATIVE = "minor, underage, childlike, celebrity, real person, distorted face, distorted hands, low detail, watermark, text"


def build_prompt_recipe(profile: CharacterProfile, request: GenerationRequest) -> PromptRecipe:
    settings = QUALITY_SETTINGS[request.quality]
    seed = _resolve_seed(profile, request)
    positive_parts = [
        "fictional adult woman, age 25 or older",
        MODE_PROMPTS[request.mode],
        profile.face_summary,
        profile.hair,
        profile.eyes,
        profile.skin_tone,
        profile.body_shape,
        profile.chest,
        profile.grooming,
        profile.style_notes,
        request.scene_prompt,
        "photorealistic, high-end creator content, consistent character identity",
    ]
    negative_parts = [BASE_NEGATIVE, profile.negative_notes, request.extra_negative]
    return PromptRecipe(
        positive=", ".join(part.strip() for part in positive_parts if part.strip()),
        negative=", ".join(part.strip() for part in negative_parts if part.strip()),
        seed=seed,
        width=settings["width"],
        height=settings["height"],
        steps=settings["steps"],
        cfg=settings["cfg"],
    )


def _resolve_seed(profile: CharacterProfile, request: GenerationRequest) -> int:
    if request.seed is not None:
        return request.seed
    if profile.seed_strategy == "locked" and profile.locked_seed is not None:
        return profile.locked_seed
    return secrets.randbelow(2_147_483_647)
```

- [ ] **Step 3: Add scene presets**

Create `config/presets/scene-presets.json`:

```json
[
  {
    "id": "balcony_selfie",
    "label": "Balcony Selfie",
    "mode": "lifestyle_post",
    "prompt": "sunlit balcony selfie, realistic phone camera, relaxed expression"
  },
  {
    "id": "clean_studio",
    "label": "Clean Studio",
    "mode": "studio",
    "prompt": "neutral studio backdrop, softbox lighting, premium editorial detail"
  },
  {
    "id": "beach_walk",
    "label": "Beach Walk",
    "mode": "full_body",
    "prompt": "golden hour beach walk, natural pose, realistic outdoor lighting"
  }
]
```

- [ ] **Step 4: Run prompt tests**

Run:

```powershell
cd app\backend
python -m pytest tests/test_prompt_builder.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add app/backend/local_model_studio/prompt_builder.py app/backend/tests/test_prompt_builder.py config/presets/scene-presets.json
git commit -m "feat: render character prompt recipes"
```

## Task 4: Runtime Detection

**Files:**
- Create: `app/backend/local_model_studio/runtime_check.py`
- Create: `tools/setup/check-system.ps1`
- Test: `app/backend/tests/test_runtime_check.py`

- [ ] **Step 1: Write runtime warning tests**

Create `app/backend/tests/test_runtime_check.py`:

```python
from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio.runtime_check import build_runtime_status


def test_runtime_status_reports_missing_comfyui(tmp_path: Path) -> None:
    status = build_runtime_status(WorkspacePaths(tmp_path), gpu_names=["AMD Radeon RX 9070 XT"], driver_version="32.0")

    assert status.comfyui_path_exists is False
    assert status.gpu_names == ["AMD Radeon RX 9070 XT"]
    assert any("ComfyUI" in warning for warning in status.warnings)


def test_runtime_status_warns_when_no_amd_gpu(tmp_path: Path) -> None:
    status = build_runtime_status(WorkspacePaths(tmp_path), gpu_names=["Microsoft Basic Display"], driver_version=None)

    assert any("AMD Radeon" in warning for warning in status.warnings)
```

- [ ] **Step 2: Implement runtime detection**

Create `app/backend/local_model_studio/runtime_check.py`:

```python
from __future__ import annotations

import platform
import subprocess
import sys

from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import RuntimeStatus


def build_runtime_status(
    paths: WorkspacePaths,
    gpu_names: list[str] | None = None,
    driver_version: str | None = None,
) -> RuntimeStatus:
    detected_gpu_names, detected_driver = _detect_windows_gpus()
    final_gpu_names = gpu_names if gpu_names is not None else detected_gpu_names
    final_driver = driver_version if driver_version is not None else detected_driver
    comfyui_exists = (paths.root / "ComfyUI").exists()
    warnings: list[str] = []
    if not any("AMD Radeon" in name for name in final_gpu_names):
        warnings.append("No AMD Radeon GPU was detected. Local GPU generation may not work.")
    if not comfyui_exists:
        warnings.append("ComfyUI is not installed in the workspace yet.")
    return RuntimeStatus(
        os_name=platform.platform(),
        python_version=sys.version.split()[0],
        cpu_name=platform.processor() or "Unknown CPU",
        total_ram_gb=_detect_ram_gb(),
        gpu_names=final_gpu_names,
        amd_driver_version=final_driver,
        comfyui_path_exists=comfyui_exists,
        warnings=warnings,
    )


def _detect_windows_gpus() -> tuple[list[str], str | None]:
    if platform.system() != "Windows":
        return [], None
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return [], None
    text = result.stdout.strip()
    if "AMD Radeon" in text:
        driver_line = next((line for line in text.splitlines() if "DriverVersion" in line), "")
        driver = driver_line.split(":", 1)[-1].strip().strip('",') if driver_line else None
        return [line.split(":", 1)[-1].strip().strip('",') for line in text.splitlines() if '"Name"' in line], driver
    return [], None


def _detect_ram_gb() -> float:
    try:
        return round((getattr(__import__("psutil"), "virtual_memory")().total / (1024**3)), 1)
    except Exception:
        return 0.0
```

- [ ] **Step 3: Add PowerShell check helper**

Create `tools/setup/check-system.ps1`:

```powershell
$ErrorActionPreference = "Stop"

Write-Host "Local Model Studio system check"
Write-Host "Windows:" (Get-CimInstance Win32_OperatingSystem).Caption
Write-Host "CPU:" (Get-CimInstance Win32_Processor).Name
$ramGb = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Write-Host "RAM:" "$ramGb GB"
Get-CimInstance Win32_VideoController | ForEach-Object {
  Write-Host "GPU:" $_.Name
  Write-Host "Driver:" $_.DriverVersion
}
Write-Host "Python:" (python --version)
```

- [ ] **Step 4: Add psutil dependency**

Modify `app/backend/pyproject.toml` dependencies:

```toml
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.8.0",
  "httpx>=0.27.0",
  "python-multipart>=0.0.9",
  "psutil>=6.0.0"
]
```

- [ ] **Step 5: Run runtime tests and local checker**

Run:

```powershell
cd app\backend
python -m pip install -e .[test]
python -m pytest tests/test_runtime_check.py -v
cd ..\..
powershell -ExecutionPolicy Bypass -File tools\setup\check-system.ps1
```

Expected: tests pass and the checker prints the RX 9070 XT.

- [ ] **Step 6: Commit**

```powershell
git add app/backend/local_model_studio/runtime_check.py app/backend/tests/test_runtime_check.py app/backend/pyproject.toml tools/setup/check-system.ps1
git commit -m "feat: add runtime detection"
```

## Task 5: ComfyUI Client and Workflow Rendering

**Files:**
- Create: `app/backend/local_model_studio/comfy_client.py`
- Create: `app/backend/local_model_studio/workflow_templates.py`
- Create: `config/workflows/README.md`
- Test: `app/backend/tests/test_comfy_client.py`
- Test: `app/backend/tests/test_workflow_templates.py`

- [ ] **Step 1: Write workflow renderer test**

Create `app/backend/tests/test_workflow_templates.py`:

```python
from local_model_studio.schemas import PromptRecipe
from local_model_studio.workflow_templates import render_sdxl_workflow


def test_render_sdxl_workflow_contains_prompt_and_seed() -> None:
    recipe = PromptRecipe(
        positive="fictional adult, realistic portrait",
        negative="minor, low detail",
        seed=42,
        width=896,
        height=1344,
        steps=26,
        cfg=5.0,
    )

    workflow = render_sdxl_workflow(recipe, checkpoint_name="example.safetensors")

    assert workflow["3"]["inputs"]["seed"] == 42
    assert workflow["5"]["inputs"]["text"] == "fictional adult, realistic portrait"
    assert workflow["6"]["inputs"]["text"] == "minor, low detail"
    assert workflow["4"]["inputs"]["ckpt_name"] == "example.safetensors"
```

- [ ] **Step 2: Write ComfyUI client test**

Create `app/backend/tests/test_comfy_client.py`:

```python
import respx
from httpx import Response

from local_model_studio.comfy_client import ComfyClient


@respx.mock
def test_queue_prompt_returns_prompt_id() -> None:
    respx.post("http://127.0.0.1:8188/prompt").mock(return_value=Response(200, json={"prompt_id": "abc123"}))
    client = ComfyClient("http://127.0.0.1:8188")

    prompt_id = client.queue_prompt({"1": {"inputs": {}}})

    assert prompt_id == "abc123"
```

- [ ] **Step 3: Implement workflow template**

Create `app/backend/local_model_studio/workflow_templates.py`:

```python
from __future__ import annotations

from typing import Any

from local_model_studio.schemas import PromptRecipe


def render_sdxl_workflow(recipe: PromptRecipe, checkpoint_name: str) -> dict[str, Any]:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": recipe.seed,
                "steps": recipe.steps,
                "cfg": recipe.cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": recipe.positive, "clip": ["4", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": recipe.negative, "clip": ["4", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": recipe.width, "height": recipe.height, "batch_size": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Liidar", "images": ["8", 0]}},
    }
```

- [ ] **Step 4: Implement ComfyUI client**

Create `app/backend/local_model_studio/comfy_client.py`:

```python
from __future__ import annotations

from typing import Any

import httpx


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/system_stats", timeout=2)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = httpx.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=30)
        response.raise_for_status()
        data = response.json()
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError("ComfyUI did not return a prompt_id.")
        return prompt_id
```

- [ ] **Step 5: Add workflow README**

Create `config/workflows/README.md`:

```markdown
# Workflow Templates

Local Model Studio renders ComfyUI workflows from Python templates.

V1 includes a minimal SDXL-style text-to-image workflow. Advanced reference-image, LoRA, upscale, and video workflows will be added after the local image loop is verified.
```

- [ ] **Step 6: Run workflow/client tests**

Run:

```powershell
cd app\backend
python -m pytest tests/test_workflow_templates.py tests/test_comfy_client.py -v
```

Expected: `2 passed`.

- [ ] **Step 7: Commit**

```powershell
git add app/backend/local_model_studio/comfy_client.py app/backend/local_model_studio/workflow_templates.py app/backend/tests/test_comfy_client.py app/backend/tests/test_workflow_templates.py config/workflows/README.md
git commit -m "feat: add comfyui workflow adapter"
```

## Task 6: FastAPI Backend Routes

**Files:**
- Create: `app/backend/local_model_studio/main.py`
- Test: `app/backend/tests/test_api.py`

- [ ] **Step 1: Write API tests**

Create `app/backend/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from local_model_studio.main import create_app
from local_model_studio.paths import WorkspacePaths


def test_profile_crud_api(tmp_path) -> None:
    client = TestClient(create_app(WorkspacePaths(tmp_path)))
    payload = {
        "display_name": "Kostantia",
        "age_category": "adult_25_plus",
        "face_summary": "defined cheekbones, brown eyes",
        "hair": "dark blonde hair",
        "skin_tone": "light olive skin",
        "body_shape": "athletic natural build",
        "chest": "medium natural chest",
        "grooming": "natural adult grooming",
        "style_notes": "realistic studio photography",
        "negative_notes": "low quality, distorted anatomy"
    }

    create_response = client.post("/api/characters", json=payload)
    assert create_response.status_code == 200
    profile_id = create_response.json()["id"]

    list_response = client.get("/api/characters")
    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == profile_id


def test_generate_preview_returns_recipe_when_comfyui_unavailable(tmp_path) -> None:
    client = TestClient(create_app(WorkspacePaths(tmp_path)))
    profile_response = client.post("/api/characters", json={"display_name": "Preview", "age_category": "adult_25_plus"})
    profile_id = profile_response.json()["id"]

    response = client.post("/api/generate/preview", json={"character_id": profile_id, "scene_prompt": "studio portrait"})

    assert response.status_code == 200
    assert "studio portrait" in response.json()["positive"]
```

- [ ] **Step 2: Implement FastAPI app**

Create `app/backend/local_model_studio/main.py`:

```python
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from local_model_studio.comfy_client import ComfyClient
from local_model_studio.paths import WorkspacePaths, default_workspace_root
from local_model_studio.profile_store import ProfileStore
from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.runtime_check import build_runtime_status
from local_model_studio.schemas import CharacterProfile, GenerationJobResponse, GenerationRequest, PromptRecipe, RuntimeStatus
from local_model_studio.workflow_templates import render_sdxl_workflow


def create_app(paths: WorkspacePaths | None = None, comfy_client: ComfyClient | None = None) -> FastAPI:
    workspace_paths = paths or WorkspacePaths(default_workspace_root())
    store = ProfileStore(workspace_paths)
    comfy = comfy_client or ComfyClient()
    app = FastAPI(title="Local Model Studio")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/runtime", response_model=RuntimeStatus)
    def runtime() -> RuntimeStatus:
        return build_runtime_status(workspace_paths)

    @app.get("/api/characters", response_model=list[CharacterProfile])
    def list_characters() -> list[CharacterProfile]:
        return store.list()

    @app.post("/api/characters", response_model=CharacterProfile)
    def save_character(profile: CharacterProfile) -> CharacterProfile:
        return store.save(profile)

    @app.get("/api/characters/{profile_id}", response_model=CharacterProfile)
    def get_character(profile_id: str) -> CharacterProfile:
        try:
            return store.get(profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Character profile not found") from exc

    @app.delete("/api/characters/{profile_id}", status_code=204)
    def delete_character(profile_id: str) -> None:
        try:
            store.delete(profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Character profile not found") from exc

    @app.post("/api/generate/preview", response_model=PromptRecipe)
    def preview_generation(request: GenerationRequest) -> PromptRecipe:
        try:
            profile = store.get(request.character_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Character profile not found") from exc
        return build_prompt_recipe(profile, request)

    @app.post("/api/generate", response_model=GenerationJobResponse)
    def generate(request: GenerationRequest) -> GenerationJobResponse:
        try:
            profile = store.get(request.character_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Character profile not found") from exc
        recipe = build_prompt_recipe(profile, request)
        workflow = render_sdxl_workflow(recipe, checkpoint_name="sdxl_base_1.0.safetensors")
        prompt_id = comfy.queue_prompt(workflow)
        return GenerationJobResponse(prompt_id=prompt_id, recipe=recipe)

    return app


app = create_app()
```

- [ ] **Step 3: Run API tests**

Run:

```powershell
cd app\backend
python -m pytest tests/test_api.py -v
```

Expected: `2 passed`.

- [ ] **Step 4: Commit**

```powershell
git add app/backend/local_model_studio/main.py app/backend/tests/test_api.py
git commit -m "feat: expose local studio api"
```

## Task 7: Frontend App

**Files:**
- Create: `app/frontend/package.json`
- Create: `app/frontend/index.html`
- Create: `app/frontend/src/main.tsx`
- Create: `app/frontend/src/types.ts`
- Create: `app/frontend/src/api.ts`
- Create: `app/frontend/src/App.tsx`
- Create: `app/frontend/src/App.test.tsx`
- Create: `app/frontend/src/styles.css`

- [ ] **Step 1: Add frontend package**

Create `app/frontend/package.json`:

```json
{
  "name": "local-model-studio-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1",
    "build": "tsc && vite build",
    "test": "vitest run --environment jsdom"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^5.0.0",
    "vite": "^7.0.0",
    "typescript": "^5.8.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "lucide-react": "^0.468.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "vitest": "^3.0.0",
    "jsdom": "^25.0.0"
  }
}
```

- [ ] **Step 2: Add frontend types and API wrapper**

Create `app/frontend/src/types.ts`:

```ts
export type AdultAgeCategory = "adult_18_plus" | "adult_21_plus" | "adult_25_plus" | "adult_30_plus";
export type QualityPreset = "fast" | "balanced" | "high" | "ultra";
export type GenerationMode = "portrait" | "full_body" | "lifestyle_post" | "studio" | "reference_match";

export interface CharacterProfile {
  id?: string;
  display_name: string;
  age_category: AdultAgeCategory;
  face_summary: string;
  hair: string;
  eyes: string;
  skin_tone: string;
  body_shape: string;
  chest: string;
  grooming: string;
  style_notes: string;
  negative_notes: string;
}

export interface PromptRecipe {
  positive: string;
  negative: string;
  seed: number;
  width: number;
  height: number;
  steps: number;
  cfg: number;
}

export interface RuntimeStatus {
  os_name: string;
  python_version: string;
  cpu_name: string;
  total_ram_gb: number;
  gpu_names: string[];
  amd_driver_version: string | null;
  comfyui_path_exists: boolean;
  warnings: string[];
}
```

Create `app/frontend/src/api.ts`:

```ts
import type { CharacterProfile, GenerationMode, PromptRecipe, QualityPreset, RuntimeStatus } from "./types";

const API_BASE = "http://127.0.0.1:8000";

export async function fetchRuntime(): Promise<RuntimeStatus> {
  return fetchJson("/api/runtime");
}

export async function fetchCharacters(): Promise<CharacterProfile[]> {
  return fetchJson("/api/characters");
}

export async function saveCharacter(profile: CharacterProfile): Promise<CharacterProfile> {
  const response = await fetch(`${API_BASE}/api/characters`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function previewGeneration(characterId: string, mode: GenerationMode, quality: QualityPreset, scenePrompt: string): Promise<PromptRecipe> {
  const response = await fetch(`${API_BASE}/api/generate/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ character_id: characterId, mode, quality, scene_prompt: scenePrompt }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
```

- [ ] **Step 3: Add React UI**

Create `app/frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Local Model Studio</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `app/frontend/src/main.tsx`:

```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

Create `app/frontend/src/App.tsx`:

```tsx
import { Play, Save, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchCharacters, fetchRuntime, previewGeneration, saveCharacter } from "./api";
import type { CharacterProfile, GenerationMode, PromptRecipe, QualityPreset, RuntimeStatus } from "./types";

const emptyProfile: CharacterProfile = {
  display_name: "",
  age_category: "adult_25_plus",
  face_summary: "",
  hair: "",
  eyes: "",
  skin_tone: "",
  body_shape: "",
  chest: "",
  grooming: "",
  style_notes: "",
  negative_notes: "",
};

export function App() {
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [characters, setCharacters] = useState<CharacterProfile[]>([]);
  const [profile, setProfile] = useState<CharacterProfile>(emptyProfile);
  const [selectedId, setSelectedId] = useState("");
  const [mode, setMode] = useState<GenerationMode>("portrait");
  const [quality, setQuality] = useState<QualityPreset>("balanced");
  const [scenePrompt, setScenePrompt] = useState("");
  const [recipe, setRecipe] = useState<PromptRecipe | null>(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    fetchRuntime().then(setRuntime).catch((error) => setMessage(String(error)));
    fetchCharacters().then(setCharacters).catch(() => setCharacters([]));
  }, []);

  async function onSave() {
    const saved = await saveCharacter(profile);
    setProfile(saved);
    setSelectedId(saved.id ?? "");
    setCharacters(await fetchCharacters());
    setMessage("Character saved.");
  }

  async function onPreview() {
    const id = selectedId || profile.id;
    if (!id) {
      setMessage("Save or select a character first.");
      return;
    }
    setRecipe(await previewGeneration(id, mode, quality, scenePrompt));
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>Local Model Studio</h1>
          <p>Fictional adult character profiles, local generation, no token meter.</p>
        </div>
        <div className="guardrail"><ShieldCheck size={18} /> Fictional 18+ only</div>
      </header>

      <section className="status">
        <strong>Runtime</strong>
        <span>{runtime?.gpu_names.join(", ") || "Checking GPU..."}</span>
        <span>{runtime?.comfyui_path_exists ? "ComfyUI found" : "ComfyUI not installed yet"}</span>
      </section>

      <div className="grid">
        <section className="panel">
          <h2>Character</h2>
          <select value={selectedId} onChange={(event) => {
            const selected = characters.find((item) => item.id === event.target.value);
            setSelectedId(event.target.value);
            if (selected) setProfile(selected);
          }}>
            <option value="">New character</option>
            {characters.map((item) => <option key={item.id} value={item.id}>{item.display_name}</option>)}
          </select>
          <input placeholder="Display name" value={profile.display_name} onChange={(event) => setProfile({ ...profile, display_name: event.target.value })} />
          <textarea placeholder="Face identity" value={profile.face_summary} onChange={(event) => setProfile({ ...profile, face_summary: event.target.value })} />
          <input placeholder="Hair" value={profile.hair} onChange={(event) => setProfile({ ...profile, hair: event.target.value })} />
          <input placeholder="Eyes" value={profile.eyes} onChange={(event) => setProfile({ ...profile, eyes: event.target.value })} />
          <input placeholder="Skin tone" value={profile.skin_tone} onChange={(event) => setProfile({ ...profile, skin_tone: event.target.value })} />
          <textarea placeholder="Body shape and proportions" value={profile.body_shape} onChange={(event) => setProfile({ ...profile, body_shape: event.target.value })} />
          <input placeholder="Chest shape/size" value={profile.chest} onChange={(event) => setProfile({ ...profile, chest: event.target.value })} />
          <input placeholder="Grooming/body hair" value={profile.grooming} onChange={(event) => setProfile({ ...profile, grooming: event.target.value })} />
          <textarea placeholder="Style notes" value={profile.style_notes} onChange={(event) => setProfile({ ...profile, style_notes: event.target.value })} />
          <textarea placeholder="Negative notes" value={profile.negative_notes} onChange={(event) => setProfile({ ...profile, negative_notes: event.target.value })} />
          <button onClick={onSave}><Save size={18} /> Save character</button>
        </section>

        <section className="panel">
          <h2>Generate</h2>
          <select value={mode} onChange={(event) => setMode(event.target.value as GenerationMode)}>
            <option value="portrait">Portrait</option>
            <option value="full_body">Full Body</option>
            <option value="lifestyle_post">Lifestyle Post</option>
            <option value="studio">Studio</option>
            <option value="reference_match">Reference Match</option>
          </select>
          <select value={quality} onChange={(event) => setQuality(event.target.value as QualityPreset)}>
            <option value="fast">Fast</option>
            <option value="balanced">Balanced</option>
            <option value="high">High</option>
            <option value="ultra">Ultra</option>
          </select>
          <textarea placeholder="Scene prompt" value={scenePrompt} onChange={(event) => setScenePrompt(event.target.value)} />
          <button onClick={onPreview}><Play size={18} /> Preview recipe</button>
          {message && <p className="message">{message}</p>}
          {recipe && (
            <div className="recipe">
              <h3>Prompt Recipe</h3>
              <p>{recipe.positive}</p>
              <strong>Negative</strong>
              <p>{recipe.negative}</p>
              <small>Seed {recipe.seed} · {recipe.width}x{recipe.height} · {recipe.steps} steps</small>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
```

- [ ] **Step 4: Add frontend styles**

Create `app/frontend/src/styles.css`:

```css
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f4f1ea;
  color: #191816;
}
button, input, select, textarea {
  font: inherit;
}
.shell {
  max-width: 1320px;
  margin: 0 auto;
  padding: 24px;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding-bottom: 18px;
  border-bottom: 1px solid #d8d1c4;
}
h1, h2, h3, p {
  margin-top: 0;
}
.guardrail, .status {
  display: flex;
  align-items: center;
  gap: 10px;
}
.status {
  margin: 18px 0;
  padding: 12px 0;
  border-bottom: 1px solid #d8d1c4;
}
.grid {
  display: grid;
  grid-template-columns: minmax(320px, 520px) minmax(320px, 1fr);
  gap: 20px;
}
.panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
input, select, textarea {
  width: 100%;
  border: 1px solid #c8bfb0;
  border-radius: 6px;
  padding: 10px 12px;
  background: #fffdf8;
  color: #191816;
}
textarea {
  min-height: 82px;
  resize: vertical;
}
button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 0;
  border-radius: 6px;
  padding: 11px 14px;
  background: #2b5f67;
  color: white;
  cursor: pointer;
}
.recipe {
  border-top: 1px solid #d8d1c4;
  padding-top: 16px;
}
.message {
  color: #2b5f67;
}
@media (max-width: 840px) {
  .topbar, .status {
    align-items: flex-start;
    flex-direction: column;
  }
  .grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 5: Add UI smoke test**

Create `app/frontend/src/App.test.tsx`:

```tsx
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "./App";

vi.stubGlobal("fetch", vi.fn((url: string) => {
  if (url.endsWith("/api/runtime")) {
    return Promise.resolve(new Response(JSON.stringify({
      os_name: "Windows",
      python_version: "3.12",
      cpu_name: "AMD Ryzen",
      total_ram_gb: 32,
      gpu_names: ["AMD Radeon RX 9070 XT"],
      amd_driver_version: "32.0",
      comfyui_path_exists: false,
      warnings: []
    })));
  }
  return Promise.resolve(new Response(JSON.stringify([])));
}));

describe("App", () => {
  it("renders character and generation panels", async () => {
    render(<App />);

    expect(await screen.findByText("Local Model Studio")).toBeInTheDocument();
    expect(screen.getByText("Character")).toBeInTheDocument();
    expect(screen.getByText("Generate")).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run frontend tests**

Run:

```powershell
cd app\frontend
npm install
npm test
npm run build
```

Expected: test passes and build completes.

- [ ] **Step 7: Commit**

```powershell
git add app/frontend
git commit -m "feat: add local studio frontend"
```

## Task 8: Setup and Launch Scripts

**Files:**
- Create: `tools/setup/install-comfyui.ps1`
- Create: `Start-Liidar.ps1`
- Modify: `docs/superpowers/specs/2026-04-30-local-model-studio-design.md` only if the implementation uncovers a necessary correction.

- [ ] **Step 1: Add ComfyUI setup script**

Create `tools/setup/install-comfyui.ps1`:

```powershell
$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$comfyPath = Join-Path $root "ComfyUI"

if (Test-Path $comfyPath) {
  Write-Host "ComfyUI already exists at $comfyPath"
  exit 0
}

Write-Host "Installing ComfyUI into $comfyPath"
git clone https://github.com/comfyanonymous/ComfyUI.git $comfyPath

Push-Location $comfyPath
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
Write-Host "Install AMD ROCm/PyTorch build according to the current AMD instructions, then run:"
Write-Host ".\.venv\Scripts\python.exe -m pip install -r requirements.txt"
Pop-Location

Write-Host "ComfyUI source installed. Model files must be placed under ComfyUI\models\checkpoints."
```

- [ ] **Step 2: Add launcher**

Create `Start-Liidar.ps1`:

```powershell
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "app\backend"
$frontend = Join-Path $root "app\frontend"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backend'; python -m uvicorn local_model_studio.main:app --reload --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontend'; npm run dev -- --port 5173"

Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
```

- [ ] **Step 3: Run launch smoke check**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\check-system.ps1
cd app\backend
python -m pytest -v
cd ..\frontend
npm test
npm run build
```

Expected: backend tests pass, frontend test passes, frontend build completes.

- [ ] **Step 4: Commit**

```powershell
git add tools/setup/install-comfyui.ps1 Start-Liidar.ps1
git commit -m "chore: add setup and launcher scripts"
```

## Task 9: Local End-to-End Verification

**Files:**
- Modify only if verification exposes failures in files from Tasks 1-8.

- [ ] **Step 1: Run full backend test suite**

Run:

```powershell
cd app\backend
python -m pytest -v
```

Expected: all backend tests pass.

- [ ] **Step 2: Run frontend checks**

Run:

```powershell
cd app\frontend
npm test
npm run build
```

Expected: Vitest passes and Vite builds without TypeScript errors.

- [ ] **Step 3: Start the app**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Start-Liidar.ps1
```

Expected: backend starts at `http://127.0.0.1:8000` and frontend starts at `http://127.0.0.1:5173`.

- [ ] **Step 4: Manual smoke test**

Open `http://127.0.0.1:5173` and verify:

- Runtime area shows RX 9070 XT or a useful warning.
- A fictional adult character can be saved.
- The saved character appears in the dropdown.
- A prompt recipe preview can be generated.
- Existing `Models/` files remain unchanged.

- [ ] **Step 5: Record verification**

Create `docs/superpowers/plans/2026-04-30-local-model-studio-v1-verification.md`:

```markdown
# Local Model Studio V1 Verification

Date: 2026-04-30

## Commands

- `python -m pytest -v`
- `npm test`
- `npm run build`
- `powershell -ExecutionPolicy Bypass -File .\tools\setup\check-system.ps1`

## Result

Record pass/fail output here after running the commands.
```

- [ ] **Step 6: Commit verification note**

```powershell
git add docs/superpowers/plans/2026-04-30-local-model-studio-v1-verification.md
git commit -m "test: record local studio verification"
```

## Self-Review

- Spec coverage: The plan implements local profiles, simple controls, runtime checks, prompt recipes, ComfyUI API submission, workflow templates, setup scripts, and launch flow. Output gallery, LoRA training, image reference slots, and video remain future work after the base local loop is proven.
- Placeholder scan: The plan contains no placeholder work items or unspecified test steps.
- Type consistency: Backend schema names and frontend TypeScript names match the planned API payloads. Python modules referenced by tests are created in earlier steps.
