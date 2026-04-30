# Training Module V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local dataset ingestion and trainer preparation for generic adult body/style LoRAs while preventing face cloning.

**Architecture:** Add backend modules for dataset scanning, face-safety policy, caption generation, training config generation, and external trainer command execution. Store sanitized datasets under `datasets/` and metadata under `config/training/`. Frontend training controls are added after the backend API exists.

**Tech Stack:** Python 3.12+, Pydantic, Pillow, OpenCV optional face detection, pytest, FastAPI routes added in the main app, external trainer adapter for ROCm-compatible LoRA training tools.

---

## File Structure

- Create `app/backend/local_model_studio/training_schemas.py`: dataset, scan, caption, trainer config, and job status models.
- Create `app/backend/local_model_studio/dataset_scanner.py`: recursive folder scan, hashing, extension filtering, safe copy layout.
- Create `app/backend/local_model_studio/face_safety.py`: face detection abstraction with fail-closed behavior when detector is unavailable.
- Create `app/backend/local_model_studio/caption_builder.py`: generic body/style captions with no identity names.
- Create `app/backend/local_model_studio/training_config.py`: SDXL LoRA config and command builder.
- Create `app/backend/local_model_studio/training_store.py`: JSON metadata persistence under `config/training/`.
- Create `app/backend/tests/test_dataset_scanner.py`.
- Create `app/backend/tests/test_face_safety.py`.
- Create `app/backend/tests/test_training_config.py`.
- Later modify `app/backend/local_model_studio/main.py` to expose training routes.
- Later modify frontend files to add Training UI.
- Modify `.gitignore` to ignore `datasets/`, `training-runs/`, and generated LoRA outputs.

## Training Task A: Dataset Ingestion and Face-Safety Scan

**Files:**
- Create: `app/backend/local_model_studio/training_schemas.py`
- Create: `app/backend/local_model_studio/face_safety.py`
- Create: `app/backend/local_model_studio/dataset_scanner.py`
- Create: `app/backend/local_model_studio/caption_builder.py`
- Test: `app/backend/tests/test_dataset_scanner.py`
- Test: `app/backend/tests/test_face_safety.py`
- Modify: `app/backend/pyproject.toml`
- Modify: `.gitignore`

Required behavior:

- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Unsupported files are ignored.
- Images are hashed with SHA-256 and duplicates are counted.
- Accepted images are copied into `datasets/<dataset_id>/accepted/`.
- Rejected images are copied into `datasets/<dataset_id>/rejected/` only when a rejection copy is needed for review.
- Face policy defaults to `reject_faces`.
- If the detector is unavailable and the dataset type is not `body_part`, the scan fails closed with a clear warning.
- Captions are generic and identity-free.

Suggested tests:

```python
from pathlib import Path

from PIL import Image

from local_model_studio.dataset_scanner import scan_dataset
from local_model_studio.face_safety import FaceScanResult
from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_schemas import DatasetScanRequest


class NoFaceDetector:
    available = True

    def scan(self, path: Path) -> FaceScanResult:
        return FaceScanResult(face_count=0, detector_available=True)


class FaceDetector:
    available = True

    def scan(self, path: Path) -> FaceScanResult:
        return FaceScanResult(face_count=1, detector_available=True)


def write_image(path: Path) -> None:
    Image.new("RGB", (64, 64), color=(200, 160, 140)).save(path)


def test_scan_accepts_unique_face_free_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "a.jpg")
    write_image(source / "b.png")
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")

    report = scan_dataset(
        WorkspacePaths(tmp_path),
        DatasetScanRequest(name="body references", source_folder=str(source), dataset_type="body_part"),
        detector=NoFaceDetector(),
    )

    assert report.accepted_count == 2
    assert report.rejected_count == 0
    assert report.ignored_count == 1
    assert all("identity" not in item.caption.lower() for item in report.accepted)


def test_scan_counts_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "a.jpg")
    (source / "copy.jpg").write_bytes((source / "a.jpg").read_bytes())

    report = scan_dataset(
        WorkspacePaths(tmp_path),
        DatasetScanRequest(name="body references", source_folder=str(source), dataset_type="body_part"),
        detector=NoFaceDetector(),
    )

    assert report.accepted_count == 1
    assert report.duplicate_count == 1


def test_scan_rejects_detected_faces(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_image(source / "face.jpg")

    report = scan_dataset(
        WorkspacePaths(tmp_path),
        DatasetScanRequest(name="style refs", source_folder=str(source), dataset_type="style"),
        detector=FaceDetector(),
    )

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert "face" in report.rejected[0].reason.lower()
```

Implementation notes:

- Use Pillow for image validation and copy only files that Pillow can open.
- Define `DatasetType = Literal["body_part", "body_shape", "pose", "style"]`.
- Define `FacePolicy = Literal["reject_faces", "redact_faces", "body_part_crops_only"]`.
- V1 can implement `redact_faces` as rejection with a warning until redaction is implemented safely.
- Use deterministic dataset IDs from `uuid4().hex`.
- Write one `.txt` caption beside each accepted image.

Verification:

```powershell
cd app\backend
py -3.12 -m pip install -e .[test]
py -3.12 -m pytest tests/test_dataset_scanner.py tests/test_face_safety.py -v
py -3.12 -m pytest -v
```

Commit:

```powershell
git add .gitignore app/backend/pyproject.toml app/backend/local_model_studio app/backend/tests
git commit -m "feat: add face-safe dataset ingestion"
```

## Training Task B: Training Config and External Trainer Adapter

**Files:**
- Create: `app/backend/local_model_studio/training_config.py`
- Create: `app/backend/local_model_studio/training_store.py`
- Test: `app/backend/tests/test_training_config.py`

Required behavior:

- Generate SDXL LoRA training config metadata with:
  - dataset path.
  - output path.
  - base model path.
  - LoRA name.
  - resolution.
  - repeats.
  - batch size.
  - max train steps or epochs.
  - learning rate.
- Build a trainer command from a user-configured executable/script path and config path.
- Report missing trainer executable clearly.
- Do not start training if accepted image count is zero.
- Register completed LoRA paths into metadata for later character/profile use.

Suggested tests:

```python
from pathlib import Path

import pytest

from local_model_studio.training_config import build_training_config, build_trainer_command
from local_model_studio.training_schemas import TrainingConfigRequest


def test_build_training_config_contains_required_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "abc" / "accepted"
    dataset.mkdir(parents=True)
    request = TrainingConfigRequest(
        dataset_id="abc",
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "training-runs"),
        base_model_path="H:/models/sdxl.safetensors",
        lora_name="natural_body_shape",
    )

    config = build_training_config(request, accepted_image_count=12)

    assert config.dataset_path == str(dataset)
    assert config.output_dir.endswith("training-runs")
    assert config.lora_name == "natural_body_shape"
    assert config.resolution == 1024


def test_training_config_rejects_empty_dataset(tmp_path: Path) -> None:
    request = TrainingConfigRequest(
        dataset_id="abc",
        dataset_path=str(tmp_path / "empty"),
        output_dir=str(tmp_path / "training-runs"),
        base_model_path="H:/models/sdxl.safetensors",
        lora_name="empty",
    )

    with pytest.raises(ValueError, match="accepted images"):
        build_training_config(request, accepted_image_count=0)


def test_build_trainer_command_requires_existing_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        build_trainer_command(str(tmp_path / "missing.ps1"), config_path)
```

Verification:

```powershell
cd app\backend
py -3.12 -m pytest tests/test_training_config.py -v
py -3.12 -m pytest -v
```

Commit:

```powershell
git add app/backend/local_model_studio/training_config.py app/backend/local_model_studio/training_store.py app/backend/tests/test_training_config.py
git commit -m "feat: add local trainer adapter"
```

## API and UI Integration Notes

After Training Tasks A and B pass:

- Extend `main.py` with `/api/training/scan`, `/api/training/config`, `/api/training/status`, and `/api/training/start`.
- Extend the frontend with a Training page/section:
  - folder path input.
  - dataset type select.
  - face policy select.
  - scan button.
  - accepted/rejected/duplicate counts.
  - base model path and LoRA name inputs.
  - train button disabled until trainer status is ready.

## Self-Review

- Spec coverage: This plan implements the user's requirement to feed hundreds or thousands of body/style images as generic training knowledge while preventing face replication.
- Placeholder scan: The plan contains no placeholder work items.
- Scope check: The implementation is split into ingestion safety and trainer adapter tasks so the app can safely accept folders before attempting long-running training.
