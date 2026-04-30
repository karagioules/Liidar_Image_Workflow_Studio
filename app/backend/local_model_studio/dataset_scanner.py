from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from .caption_builder import build_caption
from .face_safety import FaceDetector, OpenCvFaceDetector
from .paths import default_workspace_root
from .training_schemas import DatasetScanReport, DatasetScanRequest, ScannedImage


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GENERIC_DATASET_TYPES = {"body_part", "body_shape", "pose", "style"}


def scan_dataset(
    request: DatasetScanRequest,
    *,
    output_root: Path | None = None,
    detector: FaceDetector | None = None,
) -> DatasetScanReport:
    _validate_face_policy(request)

    if request.dataset_type == "fictional_face_identity":
        if not request.character_id:
            raise ValueError("fictional_face_identity datasets require character_id")
        if request.source_rights is None:
            raise ValueError("fictional_face_identity datasets require source_rights")

    source_folder = request.source_folder
    if not source_folder.is_dir():
        raise ValueError(f"source_folder does not exist or is not a directory: {source_folder}")

    dataset_id = uuid4().hex
    root = output_root or default_workspace_root() / "datasets"
    dataset_dir = root / dataset_id
    accepted_dir = dataset_dir / "accepted"
    rejected_dir = dataset_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    active_detector = detector or OpenCvFaceDetector()
    seen_hashes: set[str] = set()
    accepted: list[ScannedImage] = []
    rejected: list[ScannedImage] = []
    warnings: list[str] = []
    duplicate_count = 0
    ignored_count = 0

    for source_path in sorted(source_folder.iterdir()):
        if not source_path.is_file():
            ignored_count += 1
            continue
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            ignored_count += 1
            continue

        image_hash = _hash_validated_image(source_path)
        if image_hash is None:
            rejected.append(
                _reject(
                    source_path,
                    rejected_dir,
                    sha256="",
                    reason="invalid image",
                    caption="",
                )
            )
            continue

        if image_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(image_hash)

        face_result = active_detector.scan(source_path)
        if face_result.warning and face_result.warning not in warnings:
            warnings.append(face_result.warning)

        if not face_result.detector_available and request.dataset_type != "body_part":
            warning = face_result.warning or "face detector unavailable"
            raise ValueError(f"face detector unavailable; refusing to scan {request.dataset_type}: {warning}")

        caption = build_caption(request)
        rejection_reason = _face_rejection_reason(request, face_result.face_count)
        if rejection_reason is not None:
            rejected.append(
                _reject(
                    source_path,
                    rejected_dir,
                    sha256=image_hash,
                    reason=rejection_reason,
                    caption=caption,
                    character_id=request.character_id,
                    source_rights=request.source_rights,
                )
            )
            continue

        stored_path = accepted_dir / f"{image_hash}{source_path.suffix.lower()}"
        shutil.copy2(source_path, stored_path)
        stored_path.with_suffix(".txt").write_text(caption, encoding="utf-8")
        accepted.append(
            ScannedImage(
                source_path=str(source_path),
                stored_path=str(stored_path),
                sha256=image_hash,
                accepted=True,
                reason="accepted",
                caption=caption,
                character_id=request.character_id,
                source_rights=request.source_rights,
            )
        )

    return DatasetScanReport(
        dataset_id=dataset_id,
        name=request.name,
        dataset_type=request.dataset_type,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        duplicate_count=duplicate_count,
        ignored_count=ignored_count,
        accepted=accepted,
        rejected=rejected,
        warnings=warnings,
    )


def _hash_validated_image(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as image:
            image.load()
            normalized = image.convert("RGBA")
            digest = hashlib.sha256()
            digest.update(normalized.size[0].to_bytes(4, "big"))
            digest.update(normalized.size[1].to_bytes(4, "big"))
            digest.update(normalized.tobytes())
            return digest.hexdigest()
    except (OSError, UnidentifiedImageError):
        return None


def _face_rejection_reason(request: DatasetScanRequest, face_count: int) -> str | None:
    if request.face_policy == "redact_faces":
        return "redaction not implemented safely yet"
    if request.dataset_type == "fictional_face_identity":
        return None
    if request.dataset_type in GENERIC_DATASET_TYPES and face_count > 0:
        return f"detected {face_count} face(s); generic datasets must not preserve faces"
    return None


def _validate_face_policy(request: DatasetScanRequest) -> None:
    if request.face_policy == "body_part_crops_only" and request.dataset_type != "body_part":
        raise ValueError("face_policy body_part_crops_only is only valid for body_part datasets")


def _reject(
    source_path: Path,
    rejected_dir: Path,
    *,
    sha256: str,
    reason: str,
    caption: str,
    character_id: str | None = None,
    source_rights: str | None = None,
) -> ScannedImage:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{sha256 or uuid4().hex}{source_path.suffix.lower()}"
    stored_path = rejected_dir / stored_name
    shutil.copy2(source_path, stored_path)
    return ScannedImage(
        source_path=str(source_path),
        stored_path=str(stored_path),
        sha256=sha256,
        accepted=False,
        reason=reason,
        caption=caption,
        character_id=character_id,
        source_rights=source_rights,  # type: ignore[arg-type]
    )
