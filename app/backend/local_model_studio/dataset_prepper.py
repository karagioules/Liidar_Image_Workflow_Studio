from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path
from threading import Event
from collections.abc import Callable

import httpx
from PIL import Image, UnidentifiedImageError

from local_model_studio.file_browser import IMAGE_SUFFIXES
from local_model_studio.schemas import DatasetPrepImage, DatasetPrepRequest, DatasetPrepResponse


CropBox = tuple[int, int, int, int]
FaceBox = tuple[int, int, int, int]
ProgressCallback = Callable[[dict[str, object]], None]


def prepare_dataset_crops(
    request: DatasetPrepRequest,
    *,
    anthropic_api_key: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> DatasetPrepResponse:
    source_folder = request.source_folder
    if not source_folder.is_dir():
        raise ValueError(f"source_folder does not exist or is not a directory: {source_folder}")

    output_folder = request.output_folder or _default_output_folder(request.target)
    output_folder.mkdir(parents=True, exist_ok=True)

    paths = source_folder.rglob("*") if request.recursive else source_folder.iterdir()
    image_paths = sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

    images: list[DatasetPrepImage] = []
    warnings: list[str] = []
    ai_guided_count = 0
    face_guided_count = 0
    fallback_count = 0
    cancelled = False

    _report_progress(
        progress_callback,
        total_count=len(image_paths),
        processed_count=0,
        cropped_count=0,
        skipped_count=0,
        ai_guided_count=0,
        face_guided_count=0,
        fallback_count=0,
        active_file=None,
        output_folder=str(output_folder),
    )

    for index, image_path in enumerate(image_paths, start=1):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break
        _report_progress(
            progress_callback,
            total_count=len(image_paths),
            processed_count=len(images),
            cropped_count=sum(1 for image in images if image.accepted),
            skipped_count=sum(1 for image in images if not image.accepted),
            ai_guided_count=ai_guided_count,
            face_guided_count=face_guided_count,
            fallback_count=fallback_count,
            active_file=str(image_path),
            output_folder=str(output_folder),
        )
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                faces: list[FaceBox] = []
                crop_box: CropBox | None = None
                method = "skipped"
                if request.use_ai and index <= request.ai_max_images:
                    if not anthropic_api_key:
                        raise ValueError("Claude API key is required when AI scanning is enabled.")
                    ai_crop_box, ai_warning = _claude_crop_box(
                        image,
                        request.target,
                        anthropic_api_key=anthropic_api_key,
                        model=request.ai_model,
                    )
                    if ai_warning and ai_warning not in warnings:
                        warnings.append(ai_warning)
                    if ai_crop_box is not None:
                        crop_box = ai_crop_box
                        method = "ai_guided"
                if crop_box is None:
                    faces, face_warning = _detect_faces(image_path)
                    if face_warning and face_warning not in warnings:
                        warnings.append(face_warning)
                    crop_box, method = _crop_box_for(image.size, faces, request.target)

                if crop_box is None:
                    images.append(
                        DatasetPrepImage(
                            source_path=str(image_path),
                            width=image.width,
                            height=image.height,
                            face_count=len(faces),
                            accepted=False,
                            reason="could not determine a useful crop",
                            method="skipped",
                            crop_box=None,
                            output_path=None,
                        )
                    )
                    continue

                cropped = image.crop(crop_box)
                output_path = output_folder / f"{image_path.stem}_{request.target}_{index:04d}.jpg"
                cropped.save(output_path, "JPEG", quality=95)
                if method == "ai_guided":
                    ai_guided_count += 1
                elif method == "face_guided":
                    face_guided_count += 1
                else:
                    fallback_count += 1
                images.append(
                    DatasetPrepImage(
                        source_path=str(image_path),
                        output_path=str(output_path),
                        width=image.width,
                        height=image.height,
                        face_count=len(faces),
                        accepted=True,
                        reason=_reason_for(method, request.target),
                        method=method,
                        crop_box=list(crop_box),
                    )
                )
        except (OSError, UnidentifiedImageError):
            images.append(
                DatasetPrepImage(
                    source_path=str(image_path),
                    width=0,
                    height=0,
                    face_count=0,
                    accepted=False,
                    reason="invalid or unreadable image",
                    method="skipped",
                    crop_box=None,
                    output_path=None,
                )
            )
        _report_progress(
            progress_callback,
            total_count=len(image_paths),
            processed_count=len(images),
            cropped_count=sum(1 for image in images if image.accepted),
            skipped_count=sum(1 for image in images if not image.accepted),
            ai_guided_count=ai_guided_count,
            face_guided_count=face_guided_count,
            fallback_count=fallback_count,
            active_file=str(image_path),
            output_folder=str(output_folder),
        )

    accepted_count = sum(1 for image in images if image.accepted)
    skipped_count = len(images) - accepted_count
    if cancelled:
        warnings.append("Dataset prep was cancelled. Partial crops remain in the output folder.")
    if fallback_count:
        warnings.append(
            f"{fallback_count} crop(s) used center fallback because a face/body anchor was not detected; review those outputs manually."
        )

    response = DatasetPrepResponse(
        output_folder=str(output_folder),
        processed_count=len(images),
        cropped_count=accepted_count,
        skipped_count=skipped_count,
        ai_guided_count=ai_guided_count,
        face_guided_count=face_guided_count,
        fallback_count=fallback_count,
        warnings=warnings,
        images=images,
    )
    _report_progress(
        progress_callback,
        total_count=len(image_paths),
        processed_count=response.processed_count,
        cropped_count=response.cropped_count,
        skipped_count=response.skipped_count,
        ai_guided_count=response.ai_guided_count,
        face_guided_count=response.face_guided_count,
        fallback_count=response.fallback_count,
        active_file=None,
        output_folder=response.output_folder,
    )
    return response


def _report_progress(callback: ProgressCallback | None, **payload: object) -> None:
    if callback:
        callback(payload)


def _default_output_folder(target: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    desktop = Path.home() / "Desktop"
    root = desktop if desktop.exists() else Path.home()
    return root / "Liidar_Dataset_Crops" / f"{stamp}-{target}"


def _detect_faces(image_path: Path) -> tuple[list[FaceBox], str | None]:
    return [], "Local face detector disabled for stable background prep; using center fallback crops."


def _claude_crop_box(
    image: Image.Image,
    target: str,
    *,
    anthropic_api_key: str,
    model: str,
) -> tuple[CropBox | None, str | None]:
    encoded = _encode_image_for_ai(image)
    prompt = (
        "Return compact JSON only. This is adult-only dataset preparation. "
        "Find the best crop for the requested target without including the face when possible. "
        "Use normalized coordinates from 0 to 1 as [left, top, right, bottom]. "
        "If the target is not visible, set visible_target false. "
        f"Target: {target.replace('_', ' ')}. "
        'Schema: {"visible_target":true,"crop_box":[0.1,0.2,0.8,0.7],"confidence":0.0}'
    )
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 160,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        parsed = _parse_ai_json(text)
        if not parsed.get("visible_target"):
            return None, "Claude did not find the requested target in at least one image; local crop fallback was used."
        box = parsed.get("crop_box")
        if not isinstance(box, list) or len(box) != 4:
            return None, "Claude returned an unusable crop box for at least one image; local crop fallback was used."
        return _normalized_box_to_pixels(box, image.width, image.height), None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, f"Claude AI crop failed for at least one image; local crop fallback was used. {exc}"


def _encode_image_for_ai(image: Image.Image) -> str:
    resized = image.copy()
    resized.thumbnail((768, 768))
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_ai_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object returned")
    return json.loads(match.group(0))


def _normalized_box_to_pixels(box: list[object], width: int, height: int) -> CropBox:
    left, top, right, bottom = [float(value) for value in box]
    return _clamp_box((int(left * width), int(top * height), int(right * width), int(bottom * height)), width, height)


def _crop_box_for(size: tuple[int, int], faces: list[FaceBox], target: str) -> tuple[CropBox | None, str]:
    width, height = size
    if width <= 0 or height <= 0:
        return None, "skipped"

    face = _largest_face(faces)
    if face is not None:
        return _face_guided_crop(width, height, face, target), "face_guided"
    return _fallback_crop(width, height, target), "fallback"


def _largest_face(faces: list[FaceBox]) -> FaceBox | None:
    if not faces:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def _face_guided_crop(width: int, height: int, face: FaceBox, target: str) -> CropBox:
    x, y, w, h = face
    center_x = x + w / 2

    if target == "full_body_context":
        return _clamp_box((int(center_x - w * 3.2), max(0, y - int(h * 0.2)), int(center_x + w * 3.2), height), width, height)

    if target == "upper_torso":
        return _clamp_box(
            (int(center_x - w * 2.7), int(y + h * 0.75), int(center_x + w * 2.7), int(y + h * 5.3)),
            width,
            height,
        )

    return _clamp_box(
        (int(center_x - w * 2.1), int(y + h * 1.15), int(center_x + w * 2.1), int(y + h * 4.2)),
        width,
        height,
    )


def _fallback_crop(width: int, height: int, target: str) -> CropBox:
    if target == "full_body_context":
        return (0, 0, width, height)
    if target == "upper_torso":
        return _clamp_box((int(width * 0.12), int(height * 0.12), int(width * 0.88), int(height * 0.72)), width, height)
    return _clamp_box((int(width * 0.16), int(height * 0.18), int(width * 0.84), int(height * 0.62)), width, height)


def _clamp_box(box: CropBox, width: int, height: int) -> CropBox:
    left, top, right, bottom = box
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _reason_for(method: str, target: str) -> str:
    label = target.replace("_", " ")
    if method == "ai_guided":
        return f"{label} crop from Claude AI scan"
    if method == "face_guided":
        return f"{label} crop below detected face"
    return f"{label} center fallback crop"
