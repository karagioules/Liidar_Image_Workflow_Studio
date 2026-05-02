from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path
from threading import Event
from collections.abc import Callable, Iterable

import httpx
from PIL import Image, UnidentifiedImageError

from local_model_studio.file_browser import IMAGE_SUFFIXES
from local_model_studio.local_crop_detector import LocalCropDetection, LocalCropDetectorUnavailable, local_crop_detector
from local_model_studio.schemas import DatasetPrepImage, DatasetPrepRequest, DatasetPrepResponse


CropBox = tuple[int, int, int, int]
FaceBox = tuple[int, int, int, int]
NormalizedBox = tuple[float, float, float, float]
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
    ai_attempted_count = 0
    ai_guided_count = 0
    ai_failed_count = 0
    face_guided_count = 0
    fallback_count = 0
    cancelled = False
    scan_mode = request.effective_scan_mode()

    _report_progress(
        progress_callback,
        total_count=len(image_paths),
        processed_count=0,
        cropped_count=0,
        skipped_count=0,
        ai_attempted_count=0,
        ai_guided_count=0,
        ai_failed_count=0,
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
            ai_attempted_count=ai_attempted_count,
            ai_guided_count=ai_guided_count,
            ai_failed_count=ai_failed_count,
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
                ai_attempted = scan_mode in {"local", "claude"} and (scan_mode == "local" or index <= request.ai_max_images)
                if ai_attempted:
                    ai_attempted_count += 1
                    if scan_mode == "local":
                        ai_crop_box, ai_warning = _local_ai_crop_box(image_path, image.size, request.target)
                    else:
                        if not anthropic_api_key:
                            raise ValueError("Claude API key is required when Claude scanning is enabled.")
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
                        method = "local_ai" if scan_mode == "local" else "ai_guided"
                if crop_box is None:
                    if ai_attempted:
                        ai_failed_count += 1
                        images.append(
                            DatasetPrepImage(
                                source_path=str(image_path),
                                width=image.width,
                                height=image.height,
                                face_count=0,
                                accepted=False,
                                reason=(
                                    "Local AI scan did not find a usable tight target crop"
                                    if scan_mode == "local"
                                    else "Claude AI attempted but did not return a usable tight target crop"
                                ),
                                method="local_ai_failed" if scan_mode == "local" else "ai_failed",
                                crop_box=None,
                                output_path=None,
                            )
                        )
                        continue
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
                if method in {"ai_guided", "local_ai"}:
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
            ai_attempted_count=ai_attempted_count,
            ai_guided_count=ai_guided_count,
            ai_failed_count=ai_failed_count,
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
        ai_attempted_count=ai_attempted_count,
        ai_guided_count=ai_guided_count,
        ai_failed_count=ai_failed_count,
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
        ai_attempted_count=response.ai_attempted_count,
        ai_guided_count=response.ai_guided_count,
        ai_failed_count=response.ai_failed_count,
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


def _local_ai_crop_box(image_path: Path, size: tuple[int, int], target: str) -> tuple[CropBox | None, str | None]:
    try:
        detections = local_crop_detector().detect(image_path)
    except LocalCropDetectorUnavailable as exc:
        return None, f"Local AI crop scanner is unavailable; skipped matching crop. {exc}"
    return _crop_box_from_local_detections(detections, size, target)


def _crop_box_from_local_detections(
    detections: list[LocalCropDetection],
    size: tuple[int, int],
    target: str,
) -> tuple[CropBox | None, str | None]:
    width, height = size
    if width <= 0 or height <= 0:
        return None, "Local AI crop scanner received an invalid image size."

    face_box = _largest_normalized_box(
        _detection_box_to_normalized(detection, width, height)
        for detection in detections
        if detection.label in {"FACE_FEMALE", "FACE_MALE"} and detection.score >= 0.22
    )
    if target == "chest_detail":
        exposed_breasts = _valid_breast_boxes(detections, width, height, exposed_only=True)
        breast_boxes = exposed_breasts or _valid_breast_boxes(detections, width, height, exposed_only=False)
        if not breast_boxes:
            return None, "Local AI scan did not detect a usable breast region in at least one image; that image was skipped."
        crop_box = _breast_region_crop(breast_boxes, face_box)
        if not _is_tight_chest_region(crop_box):
            return None, "Local AI scan found breasts but the crop was too broad or portrait-like; that image was skipped."
        return _normalized_box_to_pixels(crop_box, width, height), None

    torso_boxes = [
        box
        for box in (_detection_box_to_normalized(detection, width, height) for detection in detections)
        if box is not None
    ]
    if not torso_boxes:
        return None, "Local AI scan did not find a usable body region in at least one image; that image was skipped."
    crop_box = _avoid_face_overlap(_union_boxes(torso_boxes), face_box)
    return _normalized_box_to_pixels(crop_box, width, height), None


def _valid_breast_boxes(
    detections: list[LocalCropDetection],
    width: int,
    height: int,
    *,
    exposed_only: bool,
) -> list[NormalizedBox]:
    boxes: list[NormalizedBox] = []
    for detection in detections:
        if detection.score < 0.24:
            continue
        if exposed_only and detection.label != "FEMALE_BREAST_EXPOSED":
            continue
        if not exposed_only and detection.label not in {"FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED"}:
            continue
        box = _detection_box_to_normalized(detection, width, height)
        if box is None:
            continue
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        if box_width * box_height < 0.003:
            continue
        boxes.append(box)
    return boxes


def _detection_box_to_normalized(detection: LocalCropDetection, width: int, height: int) -> NormalizedBox | None:
    x, y, box_width, box_height = detection.box
    return _normalized_box([x / width, y / height, (x + box_width) / width, (y + box_height) / height])


def _largest_normalized_box(boxes: Iterable[NormalizedBox | None]) -> NormalizedBox | None:
    valid_boxes = [box for box in boxes if box is not None]
    if not valid_boxes:
        return None
    return max(valid_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))


def _union_boxes(boxes: list[NormalizedBox]) -> NormalizedBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _claude_crop_box(
    image: Image.Image,
    target: str,
    *,
    anthropic_api_key: str,
    model: str,
) -> tuple[CropBox | None, str | None]:
    encoded = _encode_image_for_ai(image)
    try:
        response = _post_claude_crop_request(
            encoded,
            target,
            anthropic_api_key=anthropic_api_key,
            model=model,
            structured=False,
        )
        if response.status_code >= 400:
            return None, f"Claude AI crop failed: {_anthropic_error_message(response)}"
        data = response.json()
        try:
            parsed = _parse_claude_crop_response(data)
        except (ValueError, json.JSONDecodeError):
            retry_response = _post_claude_crop_request(
                encoded,
                target,
                anthropic_api_key=anthropic_api_key,
                model=model,
                structured=True,
            )
            if retry_response.status_code >= 400:
                return None, f"Claude AI crop retry failed: {_anthropic_error_message(retry_response)}"
            parsed = _parse_claude_crop_response(retry_response.json())
        if not parsed.get("visible_target"):
            return None, "Claude did not find a tight target crop in at least one image; that image was skipped."
        crop_box = _crop_box_from_ai_response(parsed, image.width, image.height, target)
        if crop_box is None:
            return None, "Claude returned a broad or unusable crop for at least one image; that image was skipped."
        return crop_box, None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, f"Claude AI crop failed for at least one image; that image was skipped. {exc}"


def _post_claude_crop_request(
    encoded_image: str,
    target: str,
    *,
    anthropic_api_key: str,
    model: str,
    structured: bool,
) -> httpx.Response:
    target_label = _target_instruction(target)
    prompt = (
        "Return compact JSON only. Coordinates are normalized [left,top,right,bottom]. "
        "For chest_detail, visible_target is true only when visible nipples/areolas and full visible breast outlines can be boxed. "
        "Return breast_boxes for full visible breast shapes and nipple_boxes for visible nipples/areolas. "
        "Exclude face/head/mouth/eyes; no portrait/full-torso crop. "
        "If nipples/areolas are hidden or cut off, visible_target=false. "
        f"Target: {target_label}. "
        'JSON: {"visible_target":true,"breast_boxes":[[0.2,0.3,0.45,0.62],[0.52,0.3,0.78,0.62]],"nipple_boxes":[[0.32,0.46,0.36,0.5],[0.64,0.46,0.68,0.5]],"face_box":null}'
    )
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": 140 if structured else 80,
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
                            "data": encoded_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if structured:
        payload["tools"] = [_claude_crop_tool()]
        payload["tool_choice"] = {"type": "tool", "name": "return_crop"}
    return httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=45,
    )


def _claude_crop_tool() -> dict:
    return {
        "name": "return_crop",
        "description": "Return tight chest crop coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "visible_target": {"type": "boolean"},
                "breast_boxes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                },
                "nipple_boxes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                },
                "face_box": {
                    "type": ["array", "null"],
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
            },
            "required": ["visible_target", "breast_boxes", "nipple_boxes", "face_box"],
            "additionalProperties": False,
        },
    }


def _anthropic_error_message(response: httpx.Response) -> str:
    detail = response.text.strip()
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                error_type = error.get("type")
                if message:
                    detail = f"{error_type}: {message}" if error_type else str(message)
            elif body.get("detail"):
                detail = str(body["detail"])
    except json.JSONDecodeError:
        pass
    return f"{response.status_code} {response.reason_phrase}" + (f" - {detail}" if detail else "")


def _target_instruction(target: str) -> str:
    if target == "chest_detail":
        return "tight breast-only crop; include the full visible breast shape and visible nipples/areolas, exclude face/head/mouth/eyes and avoid full torso"
    if target == "upper_torso":
        return "upper torso region; shoulders/chest/waist context, avoid full face"
    return "full body context"


def _encode_image_for_ai(image: Image.Image) -> str:
    resized = image.copy()
    resized.thumbnail((384, 384))
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=72)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_ai_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object returned")
    return json.loads(match.group(0))


def _parse_claude_crop_response(data: dict) -> dict:
    content = data.get("content", [])
    if not isinstance(content, list):
        raise ValueError("Claude returned an invalid content block")
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "tool_use" and part.get("name") == "return_crop":
            tool_input = part.get("input")
            if isinstance(tool_input, dict):
                return tool_input
            if isinstance(tool_input, str):
                return _parse_ai_json(tool_input)
            raise ValueError("Claude returned an invalid tool crop payload")
    text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    if text.strip():
        return _parse_ai_json(text)
    stop_reason = data.get("stop_reason")
    if stop_reason:
        raise ValueError(f"Claude returned no crop coordinates (stop_reason={stop_reason})")
    raise ValueError("Claude returned no crop coordinates")


def _crop_box_from_ai_response(parsed: dict, width: int, height: int, target: str) -> CropBox | None:
    face_box = _normalized_box(parsed.get("face_box"))
    breast_boxes = _normalized_boxes(parsed.get("breast_boxes"))
    nipple_boxes = _normalized_boxes(parsed.get("nipple_boxes"))
    target_box = _normalized_box(parsed.get("target_box")) or _normalized_box(parsed.get("crop_box"))
    crop_box = _normalized_box(parsed.get("crop_box")) or target_box
    if target == "chest_detail":
        if not breast_boxes:
            return None
        crop_basis = breast_boxes + nipple_boxes
        crop_box = _breast_region_crop(crop_basis, face_box)
        if not _is_tight_chest_region(crop_box):
            return None
        return _normalized_box_to_pixels(crop_box, width, height)

    if target_box is None or crop_box is None:
        return None

    if target == "chest_detail":
        crop_box = _tight_chest_crop(target_box, face_box)
    elif target == "upper_torso":
        crop_box = _avoid_face_overlap(crop_box, face_box)

    return _normalized_box_to_pixels(crop_box, width, height)


def _normalized_box(value: object) -> NormalizedBox | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return (
        max(0.0, min(left, 1.0)),
        max(0.0, min(top, 1.0)),
        max(0.0, min(right, 1.0)),
        max(0.0, min(bottom, 1.0)),
    )


def _normalized_boxes(value: object) -> list[NormalizedBox]:
    if not isinstance(value, list):
        return []
    boxes: list[NormalizedBox] = []
    for item in value:
        box = _normalized_box(item)
        if box is not None:
            boxes.append(box)
    return boxes


def _breast_region_crop(breast_boxes: list[NormalizedBox], face_box: NormalizedBox | None) -> NormalizedBox:
    left = min(box[0] for box in breast_boxes)
    top = min(box[1] for box in breast_boxes)
    right = max(box[2] for box in breast_boxes)
    bottom = max(box[3] for box in breast_boxes)
    width = right - left
    height = bottom - top
    pad_x = max(width * 0.16, 0.025)
    pad_y = max(height * 0.14, 0.018)
    crop = (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)
    return _avoid_face_overlap(crop, face_box)


def _tight_chest_crop(target_box: NormalizedBox, face_box: NormalizedBox | None) -> NormalizedBox:
    left, top, right, bottom = target_box
    width = right - left
    height = bottom - top
    pad_x = max(width * 0.12, 0.025)
    pad_y = max(height * 0.12, 0.018)
    crop = (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)
    crop = _avoid_face_overlap(crop, face_box)

    left, top, right, bottom = crop
    if bottom - top > 0.46:
        center_y = (top + bottom) / 2
        half_height = 0.23
        top = center_y - half_height
        bottom = center_y + half_height
    return _clamp_normalized_box((left, top, right, bottom))


def _avoid_face_overlap(crop_box: NormalizedBox, face_box: NormalizedBox | None) -> NormalizedBox:
    if face_box is None:
        return _clamp_normalized_box(crop_box)
    left, top, right, bottom = crop_box
    face_left, face_top, face_right, face_bottom = face_box
    horizontal_overlap = min(right, face_right) - max(left, face_left)
    vertical_overlap = min(bottom, face_bottom) - max(top, face_top)
    if horizontal_overlap > 0 and vertical_overlap > 0:
        top = max(top, face_bottom + 0.015)
    return _clamp_normalized_box((left, top, right, bottom))


def _clamp_normalized_box(box: NormalizedBox) -> NormalizedBox:
    left, top, right, bottom = box
    left = max(0.0, min(left, 0.99))
    top = max(0.0, min(top, 0.99))
    right = max(left + 0.01, min(right, 1.0))
    bottom = max(top + 0.01, min(bottom, 1.0))
    return left, top, right, bottom


def _is_tight_chest_region(box: NormalizedBox) -> bool:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return False
    if height > 0.72:
        return False
    if height / width > 2.0:
        return False
    if width > 0.92 and height > 0.58:
        return False
    return True


def _normalized_box_to_pixels(box: NormalizedBox, width: int, height: int) -> CropBox:
    left, top, right, bottom = box
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
    if method == "local_ai":
        return f"{label} crop from local AI scan"
    if method == "ai_guided":
        return f"{label} crop from Claude AI scan"
    if method == "face_guided":
        return f"{label} crop below detected face"
    return f"{label} center fallback crop"

