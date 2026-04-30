from __future__ import annotations

from .training_schemas import DatasetScanRequest


GENERIC_LABELS = {
    "body_part": "adult, body reference, body_part",
    "body_shape": "adult, body reference, body_shape",
    "pose": "adult, body reference, pose",
    "style": "adult, style reference, style",
}


def build_caption(request: DatasetScanRequest) -> str:
    if request.dataset_type == "fictional_face_identity":
        return (
            "fictional adult character face reference, "
            f"{request.source_rights} source, character:{request.character_id}"
        )

    parts = [GENERIC_LABELS[request.dataset_type]]
    parts.extend(tag.strip() for tag in request.tags if tag.strip())
    return ", ".join(parts)
