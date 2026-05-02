from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from local_model_studio.dataset_prepper import prepare_dataset_crops
from local_model_studio.schemas import DatasetPrepRequest
from local_model_studio.training_config import build_training_config
from local_model_studio.training_schemas import GlobalLearningRequest, GlobalLearningResponse, TrainingConfigRequest
from local_model_studio.training_store import TrainingStore


def create_global_learning_job(request: GlobalLearningRequest, training: TrainingStore) -> GlobalLearningResponse:
    prep = prepare_dataset_crops(
        DatasetPrepRequest(
            source_folder=request.source_folder,
            output_folder=None,
            target=request.target,
            recursive=request.recursive,
            scan_mode=request.scan_mode,
        )
    )
    if prep.cropped_count <= 0:
        raise ValueError("Local learning did not create any usable crops. Review the source folder or choose another crop target.")

    pack_slug = _slug(request.pack_name)
    version = _next_global_pack_version(training, pack_slug)
    lora_name = f"{pack_slug}_v{version:03d}"
    preset = _preset_settings(request.preset)
    config = build_training_config(
        TrainingConfigRequest(
            dataset_id=f"{pack_slug}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            dataset_path=prep.output_folder,
            output_dir=request.output_dir,
            base_model_path=request.base_model_path,
            lora_name=lora_name,
            dataset_type=request.dataset_type,
            global_pack=True,
            resolution=preset["resolution"],
            repeats=preset["repeats"],
            batch_size=1,
            max_train_steps=preset["max_train_steps"],
            learning_rate=preset["learning_rate"],
            network_dim=preset["network_dim"],
            network_alpha=preset["network_alpha"],
        ),
        prep.cropped_count,
    )
    saved = training.save(config)
    warnings = [
        *prep.warnings,
        "Global learning job is ready. Run the trainer with this config, then register the completed LoRA to activate it.",
    ]
    return GlobalLearningResponse(prep=prep, training_job=saved, version=version, warnings=warnings)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "global_body_pack"


def _next_global_pack_version(training: TrainingStore, pack_slug: str) -> int:
    versions: list[int] = []
    prefix = f"{pack_slug}_v"
    for job in training.list():
        if not job.global_pack or not job.lora_name.startswith(prefix):
            continue
        suffix = job.lora_name.removeprefix(prefix)
        try:
            versions.append(int(suffix))
        except ValueError:
            continue
    return max(versions, default=0) + 1


def _preset_settings(preset: str) -> dict[str, int | float]:
    if preset == "fast":
        return {"resolution": 640, "max_train_steps": 450, "learning_rate": 0.0001, "network_dim": 16, "network_alpha": 8, "repeats": 4}
    if preset == "high_quality":
        return {"resolution": 896, "max_train_steps": 1200, "learning_rate": 0.00002, "network_dim": 32, "network_alpha": 16, "repeats": 8}
    return {"resolution": 768, "max_train_steps": 900, "learning_rate": 0.00003, "network_dim": 32, "network_alpha": 16, "repeats": 6}
