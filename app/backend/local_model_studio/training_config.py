from __future__ import annotations

from pathlib import Path

from local_model_studio.training_schemas import (
    TrainerStatus,
    TrainingConfigRequest,
    TrainingJobConfig,
)


def build_training_config(
    request: TrainingConfigRequest, accepted_image_count: int
) -> TrainingJobConfig:
    if accepted_image_count <= 0:
        raise ValueError("accepted images must be greater than zero.")

    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"dataset directory must be a directory: {dataset_path}")

    if not request.base_model_path.strip():
        raise ValueError("base_model_path must be non-empty.")

    return TrainingJobConfig(
        dataset_id=request.dataset_id,
        dataset_path=str(dataset_path),
        output_dir=request.output_dir,
        base_model_path=request.base_model_path,
        lora_name=request.lora_name,
        resolution=request.resolution,
        repeats=request.repeats,
        batch_size=request.batch_size,
        max_train_steps=request.max_train_steps,
        learning_rate=request.learning_rate,
        network_dim=request.network_dim,
        network_alpha=request.network_alpha,
        accepted_image_count=accepted_image_count,
    )


def build_trainer_command(trainer_entrypoint: str | Path, config_path: str | Path) -> list[str]:
    entrypoint = Path(trainer_entrypoint)
    resolved_config_path = Path(config_path)

    if not entrypoint.exists():
        raise FileNotFoundError(f"Trainer entrypoint does not exist: {entrypoint}")
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Training config does not exist: {resolved_config_path}")

    if entrypoint.suffix.casefold() == ".ps1":
        return [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(entrypoint),
            str(resolved_config_path),
        ]

    return [str(entrypoint), str(resolved_config_path)]


def check_trainer_status(trainer_entrypoint: str | Path, config_path: str | Path) -> TrainerStatus:
    entrypoint = Path(trainer_entrypoint)
    resolved_config_path = Path(config_path)
    entrypoint_exists = entrypoint.exists()
    config_exists = resolved_config_path.exists()
    warnings: list[str] = []

    if not entrypoint_exists:
        warnings.append(f"Trainer entrypoint is missing: {entrypoint}")
    if not config_exists:
        warnings.append(f"Training config is missing: {resolved_config_path}")

    return TrainerStatus(
        trainer_entrypoint=str(entrypoint),
        config_path=str(resolved_config_path),
        trainer_entrypoint_exists=entrypoint_exists,
        config_path_exists=config_exists,
        warnings=warnings,
    )
