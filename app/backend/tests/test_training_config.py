from __future__ import annotations

from pathlib import Path

import pytest

from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_config import (
    build_trainer_command,
    build_training_config,
    check_trainer_status,
)
from local_model_studio.training_schemas import TrainingConfigRequest
from local_model_studio.training_store import TrainingStore


def _request(
    tmp_path: Path,
    dataset: Path,
    *,
    base_model_path: str = "H:/models/sdxl.safetensors",
) -> TrainingConfigRequest:
    return TrainingConfigRequest(
        dataset_id="abc",
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "training-runs"),
        base_model_path=base_model_path,
        lora_name="natural_body_shape",
    )


def test_build_training_config_contains_required_paths_and_defaults(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "abc" / "accepted"
    dataset.mkdir(parents=True)
    request = _request(tmp_path, dataset)

    config = build_training_config(request, accepted_image_count=12)

    assert config.dataset_id == "abc"
    assert config.dataset_path == str(dataset)
    assert config.output_dir.endswith("training-runs")
    assert config.base_model_path == "H:/models/sdxl.safetensors"
    assert config.lora_name == "natural_body_shape"
    assert config.resolution == 1024
    assert config.repeats == 10
    assert config.batch_size == 1
    assert config.max_train_steps == 1200
    assert config.learning_rate == 1e-4
    assert config.network_dim == 32
    assert config.network_alpha == 16
    assert config.accepted_image_count == 12


def test_training_config_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "empty"
    dataset.mkdir()
    request = _request(tmp_path, dataset)

    with pytest.raises(ValueError, match="accepted images"):
        build_training_config(request, accepted_image_count=0)


def test_training_config_rejects_missing_dataset_path(tmp_path: Path) -> None:
    request = _request(tmp_path, tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="Dataset path"):
        build_training_config(request, accepted_image_count=3)


def test_training_config_rejects_blank_base_model_path(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    request = _request(tmp_path, dataset, base_model_path="  ")

    with pytest.raises(ValueError, match="base_model_path"):
        build_training_config(request, accepted_image_count=3)


def test_build_trainer_command_requires_existing_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Trainer entrypoint"):
        build_trainer_command(str(tmp_path / "missing.ps1"), config_path)


def test_build_trainer_command_requires_existing_config(tmp_path: Path) -> None:
    trainer = tmp_path / "train.exe"
    trainer.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Training config"):
        build_trainer_command(str(trainer), tmp_path / "missing.json")


def test_build_trainer_command_uses_powershell_for_ps1_entrypoint(tmp_path: Path) -> None:
    trainer = tmp_path / "train.ps1"
    trainer.write_text("", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    command = build_trainer_command(str(trainer), config_path)

    assert command == [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(trainer),
        str(config_path),
    ]


def test_build_trainer_command_uses_entrypoint_for_normal_executable(tmp_path: Path) -> None:
    trainer = tmp_path / "train.exe"
    trainer.write_text("", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    command = build_trainer_command(str(trainer), config_path)

    assert command == [str(trainer), str(config_path)]


def test_check_trainer_status_reports_missing_backend_and_config(tmp_path: Path) -> None:
    status = check_trainer_status(tmp_path / "missing.exe", tmp_path / "missing.json")

    assert status.trainer_entrypoint_exists is False
    assert status.config_path_exists is False
    assert any("trainer entrypoint" in warning.lower() for warning in status.warnings)
    assert any("config" in warning.lower() for warning in status.warnings)


def test_training_store_save_get_list_and_register_completed_lora(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=4)
    store = TrainingStore(WorkspacePaths(tmp_path))

    saved = store.save(config)
    loaded = store.get(saved.job_id)
    jobs = store.list()
    completed = store.register_completed_lora(
        saved.job_id, tmp_path / "loras" / "body.safetensors"
    )

    assert (tmp_path / "config" / "training" / f"{saved.job_id}.json").exists()
    assert loaded == saved
    assert [job.job_id for job in jobs] == [saved.job_id]
    assert completed.completed_lora_path == str(tmp_path / "loras" / "body.safetensors")
    assert store.get(saved.job_id).completed_lora_path == str(tmp_path / "loras" / "body.safetensors")
