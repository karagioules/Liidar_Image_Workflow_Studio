from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_model_studio.paths import WorkspacePaths
from local_model_studio.training_config import (
    build_trainer_command,
    build_training_config,
    check_trainer_status,
)
import local_model_studio.training_runner as training_runner
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
    assert config.resolution == 768
    assert config.repeats == 6
    assert config.batch_size == 1
    assert config.max_train_steps == 900
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


def test_training_config_rejects_dataset_path_that_is_file(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted.txt"
    dataset.write_text("not a directory", encoding="utf-8")
    request = _request(tmp_path, dataset)

    with pytest.raises(ValueError, match="dataset directory"):
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


def test_training_cancel_terminates_windows_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        pid = 1234

        def poll(self) -> None:
            return None

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)
        return object()

    monkeypatch.setattr(training_runner.sys, "platform", "win32")
    monkeypatch.setattr(training_runner.subprocess, "run", fake_run)

    training_runner._terminate_process_tree(FakeProcess())  # type: ignore[arg-type]

    assert calls == [["taskkill", "/PID", "1234", "/T", "/F"]]


def test_training_progress_parser_reads_latest_tqdm_line() -> None:
    lines = [
        "loading model",
        " 25%|██▌       | 210/832 [06:39<24:21,  2.35s/it]",
        " 30%|███       | 250/832 [07:48<15:18,  1.58s/it]",
    ]

    progress = training_runner._parse_training_progress(lines)

    assert progress == (250, 832, 30.0)


def test_training_store_save_get_list_and_register_completed_lora(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    image_path = dataset / "crop.jpg"
    image_path.write_bytes(b"fake image bytes")
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=4)
    store = TrainingStore(WorkspacePaths(tmp_path))
    lora_path = tmp_path / "loras" / "body.safetensors"
    lora_path.parent.mkdir()
    lora_path.write_text("artifact", encoding="utf-8")

    saved = store.save(config)
    loaded = store.get(saved.job_id)
    jobs = store.list()
    completed = store.register_completed_lora(saved.job_id, lora_path)

    prepared_root = tmp_path / "config" / "training_datasets" / saved.job_id
    prepared_image = prepared_root / "6_natural_body_shape" / "00001_crop.jpg"
    assert (tmp_path / "config" / "training" / f"{saved.job_id}.json").exists()
    assert saved.dataset_path == str(prepared_root)
    assert prepared_image.exists()
    assert prepared_image.with_suffix(".caption").exists()
    assert loaded == saved
    assert [job.job_id for job in jobs] == [saved.job_id]
    assert completed.completed_lora_path == str(lora_path)
    assert store.get(saved.job_id).completed_lora_path == str(lora_path)


def test_training_store_clear_pending_global_jobs_keeps_completed_packs(tmp_path: Path) -> None:
    pending_dataset = tmp_path / "pending"
    completed_dataset = tmp_path / "completed"
    pending_dataset.mkdir()
    completed_dataset.mkdir()
    (pending_dataset / "crop.jpg").write_bytes(b"pending image")
    (completed_dataset / "crop.jpg").write_bytes(b"completed image")
    store = TrainingStore(WorkspacePaths(tmp_path))
    pending = store.save(build_training_config(_request(tmp_path, pending_dataset), accepted_image_count=1))
    completed = store.save(build_training_config(_request(tmp_path, completed_dataset), accepted_image_count=1))
    lora_path = tmp_path / "loras" / "completed.safetensors"
    lora_path.parent.mkdir()
    lora_path.write_text("artifact", encoding="utf-8")
    store.register_completed_lora(completed.job_id, lora_path)

    removed_count = store.clear_pending_global_jobs()

    assert removed_count == 1
    assert [job.job_id for job in store.list()] == [completed.job_id]
    assert not store._path_for(pending.job_id).exists()
    assert store._path_for(completed.job_id).exists()
    assert Path(pending.dataset_path).exists()


def test_training_store_repairs_stale_job_dataset_before_training(tmp_path: Path) -> None:
    dataset = tmp_path / "downloads"
    dataset.mkdir()
    (dataset / "crop.jpg").write_bytes(b"fake image bytes")
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=1)
    store = TrainingStore(WorkspacePaths(tmp_path))
    stale = config.model_copy(update={"dataset_path": str(dataset)})
    store._path_for(stale.job_id).parent.mkdir(parents=True, exist_ok=True)
    store._path_for(stale.job_id).write_text(stale.model_dump_json(indent=2), encoding="utf-8")

    repaired = store.prepare_job_dataset(stale.job_id)

    prepared_root = tmp_path / "config" / "training_datasets" / stale.job_id
    assert repaired.dataset_path == str(prepared_root)
    assert (prepared_root / "6_natural_body_shape" / "00001_crop.jpg").exists()
    assert (prepared_root / "6_natural_body_shape" / "00001_crop.caption").exists()


def test_training_store_rejects_missing_completed_lora_path(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=4)
    store = TrainingStore(WorkspacePaths(tmp_path))
    saved = store.save(config)

    with pytest.raises(FileNotFoundError, match="LoRA artifact"):
        store.register_completed_lora(saved.job_id, tmp_path / "missing.safetensors")


def test_training_store_checks_missing_job_before_missing_completed_lora_path(
    tmp_path: Path,
) -> None:
    store = TrainingStore(WorkspacePaths(tmp_path))

    with pytest.raises(KeyError):
        store.register_completed_lora("missing-job", tmp_path / "missing.safetensors")


def test_training_store_rejects_invalid_completed_lora_suffix(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=4)
    store = TrainingStore(WorkspacePaths(tmp_path))
    saved = store.save(config)
    lora_path = tmp_path / "body.txt"
    lora_path.write_text("artifact", encoding="utf-8")

    with pytest.raises(ValueError, match="LoRA artifact file must end with"):
        store.register_completed_lora(saved.job_id, lora_path)


def test_training_store_accepts_existing_completed_lora_file(tmp_path: Path) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=4)
    store = TrainingStore(WorkspacePaths(tmp_path))
    saved = store.save(config)
    lora_path = tmp_path / "body.safetensors"
    lora_path.write_text("artifact", encoding="utf-8")

    completed = store.register_completed_lora(saved.job_id, lora_path)

    assert completed.completed_lora_path == str(lora_path)


def test_training_store_overwrite_preserves_created_at_and_advances_updated_at(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "accepted"
    dataset.mkdir()
    config = build_training_config(_request(tmp_path, dataset), accepted_image_count=4)
    store = TrainingStore(WorkspacePaths(tmp_path))
    saved = store.save(
        config.model_copy(
            update={
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
        )
    )

    overwritten = store.save(saved.model_copy(update={"lora_name": "updated_name"}))

    assert overwritten.created_at == saved.created_at
    assert overwritten.updated_at > saved.updated_at
