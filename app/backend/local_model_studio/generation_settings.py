from __future__ import annotations

import json
from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import GenerationSettings


SETTINGS_FILE_NAME = "generation-settings.json"


class GenerationSettingsStore:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths

    @property
    def path(self) -> Path:
        return self.paths.workflows_dir / SETTINGS_FILE_NAME

    def load(self) -> GenerationSettings:
        if not self.path.exists():
            return GenerationSettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generation settings file is invalid JSON: {self.path}") from exc
        return GenerationSettings.model_validate(payload)

    def save(self, settings: GenerationSettings) -> GenerationSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = GenerationSettings(checkpoint_name=settings.checkpoint_name.strip())
        self.path.write_text(cleaned.model_dump_json(indent=2), encoding="utf-8")
        return cleaned


def comfy_checkpoint_path(paths: WorkspacePaths, checkpoint_name: str) -> Path:
    return paths.root / "ComfyUI" / "models" / "checkpoints" / checkpoint_name


def comfy_lora_path(paths: WorkspacePaths, lora_file: str) -> Path:
    candidate = Path(lora_file)
    if candidate.is_absolute():
        return candidate
    return paths.root / "ComfyUI" / "models" / "loras" / lora_file
