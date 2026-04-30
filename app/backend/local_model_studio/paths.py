from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def characters_dir(self) -> Path:
        return self.root / "config" / "characters"

    @property
    def presets_dir(self) -> Path:
        return self.root / "config" / "presets"

    @property
    def workflows_dir(self) -> Path:
        return self.root / "config" / "workflows"

    @property
    def training_dir(self) -> Path:
        return self.root / "config" / "training"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs" / "images"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "outputs" / "metadata"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


def default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]
