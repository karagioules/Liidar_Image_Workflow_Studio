from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import CharacterProfile


class ProfileStore:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths
        self.paths.characters_dir.mkdir(parents=True, exist_ok=True)

    def save(self, profile: CharacterProfile) -> CharacterProfile:
        profile = CharacterProfile.model_validate(profile.model_dump())
        now = datetime.now(UTC)
        existing_created_at = profile.created_at
        path = self._path_for(profile.id)
        if path.exists():
            existing_created_at = CharacterProfile.model_validate_json(
                path.read_text(encoding="utf-8")
            ).created_at
        saved = profile.model_copy(update={"created_at": existing_created_at, "updated_at": now})
        path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
        return saved

    def list(self) -> list[CharacterProfile]:
        profiles = [self._load_path(path) for path in self.paths.characters_dir.glob("*.json")]
        return sorted(profiles, key=lambda profile: profile.display_name.casefold())

    def get(self, profile_id: str) -> CharacterProfile:
        path = self._path_for(profile_id)
        if not path.exists():
            raise KeyError(profile_id)
        return self._load_path(path)

    def delete(self, profile_id: str) -> None:
        path = self._path_for(profile_id)
        if not path.exists():
            raise KeyError(profile_id)
        path.unlink()

    def _path_for(self, profile_id: str) -> Path:
        safe_id = "".join(char for char in profile_id if char.isalnum() or char in {"-", "_"})
        if not safe_id:
            raise ValueError("Profile id must contain safe filename characters.")
        return self.paths.characters_dir / f"{safe_id}.json"

    def _load_path(self, path: Path) -> CharacterProfile:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CharacterProfile.model_validate(data)
