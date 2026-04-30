from __future__ import annotations

import os
import string
from io import BytesIO
from pathlib import Path

from PIL import Image

from local_model_studio.schemas import PathBrowserEntry, PathBrowserResponse


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def browse_filesystem(raw_path: str | None, workspace_root: Path) -> PathBrowserResponse:
    if raw_path is None or not raw_path.strip():
        return PathBrowserResponse(
            current_path=None,
            parent_path=None,
            entries=_default_entries(workspace_root),
        )

    target = Path(raw_path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    if target.is_file():
        target = target.parent
    if not target.is_dir():
        raise ValueError(f"Path is not a folder: {target}")

    entries = [_entry for child in _safe_children(target) if (_entry := _to_entry(child)) is not None]
    return PathBrowserResponse(
        current_path=str(target),
        parent_path=str(target.parent) if target.parent != target else None,
        entries=sorted(entries, key=lambda entry: (entry.kind == "file", entry.name.lower())),
    )


def render_thumbnail(raw_path: str, max_size: int = 320) -> bytes:
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Path is not a supported image file: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        output = BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
        return output.getvalue()


def _default_entries(workspace_root: Path) -> list[PathBrowserEntry]:
    entries = [
        PathBrowserEntry(name="Workspace", path=str(workspace_root), kind="directory"),
        PathBrowserEntry(name="Home", path=str(Path.home()), kind="directory"),
    ]
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                entries.append(PathBrowserEntry(name=f"{letter}:\\", path=str(drive), kind="drive"))
    else:
        entries.append(PathBrowserEntry(name="/", path="/", kind="drive"))
    return entries


def _safe_children(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _to_entry(path: Path) -> PathBrowserEntry | None:
    try:
        if path.is_dir():
            return PathBrowserEntry(name=path.name, path=str(path), kind="directory")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return PathBrowserEntry(name=path.name, path=str(path), kind="file")
    except OSError:
        return None
    return None
