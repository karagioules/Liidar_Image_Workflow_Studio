from pathlib import Path

from local_model_studio.paths import WorkspacePaths


def test_workspace_paths_are_rooted_in_repo(tmp_path: Path) -> None:
    paths = WorkspacePaths(tmp_path)

    assert paths.root == tmp_path
    assert paths.characters_dir == tmp_path / "config" / "characters"
    assert paths.presets_dir == tmp_path / "config" / "presets"
    assert paths.workflows_dir == tmp_path / "config" / "workflows"
    assert paths.outputs_dir == tmp_path / "outputs" / "images"
    assert paths.metadata_dir == tmp_path / "outputs" / "metadata"
    assert paths.logs_dir == tmp_path / "logs"
