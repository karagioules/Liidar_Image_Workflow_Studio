from pathlib import Path

import pytest

from local_model_studio.paths import WorkspacePaths
from local_model_studio.profile_store import ProfileStore
from local_model_studio.schemas import CharacterProfile


def make_profile(name: str = "Domna Test") -> CharacterProfile:
    return CharacterProfile(
        display_name=name,
        age_category="adult_25_plus",
        face_summary="soft oval face, brown eyes, warm smile",
        hair="long dark brunette hair",
        skin_tone="olive skin tone",
        body_shape="curvy natural build",
        chest="natural teardrop chest shape",
        grooming="natural adult grooming",
        style_notes="Mediterranean lifestyle photography",
        negative_notes="cartoon, distorted hands, low detail",
    )


def test_save_and_load_profile(tmp_path: Path) -> None:
    store = ProfileStore(WorkspacePaths(tmp_path))
    saved = store.save(make_profile())

    loaded = store.get(saved.id)

    assert loaded.id == saved.id
    assert loaded.display_name == "Domna Test"
    assert loaded.age_category == "adult_25_plus"


def test_rejects_non_adult_age_category(tmp_path: Path) -> None:
    store = ProfileStore(WorkspacePaths(tmp_path))

    with pytest.raises(ValueError, match="adult"):
        store.save(make_profile().model_copy(update={"age_category": "teen"}))


def test_list_profiles_sorts_by_display_name(tmp_path: Path) -> None:
    store = ProfileStore(WorkspacePaths(tmp_path))
    store.save(make_profile("Zeta"))
    store.save(make_profile("Alpha"))

    names = [profile.display_name for profile in store.list()]

    assert names == ["Alpha", "Zeta"]
