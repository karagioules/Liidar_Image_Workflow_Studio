from local_model_studio.prompt_builder import build_prompt_recipe
from local_model_studio.schemas import CharacterProfile, GenerationRequest


def profile() -> CharacterProfile:
    return CharacterProfile(
        display_name="Sample Profile",
        age_category="adult_25_plus",
        face_summary="consistent soft face, green eyes",
        hair="wavy black hair",
        skin_tone="warm olive skin",
        body_shape="petite natural build",
        chest="small natural chest",
        grooming="soft natural adult grooming",
        style_notes="realistic influencer photography",
        negative_notes="deformed anatomy, extra fingers",
        locked_seed=1234,
        seed_strategy="locked",
    )


def test_builds_locked_identity_prompt() -> None:
    recipe = build_prompt_recipe(
        profile(),
        GenerationRequest(character_id="abc", mode="lifestyle_post", scene_prompt="sunlit balcony selfie"),
    )

    assert "Sample Profile" not in recipe.positive
    assert "consistent soft face" in recipe.positive
    assert "small natural chest" in recipe.positive
    assert "sunlit balcony selfie" in recipe.positive
    assert "deformed anatomy" in recipe.negative
    assert recipe.seed == 1234


def test_recipe_biases_toward_believable_photos() -> None:
    recipe = build_prompt_recipe(
        profile(),
        GenerationRequest(character_id="abc", mode="portrait", scene_prompt="coffee shop window light"),
    )

    assert "natural skin texture" in recipe.positive
    assert "realistic camera photo" in recipe.positive
    assert "imperfect candid realism" in recipe.positive
    assert "plastic skin" in recipe.negative
    assert "airbrushed" in recipe.negative
    assert "overprocessed" in recipe.negative
    assert "AI-looking" in recipe.negative
    assert "uncanny symmetry" in recipe.negative
    assert "waxy skin" in recipe.negative
    assert "minor" in recipe.negative
    assert "underage" in recipe.negative
    assert "childlike" in recipe.negative
    assert "celebrity" in recipe.negative
    assert "real person" in recipe.negative
    assert "video" not in recipe.positive.lower()
    assert "video" not in recipe.negative.lower()


def test_quality_changes_dimensions_and_steps() -> None:
    fast = build_prompt_recipe(profile(), GenerationRequest(character_id="abc", quality="fast"))
    ultra = build_prompt_recipe(profile(), GenerationRequest(character_id="abc", quality="ultra"))

    assert fast.steps < ultra.steps
    assert fast.width <= ultra.width
    assert fast.height <= ultra.height
