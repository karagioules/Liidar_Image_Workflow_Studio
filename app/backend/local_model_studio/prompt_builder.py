from __future__ import annotations

import secrets

from local_model_studio.schemas import CharacterProfile, GenerationRequest, PromptRecipe


QUALITY_SETTINGS = {
    "fast": {"width": 832, "height": 1216, "steps": 18, "cfg": 4.5},
    "balanced": {"width": 896, "height": 1344, "steps": 26, "cfg": 5.0},
    "high": {"width": 1024, "height": 1536, "steps": 34, "cfg": 5.5},
    "ultra": {"width": 1024, "height": 1536, "steps": 42, "cfg": 6.0},
}

MODE_PROMPTS = {
    "portrait": "professional realistic portrait, sharp eyes, natural skin texture",
    "full_body": "realistic full body photo, coherent anatomy, natural proportions",
    "lifestyle_post": "realistic lifestyle social media photo, candid composition",
    "studio": "professional studio photography, clean lighting, high detail",
    "reference_match": "realistic reference-guided composition, consistent identity",
}

BASE_NEGATIVE = "minor, underage, childlike, celebrity, real person, distorted face, distorted hands, low detail, watermark, text"


def build_prompt_recipe(profile: CharacterProfile, request: GenerationRequest) -> PromptRecipe:
    settings = QUALITY_SETTINGS[request.quality]
    seed = _resolve_seed(profile, request)
    positive_parts = [
        "fictional adult woman, age 25 or older",
        MODE_PROMPTS[request.mode],
        profile.face_summary,
        profile.hair,
        profile.eyes,
        profile.skin_tone,
        profile.body_shape,
        profile.chest,
        profile.grooming,
        profile.style_notes,
        request.scene_prompt,
        "photorealistic, high-end creator content, consistent character identity",
    ]
    negative_parts = [BASE_NEGATIVE, profile.negative_notes, request.extra_negative]
    return PromptRecipe(
        positive=", ".join(part.strip() for part in positive_parts if part.strip()),
        negative=", ".join(part.strip() for part in negative_parts if part.strip()),
        seed=seed,
        width=settings["width"],
        height=settings["height"],
        steps=settings["steps"],
        cfg=settings["cfg"],
    )


def _resolve_seed(profile: CharacterProfile, request: GenerationRequest) -> int:
    if request.seed is not None:
        return request.seed
    if profile.seed_strategy == "locked" and profile.locked_seed is not None:
        return profile.locked_seed
    return secrets.randbelow(2_147_483_647)
