from __future__ import annotations

import re
from dataclasses import dataclass

from local_model_studio.schemas import CharacterProfile, GenerationMode, PromptEnhanceResponse


@dataclass(frozen=True)
class BriefAnalysis:
    cleaned: str
    preset_id: str
    preset_label: str
    subject: str
    action: str
    setting: str
    lighting: str
    style: str
    body_focus: bool
    nude: bool
    summary_parts: list[str]


@dataclass(frozen=True)
class PromptPreset:
    id: str
    label: str
    keywords: tuple[str, ...]
    scene_cues: tuple[str, ...]
    body_cues: tuple[str, ...] = ()
    negative_cues: tuple[str, ...] = ()
    mode_bonus: tuple[GenerationMode, ...] = ()


PROMPT_PRESETS: tuple[PromptPreset, ...] = (
    PromptPreset(
        id="bedroom_candid",
        label="bedroom candid",
        keywords=("bed", "bedroom", "hotel", "sheets", "pillow"),
        scene_cues=(
            "intimate bedroom still photo",
            "soft fabric contact shadows",
            "natural bed-level perspective",
            "unforced candid framing",
        ),
        body_cues=("believable body weight on bedding", "natural skin compression where body meets fabric"),
        negative_cues=("floating above bed", "warped bedding", "melted fabric"),
    ),
    PromptPreset(
        id="mirror_selfie",
        label="mirror selfie",
        keywords=("mirror", "selfie", "phone"),
        scene_cues=("realistic mirror selfie", "visible phone-camera perspective", "casual indoor lighting", "slight handheld imperfection"),
        negative_cues=("impossible reflection", "duplicated phone", "wrong mirror geometry"),
    ),
    PromptPreset(
        id="studio_portrait",
        label="studio portrait",
        keywords=("studio", "backdrop", "professional headshot"),
        scene_cues=("controlled studio setup", "clean backdrop", "soft key light", "subtle fill light", "professional portrait lens"),
        negative_cues=("flat passport lighting", "over-smoothed skin"),
        mode_bonus=("studio", "portrait"),
    ),
    PromptPreset(
        id="lifestyle_candid",
        label="lifestyle candid",
        keywords=("candid", "lifestyle", "social media", "casual", "coffee", "apartment"),
        scene_cues=("candid lifestyle composition", "natural everyday environment", "unposed expression", "plausible casual framing"),
        negative_cues=("staged stock photo", "advertising pose"),
        mode_bonus=("lifestyle_post",),
    ),
    PromptPreset(
        id="beach_outdoor",
        label="outdoor beach",
        keywords=("beach", "sand", "sea", "ocean", "pool", "sun"),
        scene_cues=("outdoor natural light", "realistic environmental bounce light", "natural wind and posture", "believable outdoor exposure"),
        negative_cues=("painted sky", "fake horizon", "plastic water"),
    ),
    PromptPreset(
        id="low_light_phone",
        label="low-light phone",
        keywords=("low light", "night", "flash", "club", "bar"),
        scene_cues=("low-light phone photo realism", "controlled sensor noise", "natural shadow falloff", "imperfect autofocus realism"),
        negative_cues=("crushed shadows", "overexposed flash face", "muddy detail"),
    ),
    PromptPreset(
        id="full_body_reference",
        label="full-body reference",
        keywords=("full body", "standing", "reference", "front view", "side view"),
        scene_cues=("clear full-body framing", "neutral camera height", "complete silhouette visible", "minimal perspective distortion"),
        body_cues=("readable shoulder waist hip relationship", "balanced limb proportions"),
        negative_cues=("cropped feet", "cropped head", "wide-angle distortion"),
        mode_bonus=("full_body", "reference_match"),
    ),
)


MODE_CUES: dict[GenerationMode, list[str]] = {
    "portrait": ["portrait composition", "sharp eyes", "relaxed expression", "natural head position"],
    "full_body": ["full body composition", "coherent proportions", "visible silhouette", "grounded natural stance"],
    "lifestyle_post": ["candid lifestyle photo", "casual framing", "believable social media still", "unforced pose"],
    "studio": ["studio photography", "controlled soft key light", "clean background", "professional camera look"],
    "reference_match": ["reference-inspired composition", "consistent identity", "natural camera perspective"],
}

STYLE_CUES = {
    "polaroid": "instant film look",
    "film": "subtle film grain",
    "cinematic": "cinematic natural color grade",
    "selfie": "casual phone camera perspective",
    "mirror": "mirror photo composition",
    "professional": "professional camera photo",
    "candid": "candid documentary feel",
}

LIGHTING_CUES = {
    "soft natural light": "soft natural light",
    "natural light": "natural light",
    "window light": "soft window light",
    "sunset": "warm sunset light",
    "golden hour": "golden hour light",
    "studio light": "controlled studio lighting",
    "flash": "direct flash photography",
    "low light": "realistic low light",
    "morning": "gentle morning light",
}

SETTING_CUES = {
    "bed": "bedroom setting",
    "bedroom": "bedroom setting",
    "hotel": "hotel room setting",
    "balcony": "balcony setting",
    "bathroom": "bathroom setting",
    "shower": "bathroom shower setting",
    "beach": "beach setting",
    "street": "street setting",
    "car": "inside a car",
    "kitchen": "kitchen interior",
    "studio": "photo studio setting",
}

ACTION_CUES = {
    "bending over": "bending over in a natural pose",
    "bend over": "bending over in a natural pose",
    "lying": "lying down naturally",
    "laying": "lying down naturally",
    "sitting": "sitting naturally",
    "standing": "standing naturally",
    "kneeling": "kneeling naturally",
    "walking": "walking naturally",
    "looking back": "looking back toward the camera",
    "over shoulder": "over-the-shoulder pose",
    "stretching": "stretching naturally",
}

BODY_FOCUS_TERMS = {
    "nude",
    "naked",
    "body",
    "full body",
    "figure",
    "curves",
    "torso",
    "waist",
    "hips",
    "chest",
    "breasts",
    "bending",
    "pose",
}

BASE_SCENE_CUES = [
    "photorealistic adult still photo",
    "realistic camera perspective",
    "natural skin texture",
    "believable candid imperfections",
    "accurate contact shadows",
]

BASE_NEGATIVE = [
    "minor",
    "childlike",
    "underage",
    "low quality",
    "blurry",
    "distorted anatomy",
    "distorted hands",
    "extra fingers",
    "missing fingers",
    "extra limbs",
    "plastic skin",
    "waxy skin",
    "overprocessed",
    "harsh AI artifacts",
    "text",
    "watermark",
    "logo",
]

POSE_NEGATIVE = {
    "bending over": ["broken spine", "impossible back bend", "detached hips", "unnatural hip rotation"],
    "lying": ["floating body", "melted limbs"],
    "kneeling": ["broken knees", "impossible leg angle"],
}


def enhance_photo_brief(profile: CharacterProfile, brief: str, mode: GenerationMode) -> PromptEnhanceResponse:
    analysis = _analyze_brief(brief, mode)
    scene_prompt = _scene_prompt(analysis, mode)
    body_detail = _body_detail_prompt(profile, analysis)
    negative = _negative_prompt(analysis)
    return PromptEnhanceResponse(
        scene_prompt=scene_prompt,
        body_detail_prompt=body_detail,
        extra_negative=negative,
        summary=" · ".join(analysis.summary_parts),
    )


def _analyze_brief(brief: str, mode: GenerationMode) -> BriefAnalysis:
    cleaned = _clean_phrase(brief)
    lowered = cleaned.lower()
    preset = _select_preset(lowered, mode_hint=mode)
    nude = _has_any(lowered, ["nude", "naked", "unclothed"])
    action = _first_match(lowered, ACTION_CUES) or "natural relaxed pose"
    setting = _first_match(lowered, SETTING_CUES) or ""
    lighting = _first_match(lowered, LIGHTING_CUES) or "soft natural light"
    style = _first_match(lowered, STYLE_CUES) or "realistic camera photo"
    body_focus = nude or any(term in lowered for term in BODY_FOCUS_TERMS)
    subject = "fictional adult woman"

    summary_parts = [
        preset.label,
        "nude/adult pose" if nude else _summary_label(action),
        _summary_label(lighting),
    ]
    if body_focus:
        summary_parts.append("body detail included")
    return BriefAnalysis(
        cleaned=cleaned,
        preset_id=preset.id,
        preset_label=preset.label,
        subject=subject,
        action=action,
        setting=setting,
        lighting=lighting,
        style=style,
        body_focus=body_focus,
        nude=nude,
        summary_parts=_dedupe(summary_parts),
    )


def _scene_prompt(analysis: BriefAnalysis, mode: GenerationMode) -> str:
    preset = _preset_by_id(analysis.preset_id)
    original_detail = _remove_duplicate_fragments(
        analysis.cleaned,
        [analysis.lighting, analysis.setting, analysis.action, analysis.style],
    )
    parts = [
        analysis.subject,
        original_detail,
        analysis.setting,
        analysis.action,
        analysis.lighting,
        analysis.style,
        *preset.scene_cues,
        *MODE_CUES[mode],
        *BASE_SCENE_CUES,
        "natural body weight and posture",
    ]
    if analysis.nude:
        parts.insert(2, "tasteful adult nude pose")
    return _join_prompt_parts(parts)


def _body_detail_prompt(profile: CharacterProfile, analysis: BriefAnalysis) -> str:
    if not analysis.body_focus:
        return ""
    preset = _preset_by_id(analysis.preset_id)
    parts = [
        profile.body_shape,
        profile.chest,
        profile.grooming,
        *preset.body_cues,
        "natural adult anatomy",
        "coherent torso and hip alignment",
        "realistic body proportions",
        "pose-specific skin folds",
        "accurate contact shadows",
        "body detail supports the photo without overpowering the scene",
    ]
    if "bending over" in analysis.action:
        parts.extend(["believable spine curve", "grounded hips and shoulders"])
    return _join_prompt_parts(parts)


def _negative_prompt(analysis: BriefAnalysis) -> str:
    preset = _preset_by_id(analysis.preset_id)
    negatives = list(BASE_NEGATIVE)
    negatives.extend(preset.negative_cues)
    for key, values in POSE_NEGATIVE.items():
        if key in analysis.action:
            negatives.extend(values)
    if analysis.nude:
        negatives.extend(["sexualized childlike features", "teen", "young-looking face"])
    return _join_prompt_parts(negatives)


def _select_preset(value: str, mode_hint: GenerationMode | None) -> PromptPreset:
    best = PROMPT_PRESETS[0]
    best_score = -1
    for preset in PROMPT_PRESETS:
        score = sum(2 for keyword in preset.keywords if keyword in value)
        if mode_hint in preset.mode_bonus:
            score += 1
        if score > best_score:
            best = preset
            best_score = score
    if best_score <= 0:
        return _preset_by_id("lifestyle_candid")
    return best


def _preset_by_id(preset_id: str) -> PromptPreset:
    return next((preset for preset in PROMPT_PRESETS if preset.id == preset_id), PROMPT_PRESETS[0])


def _first_match(value: str, mapping: dict[str, str]) -> str:
    for needle, cue in mapping.items():
        if needle in value:
            return cue
    return ""


def _has_any(value: str, needles: list[str]) -> bool:
    return any(needle in value for needle in needles)


def _remove_duplicate_fragments(value: str, fragments: list[str]) -> str:
    fragment_keys = {fragment.lower() for fragment in fragments if fragment}
    clauses = [_clean_phrase(clause).strip(" ,") for clause in re.split(r",|;", value)]
    kept = [
        clause
        for clause in clauses
        if clause
        and clause.lower() not in fragment_keys
        and clause.lower() != "nude"
        and not _is_generic_photo_clause(clause)
        and not _is_structural_pose_location_clause(clause)
    ]
    return ", ".join(kept)


def _is_generic_photo_clause(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"believable still photo", "realistic photo", "photo", "still photo", "photorealistic"}


def _is_structural_pose_location_clause(value: str) -> bool:
    lowered = value.lower()
    has_action = any(needle in lowered for needle in ACTION_CUES)
    has_setting = any(needle in lowered for needle in SETTING_CUES)
    return has_action and has_setting


def _summary_label(value: str) -> str:
    return value.replace(" setting", "").replace("photography", "photo").strip()


def _clean_phrase(value: str) -> str:
    return " ".join(value.strip().split())


def _join_prompt_parts(parts: list[str]) -> str:
    return ", ".join(_dedupe(parts))


def _dedupe(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for part in parts:
        item = _clean_phrase(part).strip(" ,")
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)
    return cleaned
