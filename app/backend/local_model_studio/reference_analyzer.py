from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

from local_model_studio.schemas import BodyAttributeCues, ReferenceAnalysisImage, ReferenceAnalysisRequest, ReferenceAnalysisResponse


NAME_SKIP_FOLDERS = {"ai faces", "free_posts", "may2025", "models", "reference", "references"}


@dataclass(frozen=True)
class ImageSignals:
    width: int
    height: int
    red: float
    green: float
    blue: float
    brightness: float
    texture: float


def analyze_references(
    request: ReferenceAnalysisRequest,
    captions: list[str] | None = None,
) -> ReferenceAnalysisResponse:
    paths = [Path(path) for path in request.reference_images]
    signals = [_read_image_signals(path) for path in paths]
    if not signals:
        raise ValueError("At least one reference image is required.")

    display_name = request.display_name.strip() or _display_name_from_paths(paths)
    orientation = _orientation_note(signals)
    warmth = _warmth_note(signals)
    lighting = _lighting_note(signals)
    texture = _texture_note(signals)
    count = len(signals)
    plural = "images" if count != 1 else "image"
    caption_text = _caption_text(captions)
    vision_prefix = f"local vision model caption: {caption_text}; " if caption_text else ""
    analysis_images = _analysis_images(paths, signals, captions)
    body_cues = [image.body_attributes for image in analysis_images]
    warnings = _analysis_warnings(signals, captions)
    consistency_score = _consistency_score(signals, captions)

    return ReferenceAnalysisResponse(
        id=request.id,
        display_name=display_name,
        age_category=request.age_category,
        face_summary=_fit_text(
            f"offline image analysis from {count} reference {plural}; {vision_prefix}fictional adult face identity guided by the selected local images; "
            "preserve recurring face structure, expression style, and natural asymmetry without matching any real person",
            600,
        ),
        hair=_fit_text(f"{vision_prefix}reference-inferred hair color, length, volume, and styling; keep it consistent unless manually changed", 240),
        eyes=_fit_text(f"{vision_prefix}reference-inferred eye shape and color; keep gaze and expression style consistent", 160),
        skin_tone=f"{warmth} natural skin tone inferred from local image color balance; keep skin texture believable",
        body_shape="reference-inferred adult body proportions and posture; keep anatomy natural and physically plausible",
        chest=_chest_note(body_cues),
        grooming=_fit_text(f"{vision_prefix}reference-inferred grooming and presentation; keep details realistic and consistent", 240),
        style_notes=_fit_text(
            f"offline image analysis: {orientation}, {lighting}, {texture}; {vision_prefix}reference-guided believable still photo style "
            "with natural camera rendering and no overpolished AI look",
            600,
        ),
        negative_notes="plastic skin, airbrushed, overprocessed, uncanny symmetry, celebrity, real person, underage, childlike",
        reference_images=[str(path) for path in paths],
        lora_files=request.lora_files,
        seed_strategy=request.seed_strategy,
        locked_seed=request.locked_seed,
        consistency_score=consistency_score,
        analysis_warnings=warnings,
        analysis_images=analysis_images,
    )


def _caption_text(captions: list[str] | None) -> str:
    if not captions:
        return ""
    cleaned = [caption.strip().rstrip(".") for caption in captions if caption.strip()]
    return "; ".join(cleaned[:3])


def _fit_text(value: str, max_length: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip(" ;,.") + "..."


def _analysis_images(
    paths: list[Path],
    signals: list[ImageSignals],
    captions: list[str] | None,
) -> list[ReferenceAnalysisImage]:
    return [
        ReferenceAnalysisImage(
            path=str(path),
            file_name=path.name,
            width=signal.width,
            height=signal.height,
            orientation=_single_orientation(signal),
            brightness=_lighting_note([signal]),
            tone=_warmth_note([signal]),
            texture=_texture_note([signal]),
            caption=captions[index].strip() if captions and index < len(captions) and captions[index].strip() else None,
            body_attributes=_body_attribute_cues(signal, captions[index] if captions and index < len(captions) else ""),
        )
        for index, (path, signal) in enumerate(zip(paths, signals, strict=True))
    ]


def _body_attribute_cues(signal: ImageSignals, caption: str | None) -> BodyAttributeCues:
    caption_text = (caption or "").strip().lower()
    coverage, coverage_evidence, coverage_confidence = _coverage_from_caption(caption_text)
    chest_visibility = _chest_visibility_from_caption(caption_text, coverage)
    pose_framing = _pose_framing(signal, caption_text)
    evidence = coverage_evidence or "caption does not provide clear clothing or coverage evidence"
    confidence = coverage_confidence
    if "full" in pose_framing or "upper body" in pose_framing:
        confidence = min(100, confidence + 8)
    return BodyAttributeCues(
        coverage=coverage,
        chest_visibility=chest_visibility,
        pose_framing=pose_framing,
        confidence=confidence,
        evidence=_fit_text(evidence, 240),
    )


def _coverage_from_caption(caption: str) -> tuple[str, str, int]:
    if _contains_any(caption, ["nude", "naked", "topless", "bare chest"]):
        return "uncovered or explicit", "caption mentions uncovered body terms", 76
    if _contains_any(caption, ["bikini", "swimsuit", "swimwear", "lingerie", "underwear", "bra"]):
        keyword = _first_present(caption, ["bikini", "swimsuit", "swimwear", "lingerie", "underwear", "bra"])
        return "swimwear" if keyword in {"bikini", "swimsuit", "swimwear"} else "underwear or lingerie", f"caption mentions {keyword}", 72
    if _contains_any(caption, ["shirt", "dress", "top", "jacket", "hoodie", "sweater", "clothed", "wearing"]):
        keyword = _first_present(caption, ["shirt", "dress", "top", "jacket", "hoodie", "sweater", "clothed", "wearing"])
        return "clothed", f"caption mentions {keyword}", 62
    return "unclear", "", 35


def _chest_visibility_from_caption(caption: str, coverage: str) -> str:
    if coverage == "uncovered or explicit":
        return "visible, requires manual review"
    if coverage in {"swimwear", "underwear or lingerie"}:
        return "covered by swimwear or clothing"
    if _contains_any(caption, ["close up", "upper body", "portrait"]):
        return "partly visible through framing"
    return "not clearly described"


def _pose_framing(signal: ImageSignals, caption: str) -> str:
    if _contains_any(caption, ["full body", "standing", "on the beach", "bikini"]):
        return "full or upper body visible"
    if _contains_any(caption, ["portrait", "close up", "headshot"]):
        return "portrait or close framing"
    if signal.height > signal.width * 1.08:
        return "portrait frame, body visibility depends on crop"
    return "unclear framing"


def _chest_note(cues: list[BodyAttributeCues]) -> str:
    coverages = {cue.coverage for cue in cues}
    if "uncovered or explicit" in coverages:
        return "reference cues indicate visible chest anatomy; use manual review for exact attributes and keep results natural"
    if "swimwear" in coverages:
        return "reference cues indicate swimwear-covered chest/body shape; infer only broad natural proportions unless manually specified"
    if "underwear or lingerie" in coverages:
        return "reference cues indicate underwear or lingerie coverage; infer only broad natural proportions unless manually specified"
    return "reference-inferred natural adult body shape; avoid exaggerated or plastic-looking anatomy"


def _contains_any(value: str, needles: list[str]) -> bool:
    return any(needle in value for needle in needles)


def _first_present(value: str, needles: list[str]) -> str:
    return next((needle for needle in needles if needle in value), needles[0])


def _analysis_warnings(signals: list[ImageSignals], captions: list[str] | None) -> list[str]:
    warnings: list[str] = []
    if len(signals) < 3:
        warnings.append("Add 2-4 more references for stronger identity consistency.")
    orientations = {_single_orientation(signal) for signal in signals}
    if len(orientations) > 1:
        warnings.append("References mix orientations; add more matching angles if you want tighter consistency.")
    if captions is None:
        warnings.append("Local vision captions were not used; analysis is based on image signals only.")
    elif any(not caption.strip() for caption in captions):
        warnings.append("One or more references did not produce a local vision caption.")
    return warnings


def _consistency_score(signals: list[ImageSignals], captions: list[str] | None) -> int:
    score = 45
    score += min(len(signals), 5) * 8
    if len({_single_orientation(signal) for signal in signals}) == 1:
        score += 12
    brightness_values = [signal.brightness for signal in signals]
    if max(brightness_values) - min(brightness_values) <= 45:
        score += 10
    if captions and all(caption.strip() for caption in captions):
        score += 8
    return max(0, min(100, score))


def _read_image_signals(path: Path) -> ImageSignals:
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Reference path is not a file: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        sampled = image.resize((64, 64))
        stat = ImageStat.Stat(sampled)
        red, green, blue = stat.mean
        brightness = (red + green + blue) / 3
        texture = sum(stat.stddev) / 3
    return ImageSignals(width, height, red, green, blue, brightness, texture)


def _display_name_from_paths(paths: list[Path]) -> str:
    first = paths[0]
    parent = first.parent.name
    if parent and parent.lower() not in NAME_SKIP_FOLDERS:
        return _humanize_name(parent)
    return _humanize_name(first.stem)


def _humanize_name(value: str) -> str:
    cleaned = value.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in cleaned.split()) or "New Character"


def _orientation_note(signals: list[ImageSignals]) -> str:
    portrait = sum(1 for signal in signals if signal.height > signal.width * 1.08)
    landscape = sum(1 for signal in signals if signal.width > signal.height * 1.08)
    if portrait >= landscape and portrait > 0:
        return "portrait-oriented references"
    if landscape > 0:
        return "landscape-oriented references"
    return "square or balanced-frame references"


def _single_orientation(signal: ImageSignals) -> str:
    if signal.height > signal.width * 1.08:
        return "portrait"
    if signal.width > signal.height * 1.08:
        return "landscape"
    return "square"


def _warmth_note(signals: list[ImageSignals]) -> str:
    red = sum(signal.red for signal in signals) / len(signals)
    blue = sum(signal.blue for signal in signals) / len(signals)
    if red - blue > 8:
        return "warm"
    if blue - red > 8:
        return "cool"
    return "neutral"


def _lighting_note(signals: list[ImageSignals]) -> str:
    brightness = sum(signal.brightness for signal in signals) / len(signals)
    if brightness >= 190:
        return "bright high-key lighting"
    if brightness >= 95:
        return "natural mid-key lighting"
    return "low-key lighting"


def _texture_note(signals: list[ImageSignals]) -> str:
    texture = sum(signal.texture for signal in signals) / len(signals)
    if texture >= 55:
        return "high visual texture"
    if texture >= 22:
        return "moderate natural texture"
    return "clean low-noise texture"
