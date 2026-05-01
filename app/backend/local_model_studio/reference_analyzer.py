from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageStat

from local_model_studio.schemas import (
    AggregateReferenceIntelligence,
    AdultContentSignals,
    AttributeDetail,
    BodyAttributeCues,
    ReferenceAnalysisImage,
    ReferenceAnalysisRequest,
    ReferenceAnalysisResponse,
    ReferenceImageDetails,
    VisionTag,
)


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


class ReferenceTagReport(Protocol):
    tags: list
    rating: str
    rating_confidence: float
    source: str

    def reference_text(self) -> str:
        ...


def analyze_references(
    request: ReferenceAnalysisRequest,
    captions: list[str] | None = None,
    tag_reports: list[ReferenceTagReport] | None = None,
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
    caption_text = _caption_text(captions, tag_reports)
    vision_prefix = f"local vision model caption: {caption_text}; " if caption_text else ""
    analysis_images = _analysis_images(paths, signals, captions, tag_reports)
    body_cues = [image.body_attributes for image in analysis_images]
    aggregate_intelligence = _aggregate_intelligence(analysis_images)
    warnings = _analysis_warnings(signals, captions)
    consistency_score = _consistency_score(signals, captions)

    return ReferenceAnalysisResponse(
        id=request.id,
        display_name=display_name,
        age_category=request.age_category,
        face_summary=_fit_text(
            f"offline image analysis from {count} reference {plural}; {vision_prefix}{aggregate_intelligence.face.summary}; "
            "fictional adult face identity guided by the selected local images; preserve recurring face structure, expression style, "
            "and natural asymmetry without matching any real person",
            600,
        ),
        hair=_fit_text(f"{aggregate_intelligence.hair.summary}; keep color, length, volume, and styling consistent unless manually changed", 240),
        eyes=_fit_text(f"{vision_prefix}reference-inferred eye shape and color; keep gaze and expression style consistent", 160),
        skin_tone=f"{warmth} natural skin tone inferred from local image color balance; keep skin texture believable",
        body_shape=_fit_text(
            f"{aggregate_intelligence.body_shape.summary}; {aggregate_intelligence.waist_hips.summary}; "
            "use these reference proportions as the consistency base for future generations",
            400,
        ),
        chest=_fit_text(
            f"{aggregate_intelligence.chest.summary}; breast visibility: {aggregate_intelligence.adult_content.breast_visibility}; "
            f"nipple/areola visibility: {aggregate_intelligence.adult_content.nipple_areola_visibility}",
            240,
        ),
        grooming=_fit_text(f"{vision_prefix}reference-inferred grooming and presentation; keep details realistic and consistent", 240),
        style_notes=_fit_text(
            f"offline image analysis: {orientation}, {lighting}, {texture}; {vision_prefix}{aggregate_intelligence.prompt_summary}; "
            "reference-guided believable still photo style with natural camera rendering and no overpolished AI look",
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
        aggregate_intelligence=aggregate_intelligence,
    )


def _caption_text(captions: list[str] | None, tag_reports: list[ReferenceTagReport] | None = None) -> str:
    if not captions and not tag_reports:
        return ""
    cleaned = [caption.strip().rstrip(".") for caption in (captions or []) if caption.strip()]
    if tag_reports:
        cleaned.extend(report.reference_text().strip().rstrip(".") for report in tag_reports if report.reference_text().strip())
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
    tag_reports: list[ReferenceTagReport] | None,
) -> list[ReferenceAnalysisImage]:
    images: list[ReferenceAnalysisImage] = []
    for index, (path, signal) in enumerate(zip(paths, signals, strict=True)):
        caption = captions[index].strip() if captions and index < len(captions) and captions[index].strip() else None
        tag_report = tag_reports[index] if tag_reports and index < len(tag_reports) else None
        analysis_text = _analysis_text(caption, tag_report)
        body_attributes = _body_attribute_cues(signal, analysis_text)
        image_details = _image_details(signal, analysis_text, body_attributes)
        adult_content = _adult_content_signals(analysis_text, body_attributes, tag_report)
        images.append(ReferenceAnalysisImage(
            path=str(path),
            file_name=path.name,
            width=signal.width,
            height=signal.height,
            orientation=_single_orientation(signal),
            brightness=_lighting_note([signal]),
            tone=_warmth_note([signal]),
            texture=_texture_note([signal]),
            caption=caption,
            body_attributes=body_attributes,
            image_details=image_details,
            adult_content=adult_content,
            vision_tags=_vision_tags(tag_report),
        ))
    return images


def _analysis_text(caption: str | None, tag_report: ReferenceTagReport | None) -> str | None:
    parts = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if tag_report is not None:
        report_text = tag_report.reference_text().strip()
        if report_text:
            parts.append(report_text)
    return "; ".join(parts) if parts else None


def _vision_tags(tag_report: ReferenceTagReport | None) -> list[VisionTag]:
    if tag_report is None:
        return []
    tags = [
        VisionTag(
            name=_display_tag(getattr(tag, "name", "")),
            confidence=round(float(getattr(tag, "confidence", 0.0)) * 100),
            source=tag_report.source,
        )
        for tag in tag_report.tags[:40]
        if getattr(tag, "name", "")
    ]
    rating_confidence = round(float(tag_report.rating_confidence) * 100)
    if tag_report.rating and tag_report.rating != "unknown":
        tags.insert(0, VisionTag(name=f"rating: {tag_report.rating}", confidence=rating_confidence, source=tag_report.source))
    return tags[:40]


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


def _image_details(signal: ImageSignals, caption: str | None, body_attributes: BodyAttributeCues) -> ReferenceImageDetails:
    caption_text = (caption or "").strip().lower()
    return ReferenceImageDetails(
        face=_face_detail(caption_text),
        hair=_hair_detail(caption_text),
        skin=_skin_detail(signal, caption_text),
        body_shape=_body_shape_detail(caption_text),
        chest=_chest_detail(caption_text, body_attributes),
        waist_hips=_waist_hips_detail(caption_text),
        pose=_pose_detail(caption_text),
        clothing=_clothing_detail(caption_text),
        lighting=_detail("clear", 75, _lighting_note([signal]), "image brightness"),
        camera=_camera_detail(signal, caption_text),
        background=_background_detail(caption_text),
        quality=_detail("clear", 78, _texture_note([signal]), "image texture"),
    )


def _adult_content_signals(
    caption: str | None,
    body_attributes: BodyAttributeCues,
    tag_report: ReferenceTagReport | None = None,
) -> AdultContentSignals:
    caption_text = (caption or "").strip().lower()
    tag_names = _tag_names(tag_report)
    explicit = _contains_any(caption_text, ["nude", "naked", "topless", "vulva", "genital", "nipples", "areola", "explicit", "sex"]) or _contains_tag(
        tag_names, ["rating:explicit", "nude", "topless", "pussy", "vulva", "nipples", "areolae", "sex"]
    )
    breast_visible = _contains_any(caption_text, ["breast", "breasts", "topless", "bare chest", "nude", "naked"]) or _contains_tag(
        tag_names, ["breasts", "large_breasts", "medium_breasts", "small_breasts", "topless", "nude"]
    )
    nipple_visible = _contains_any(caption_text, ["nipple", "nipples", "areola", "areolas"]) or _contains_tag(tag_names, ["nipples", "areolae"])
    genital_visible = _contains_any(caption_text, ["vulva", "genital", "genitals", "pussy", "crotch visible"]) or _contains_tag(
        tag_names, ["pussy", "vulva", "cameltoe"]
    )
    buttocks_visible = _contains_any(caption_text, ["buttocks", "butt", "ass", "rear", "from behind"]) or _contains_tag(
        tag_names, ["ass", "buttocks", "from_behind"]
    )
    sexual_activity = _contains_any(caption_text, ["sex", "sexual", "intercourse", "penetration", "oral", "masturbat"])

    if explicit:
        nudity_level = "explicit nudity"
        confidence = 86
    elif body_attributes.coverage in {"swimwear", "underwear or lingerie"}:
        nudity_level = "non-explicit revealing clothing"
        confidence = body_attributes.confidence
    elif body_attributes.coverage == "clothed":
        nudity_level = "non-explicit clothed"
        confidence = body_attributes.confidence
    else:
        nudity_level = "unclear"
        confidence = 35

    evidence_terms = [
        term
        for term in ["nude", "topless", "breasts", "nipples", "areola", "vulva", "genital", "buttocks", "sex", "bikini", "lingerie"]
        if term in caption_text
    ]
    tag_evidence = _tag_evidence(tag_report)
    if tag_evidence:
        evidence = tag_evidence
    elif evidence_terms:
        evidence = f"caption mentions {', '.join(evidence_terms)}"
    else:
        evidence = body_attributes.evidence
    return AdultContentSignals(
        nudity_level=nudity_level,
        breast_visibility="visible" if breast_visible else _adult_part_fallback(body_attributes, "chest"),
        nipple_areola_visibility="visible" if nipple_visible else "not clearly described",
        genital_visibility="visible" if genital_visible else "not clearly described",
        buttocks_visibility="visible" if buttocks_visible else "not clearly described",
        sexual_activity="explicit sexual pose or activity cue" if sexual_activity else "none described",
        confidence=confidence,
        evidence=_fit_text(evidence, 320),
    )


def _adult_part_fallback(body_attributes: BodyAttributeCues, part: str) -> str:
    if part == "chest" and body_attributes.coverage in {"swimwear", "underwear or lingerie"}:
        return "covered or partially visible"
    return "not clearly described"


def _face_detail(caption: str) -> AttributeDetail:
    if _contains_any(caption, ["face", "portrait", "woman", "girl", "person"]):
        return _detail("partial", 55, "face present but not identity-locked", "caption indicates a person")
    return _detail("unclear", 30, "face not clearly described", "caption lacks face detail")


def _hair_detail(caption: str) -> AttributeDetail:
    if "hair" not in caption:
        return _detail("unclear", 35, "hair not clearly described", "caption lacks hair detail")
    descriptors = [word for word in ["long", "short", "shoulder-length", "dark", "black", "brown", "blonde", "auburn", "wavy", "curly", "straight"] if word in caption]
    if "red hair" in caption or "red-haired" in caption:
        descriptors.append("red")
    summary = " ".join(descriptors + ["hair"]) if descriptors else "hair visible"
    return _detail("clear", 82, summary, "caption mentions hair")


def _skin_detail(signal: ImageSignals, caption: str) -> AttributeDetail:
    tone = _warmth_note([signal])
    visibility = "partial" if _contains_any(caption, ["bikini", "nude", "skin", "shorts", "topless"]) else "unclear"
    confidence = 66 if visibility == "partial" else 45
    return _detail(visibility, confidence, f"{tone} natural skin tone", "image color balance")


def _body_shape_detail(caption: str) -> AttributeDetail:
    if _contains_any(caption, ["full body", "standing", "bikini", "nude"]):
        return _detail("partial", 70, "adult full-body proportions visible", "caption indicates full or revealing body framing")
    return _detail("unclear", 35, "body shape not clearly described", "caption lacks body-shape detail")


def _chest_detail(caption: str, body_attributes: BodyAttributeCues) -> AttributeDetail:
    if _contains_any(caption, ["breast", "breasts", "topless", "bare chest", "nude", "naked"]):
        return _detail("clear", 82, "visible chest anatomy; exact attributes require targeted review", "caption mentions visible chest anatomy")
    if body_attributes.coverage in {"swimwear", "underwear or lingerie"}:
        return _detail("partial", body_attributes.confidence, "swimwear-covered chest/body shape; exact attributes require clearer references", body_attributes.evidence)
    if body_attributes.coverage == "clothed":
        return _detail("covered", body_attributes.confidence, "chest covered by clothing", body_attributes.evidence)
    return _detail("unclear", 35, "chest attributes not clearly described", "caption lacks chest detail")


def _waist_hips_detail(caption: str) -> AttributeDetail:
    if _contains_any(caption, ["full body", "standing", "bikini", "shorts", "from behind"]):
        return _detail("partial", 64, "partial waist and hip cues from body framing", "caption indicates body framing")
    return _detail("unclear", 35, "waist and hip cues not clearly described", "caption lacks waist/hip detail")


def _pose_detail(caption: str) -> AttributeDetail:
    if "standing" in caption and "beach" in caption:
        return _detail("clear", 80, "standing beach pose", "caption mentions standing and beach")
    if "full body" in caption:
        return _detail("clear", 78, "full body or upper body visible", "caption mentions full body")
    if _contains_any(caption, ["full body", "standing", "from behind", "portrait"]):
        return _detail("clear", 76, _pose_framing(ImageSignals(1, 2, 0, 0, 0, 0, 0), caption), "caption mentions pose or framing")
    return _detail("unclear", 35, "pose not clearly described", "caption lacks pose detail")


def _clothing_detail(caption: str) -> AttributeDetail:
    if "red bikini" in caption:
        return _detail("clear", 84, "red bikini", "caption mentions red bikini")
    if "bikini" in caption:
        return _detail("clear", 80, "bikini", "caption mentions bikini")
    if "denim shorts" in caption and "white shirt" in caption:
        return _detail("clear", 78, "white shirt and denim shorts", "caption mentions white shirt and denim shorts")
    if "shirt" in caption:
        return _detail("clear", 68, "shirt", "caption mentions shirt")
    if _contains_any(caption, ["nude", "naked"]):
        return _detail("clear", 82, "no clothing described", "caption mentions nudity")
    return _detail("unclear", 35, "clothing not clearly described", "caption lacks clothing detail")


def _camera_detail(signal: ImageSignals, caption: str) -> AttributeDetail:
    if "full body" in caption:
        return _detail("partial", 68, "full-body framing", "caption mentions full body")
    return _detail("partial", 60, f"{_single_orientation(signal)} frame", "image orientation")


def _background_detail(caption: str) -> AttributeDetail:
    if "beach" in caption:
        return _detail("clear", 82, "beach setting", "caption mentions beach")
    return _detail("unclear", 35, "background not clearly described", "caption lacks background detail")


def _detail(visibility: str, confidence: int, summary: str, evidence: str) -> AttributeDetail:
    return AttributeDetail(
        visibility=visibility,
        confidence=confidence,
        summary=_fit_text(summary, 260),
        evidence=_fit_text(evidence, 260),
    )


def _aggregate_intelligence(images: list[ReferenceAnalysisImage]) -> AggregateReferenceIntelligence:
    details = [image.image_details for image in images]
    adult_signals = [image.adult_content for image in images]
    aggregate = AggregateReferenceIntelligence(
        face=_aggregate_detail(details, "face"),
        hair=_aggregate_detail(details, "hair"),
        skin=_aggregate_detail(details, "skin"),
        body_shape=_aggregate_detail(details, "body_shape"),
        chest=_aggregate_detail(details, "chest"),
        waist_hips=_aggregate_detail(details, "waist_hips"),
        pose=_aggregate_detail(details, "pose"),
        clothing=_aggregate_detail(details, "clothing"),
        lighting=_aggregate_detail(details, "lighting"),
        camera=_aggregate_detail(details, "camera"),
        background=_aggregate_detail(details, "background"),
        quality=_aggregate_detail(details, "quality"),
        adult_content=_aggregate_adult_content(adult_signals),
        strong_tags=_aggregate_strong_tags(images),
        prompt_summary="",
        uncertainty_notes=[],
    )
    aggregate = _upgrade_reference_pack_confidence(aggregate, images)
    prompt_summary = _aggregate_prompt_summary(aggregate)
    uncertainty_notes = _aggregate_uncertainty_notes(aggregate, images)
    return aggregate.model_copy(update={"prompt_summary": prompt_summary, "uncertainty_notes": uncertainty_notes})


def _aggregate_detail(details: list[ReferenceImageDetails], field_name: str) -> AttributeDetail:
    values = [getattr(detail, field_name) for detail in details]
    visibility_rank = {"clear": 5, "partial": 4, "covered": 3, "unclear": 2, "not_visible": 1}
    best = max(values, key=lambda value: (visibility_rank[value.visibility], value.confidence))
    summaries = _unique_phrases([value.summary for value in values if value.summary and value.visibility != "not_visible"])
    evidences = _unique_phrases([value.evidence for value in values if value.evidence])
    summary = summaries[0] if len(summaries) == 1 else "; ".join(summaries[:3])
    return _detail(
        best.visibility,
        max(value.confidence for value in values),
        summary or best.summary,
        "; ".join(evidences[:3]) or best.evidence,
    )


def _upgrade_reference_pack_confidence(
    aggregate: AggregateReferenceIntelligence,
    images: list[ReferenceAnalysisImage],
) -> AggregateReferenceIntelligence:
    reference_count = len(images)
    face_signal_count = _count_images_with_signal(images, "face")
    body_signal_count = _count_body_signal_images(images)
    rear_signal_count = _count_text_matches(images, ["from behind", "rear", "back view", "backshot"])
    side_signal_count = _count_text_matches(images, ["side", "side view", "profile"])
    chest_signal_count = sum(
        1
        for image in images
        if image.image_details.chest.visibility in {"clear", "partial", "covered"}
        or image.adult_content.breast_visibility != "not clearly described"
    )
    tag_names = _aggregate_tag_name_set(aggregate)

    updates: dict[str, AttributeDetail] = {}
    if reference_count >= 4 and face_signal_count >= 3 and aggregate.face.visibility != "clear":
        updates["face"] = _detail(
            "clear",
            max(aggregate.face.confidence, 78),
            f"identity-ready fictional face reference set from {reference_count} images; preserve recurring facial structure and expression style",
            "multiple selected references include face/person cues",
        )

    if reference_count >= 5 and body_signal_count >= 3 and aggregate.body_shape.visibility != "clear":
        updates["body_shape"] = _detail(
            "clear",
            max(aggregate.body_shape.confidence, 78),
            f"body consistency base built from {body_signal_count} usable body references",
            "selected references include repeated full-body, pose, or body framing cues",
        )

    if reference_count >= 5 and body_signal_count >= 3 and aggregate.waist_hips.visibility != "clear":
        evidence_bits = ["front/body framing cues"]
        if rear_signal_count:
            evidence_bits.append("rear-view cues")
        if side_signal_count:
            evidence_bits.append("side-view cues")
        updates["waist_hips"] = _detail(
            "clear",
            max(aggregate.waist_hips.confidence, 76),
            "waist, hip, and lower-body proportions have enough reference coverage for broad consistency",
            "; ".join(evidence_bits),
        )

    if (
        reference_count >= 4
        and chest_signal_count >= 2
        and aggregate.chest.visibility != "clear"
        and (
            aggregate.adult_content.breast_visibility != "not clearly described"
            or _has_any_tag(tag_names, ["breasts", "large breasts", "medium breasts", "small breasts", "bikini", "bra", "underwear", "lingerie"])
        )
    ):
        updates["chest"] = _detail(
            "clear",
            max(aggregate.chest.confidence, 76),
            "chest/body-shape coverage is sufficient for broad character consistency; exact explicit anatomy can still be prompt-controlled",
            "multiple references or local tags include chest, swimwear, underwear, or torso cues",
        )

    return aggregate.model_copy(update=updates) if updates else aggregate


def _aggregate_adult_content(signals: list[AdultContentSignals]) -> AdultContentSignals:
    nudity_rank = {
        "explicit nudity": 4,
        "non-explicit revealing clothing": 3,
        "non-explicit clothed": 2,
        "unclear": 1,
    }
    best_nudity = max(signals, key=lambda signal: (nudity_rank.get(signal.nudity_level, 0), signal.confidence))

    return AdultContentSignals(
        nudity_level=best_nudity.nudity_level,
        breast_visibility=_strongest_adult_signal([signal.breast_visibility for signal in signals]),
        nipple_areola_visibility=_strongest_adult_signal([signal.nipple_areola_visibility for signal in signals]),
        genital_visibility=_strongest_adult_signal([signal.genital_visibility for signal in signals]),
        buttocks_visibility=_strongest_adult_signal([signal.buttocks_visibility for signal in signals]),
        sexual_activity=_strongest_activity_signal([signal.sexual_activity for signal in signals]),
        confidence=max(signal.confidence for signal in signals),
        evidence=_fit_text("; ".join(_unique_phrases([signal.evidence for signal in signals if signal.evidence])[:3]), 320),
    )


def _strongest_adult_signal(values: list[str]) -> str:
    rank = {
        "visible": 5,
        "covered or partially visible": 4,
        "not clearly described": 2,
    }
    return max(values, key=lambda value: rank.get(value, 1))


def _strongest_activity_signal(values: list[str]) -> str:
    if any(value == "explicit sexual pose or activity cue" for value in values):
        return "explicit sexual pose or activity cue"
    return "none described"


def _aggregate_prompt_summary(aggregate: AggregateReferenceIntelligence) -> str:
    tag_summary = ", ".join(tag.name for tag in aggregate.strong_tags[:12])
    parts = [
        aggregate.face.summary,
        aggregate.hair.summary,
        aggregate.skin.summary,
        aggregate.body_shape.summary,
        aggregate.chest.summary,
        aggregate.waist_hips.summary,
        aggregate.pose.summary,
        aggregate.clothing.summary,
        f"adult-content read: {aggregate.adult_content.nudity_level}; breast visibility {aggregate.adult_content.breast_visibility}; "
        f"nipple/areola visibility {aggregate.adult_content.nipple_areola_visibility}; genital visibility {aggregate.adult_content.genital_visibility}; "
        f"buttocks visibility {aggregate.adult_content.buttocks_visibility}",
        f"strong local tags: {tag_summary}" if tag_summary else "",
    ]
    return _fit_text("Prompt-ready reference intelligence: " + "; ".join(part for part in parts if part), 900)


def _aggregate_uncertainty_notes(
    aggregate: AggregateReferenceIntelligence,
    images: list[ReferenceAnalysisImage],
) -> list[str]:
    notes: list[str] = []
    if aggregate.face.visibility != "clear":
        notes.append("Face identity is not fully locked; add clear synthetic face references for stronger consistency.")
    if aggregate.chest.visibility != "clear":
        notes.append("Exact chest anatomy requires clearer uncovered or targeted references.")
    if aggregate.waist_hips.visibility != "clear":
        notes.append("Waist, hip, and lower-body consistency is partial; add front, side, and rear body references.")
    explicit_pack = any(image.adult_content.nudity_level == "explicit nudity" for image in images)
    if explicit_pack and aggregate.adult_content.genital_visibility == "not clearly described":
        notes.append("Explicit references were detected, but no clear genital signal was found; exact anatomy will rely on prompts or targeted references.")
    if explicit_pack and aggregate.adult_content.nipple_areola_visibility == "not clearly described":
        notes.append("Explicit references were detected, but no clear nipple/areola signal was found; exact anatomy will rely on prompts or targeted references.")
    return notes[:5]


def _unique_phrases(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _aggregate_strong_tags(images: list[ReferenceAnalysisImage]) -> list[VisionTag]:
    best_by_name: dict[str, VisionTag] = {}
    for image in images:
        for tag in image.vision_tags:
            current = best_by_name.get(tag.name)
            if current is None or tag.confidence > current.confidence:
                best_by_name[tag.name] = tag
    return sorted(best_by_name.values(), key=lambda tag: tag.confidence, reverse=True)[:60]


def _count_images_with_signal(images: list[ReferenceAnalysisImage], field_name: str) -> int:
    return sum(
        1
        for image in images
        if getattr(image.image_details, field_name).visibility in {"clear", "partial"}
    )


def _count_body_signal_images(images: list[ReferenceAnalysisImage]) -> int:
    return sum(
        1
        for image in images
        if image.image_details.body_shape.visibility in {"clear", "partial"}
        or image.image_details.waist_hips.visibility in {"clear", "partial"}
        or _contains_any(_image_reference_text(image), ["full body", "standing", "from behind", "bikini", "shorts", "underwear", "lingerie"])
    )


def _count_text_matches(images: list[ReferenceAnalysisImage], needles: list[str]) -> int:
    return sum(1 for image in images if _contains_any(_image_reference_text(image), needles))


def _aggregate_tag_name_set(aggregate: AggregateReferenceIntelligence) -> set[str]:
    return {tag.name.casefold().replace("_", " ") for tag in aggregate.strong_tags}


def _has_any_tag(tags: set[str], needles: list[str]) -> bool:
    return any(needle in tags for needle in needles)


def _image_reference_text(image: ReferenceAnalysisImage) -> str:
    parts = [
        image.caption or "",
        image.body_attributes.coverage,
        image.body_attributes.chest_visibility,
        image.body_attributes.pose_framing,
        image.body_attributes.evidence,
        image.adult_content.evidence,
        " ".join(tag.name for tag in image.vision_tags),
    ]
    return " ".join(parts).casefold().replace("_", " ")


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


def _contains_tag(tags: set[str], needles: list[str]) -> bool:
    return any(needle in tags for needle in needles)


def _tag_names(tag_report: ReferenceTagReport | None) -> set[str]:
    if tag_report is None:
        return set()
    names = {getattr(tag, "name", "").casefold() for tag in tag_report.tags if getattr(tag, "name", "")}
    if tag_report.rating:
        names.add(f"rating:{tag_report.rating.casefold()}")
    return names


def _tag_evidence(tag_report: ReferenceTagReport | None) -> str:
    if tag_report is None:
        return ""
    tags = [
        f"{_display_tag(getattr(tag, 'name', ''))} {round(float(getattr(tag, 'confidence', 0.0)) * 100)}%"
        for tag in tag_report.tags[:8]
        if getattr(tag, "name", "")
    ]
    if tag_report.rating:
        tags.insert(0, f"rating {tag_report.rating} {round(float(tag_report.rating_confidence) * 100)}%")
    return _fit_text(f"{tag_report.source}: " + ", ".join(tags), 320) if tags else ""


def _display_tag(tag: str) -> str:
    return tag.replace("_", " ").replace("rating:", "rating: ")


def _first_present(value: str, needles: list[str]) -> str:
    return next((needle for needle in needles if needle in value), needles[0])


def _analysis_warnings(signals: list[ImageSignals], captions: list[str] | None) -> list[str]:
    warnings: list[str] = []
    if len(signals) < 3:
        warnings.append("Add 2-4 more references for stronger identity consistency.")
    orientations = {_single_orientation(signal) for signal in signals}
    if len(orientations) > 1 and len(signals) < 6:
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
