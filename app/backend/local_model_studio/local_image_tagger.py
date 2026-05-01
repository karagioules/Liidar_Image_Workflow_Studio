from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_TAGGER_MODEL = "SmilingWolf/wd-vit-large-tagger-v3"
TAGGER_MODEL_FILE = "model.onnx"
TAGGER_TAGS_FILE = "selected_tags.csv"
TAG_THRESHOLD = 0.30
REFERENCE_TAGS = {
    "rating:safe",
    "rating:questionable",
    "rating:explicit",
    "nude",
    "topless",
    "breasts",
    "large_breasts",
    "medium_breasts",
    "small_breasts",
    "nipples",
    "areolae",
    "pussy",
    "vulva",
    "cameltoe",
    "ass",
    "buttocks",
    "from_behind",
    "spread_legs",
    "sex",
    "oral",
    "cum",
    "masturbation",
    "bikini",
    "bra",
    "panties",
    "underwear",
    "lingerie",
    "shorts",
    "shirt",
    "denim",
    "looking_at_viewer",
    "standing",
    "full_body",
    "solo",
    "long_hair",
    "short_hair",
    "brown_hair",
    "black_hair",
    "blonde_hair",
    "wavy_hair",
    "curly_hair",
}


@dataclass(frozen=True)
class TaggerTag:
    name: str
    confidence: float


@dataclass(frozen=True)
class ImageTagReport:
    tags: list[TaggerTag]
    rating: str
    rating_confidence: float
    source: str

    def reference_text(self) -> str:
        tag_text = ", ".join(tag.name for tag in self.tags[:40])
        return f"{self.source} rating {self.rating}; local visual tags: {tag_text}".strip()


@dataclass(frozen=True)
class _ModelTag:
    name: str
    category: int


class LocalImageTagger:
    def __init__(
        self,
        model_id: str = DEFAULT_TAGGER_MODEL,
        cache_dir: Path | None = None,
        threshold: float = TAG_THRESHOLD,
        provider_preference: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.provider_preference = provider_preference
        self._session = None
        self._input_name = ""
        self._input_size = 448
        self._active_provider = "unknown"
        self._tags: list[_ModelTag] = []

    def __call__(self, image_paths: Iterable[Path]) -> list[ImageTagReport]:
        self._ensure_loaded()
        return [self._tag_one(path) for path in image_paths]

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Local tagger dependencies are missing. Install onnxruntime and huggingface_hub in the backend environment."
            ) from exc

        kwargs = {"cache_dir": str(self.cache_dir)} if self.cache_dir else {}
        model_path = hf_hub_download(self.model_id, TAGGER_MODEL_FILE, **kwargs)
        tags_path = hf_hub_download(self.model_id, TAGGER_TAGS_FILE, **kwargs)
        self._tags = _load_tags(Path(tags_path))
        providers = _preferred_providers(ort.get_available_providers(), self.provider_preference)
        session_options = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self._active_provider = self._session.get_providers()[0]
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape
        if isinstance(shape[1], int):
            self._input_size = shape[1]
        elif isinstance(shape[2], int):
            self._input_size = shape[2]

    def _tag_one(self, image_path: Path) -> ImageTagReport:
        if self._session is None:
            raise RuntimeError("Local image tagger is not loaded.")
        image = _prepare_image(image_path, self._input_size)
        raw_output = self._session.run(None, {self._input_name: image})[0][0]
        scored_tags = [
            TaggerTag(name=model_tag.name, confidence=float(score))
            for model_tag, score in zip(self._tags, raw_output, strict=False)
        ]
        rating_tags = [tag for tag in scored_tags if tag.name.startswith("rating:")]
        best_rating = max(rating_tags, key=lambda tag: tag.confidence, default=TaggerTag("rating:unknown", 0.0))
        reference_tags = [
            tag
            for tag in scored_tags
            if (tag.confidence >= self.threshold and (tag.name in REFERENCE_TAGS or _is_body_or_adult_tag(tag.name)))
        ]
        reference_tags.sort(key=lambda tag: tag.confidence, reverse=True)
        return ImageTagReport(
            tags=reference_tags[:80],
            rating=best_rating.name.replace("rating:", ""),
            rating_confidence=best_rating.confidence,
            source=f"wd tagger {self.model_id} via {self._active_provider}",
        )


def local_tagger_from_workspace(workspace_root: Path) -> LocalImageTagger:
    model_id = os.environ.get("LIIDAR_TAGGER_MODEL", DEFAULT_TAGGER_MODEL)
    cache_dir = Path(os.environ.get("LIIDAR_TAGGER_CACHE", workspace_root / "models" / "tagger-cache"))
    threshold = float(os.environ.get("LIIDAR_TAGGER_THRESHOLD", str(TAG_THRESHOLD)))
    provider_preference = os.environ.get("LIIDAR_TAGGER_PROVIDER", "auto")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return LocalImageTagger(model_id=model_id, cache_dir=cache_dir, threshold=threshold, provider_preference=provider_preference)


def _preferred_providers(available_providers: list[str], preference: str) -> list[str]:
    normalized = preference.strip().casefold()
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    providers: list[str] = []
    if normalized in {"auto", "gpu", "directml", "dml"} and "DmlExecutionProvider" in available_providers:
        providers.append("DmlExecutionProvider")
    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available_providers
    return providers


def _load_tags(tags_path: Path) -> list[_ModelTag]:
    with tags_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            _ModelTag(name=row["name"], category=int(row.get("category", "0") or 0))
            for row in reader
            if row.get("name")
        ]


def _prepare_image(image_path: Path, input_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        canvas = canvas.resize((input_size, input_size), Image.Resampling.BICUBIC)
        array = np.asarray(canvas, dtype=np.float32)
    return array[None, :, :, :]


def _is_body_or_adult_tag(tag: str) -> bool:
    parts = [
        "breast",
        "nipple",
        "areola",
        "pussy",
        "vulva",
        "genital",
        "nude",
        "topless",
        "ass",
        "butt",
        "sex",
        "oral",
        "panties",
        "bikini",
        "lingerie",
        "underwear",
        "hair",
        "body",
        "thigh",
        "hip",
        "waist",
    ]
    return any(part in tag for part in parts)
