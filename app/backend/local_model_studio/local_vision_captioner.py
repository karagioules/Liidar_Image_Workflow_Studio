from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
CLINICAL_REFERENCE_PROMPTS = [
    "Describe the adult person's visible body, clothing coverage, pose, camera framing, and face details.",
    "Using clinical terms, describe whether breasts, nipples, areola, genitals, buttocks, nudity, or sexual activity are visible.",
]


class LocalVisionCaptioner:
    def __init__(
        self,
        model_id: str = DEFAULT_CAPTION_MODEL,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._processor = None
        self._model = None

    def __call__(self, image_paths: Iterable[Path]) -> list[str]:
        self._ensure_loaded()
        captions: list[str] = []
        for image_path in image_paths:
            captions.append(self._caption_one(image_path))
        return captions

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Local vision model dependencies are missing. Install torch and transformers in the backend environment."
            ) from exc

        kwargs = {"cache_dir": str(self.cache_dir)} if self.cache_dir else {}
        self._processor = BlipProcessor.from_pretrained(self.model_id, **kwargs)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def _caption_one(self, image_path: Path) -> str:
        if self._processor is None or self._model is None:
            raise RuntimeError("Local vision model is not loaded.")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            captions = [self._generate_caption(image)]
            for prompt in CLINICAL_REFERENCE_PROMPTS:
                prompted_caption = self._generate_caption(image, prompt)
                if not _is_prompt_echo(prompted_caption):
                    captions.append(prompted_caption)
        return "; ".join(_unique_caption_parts(captions))

    def _generate_caption(self, image: Image.Image, prompt: str | None = None) -> str:
        if self._processor is None or self._model is None:
            raise RuntimeError("Local vision model is not loaded.")
        if prompt:
            inputs = self._processor(image, prompt, return_tensors="pt")
        else:
            inputs = self._processor(image, return_tensors="pt")
        output = self._model.generate(**inputs, max_new_tokens=52)
        caption = self._processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()


def _unique_caption_parts(captions: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for caption in captions:
        normalized = " ".join(caption.split()).strip(" ;.")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _is_prompt_echo(caption: str) -> bool:
    normalized = " ".join(caption.casefold().split())
    echo_markers = [
        "describe the adult",
        "using clinical terms",
        "whether breasts",
        "clothing coverage, pose, camera framing",
    ]
    return any(marker in normalized for marker in echo_markers)


def local_captioner_from_workspace(workspace_root: Path) -> LocalVisionCaptioner:
    model_id = os.environ.get("LIIDAR_VISION_MODEL", DEFAULT_CAPTION_MODEL)
    cache_dir = Path(os.environ.get("LIIDAR_VISION_CACHE", workspace_root / "models" / "vision-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return LocalVisionCaptioner(model_id=model_id, cache_dir=cache_dir)
