from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class LocalCropDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]
    provider: str


class LocalCropDetectorUnavailable(RuntimeError):
    pass


class LocalNudeNetCropDetector:
    def __init__(self, *, inference_resolution: int = 320) -> None:
        self.inference_resolution = inference_resolution
        self._session = None
        self._input_name: str | None = None
        self._provider = "unloaded"
        self._lock = Lock()

    @property
    def provider(self) -> str:
        self._ensure_loaded()
        return self._provider

    def detect(self, image_path: Path) -> list[LocalCropDetection]:
        self._ensure_loaded()
        try:
            from nudenet import nudenet as nudenet_impl
        except Exception as exc:
            raise LocalCropDetectorUnavailable("NudeNet is not installed in the backend environment.") from exc

        if self._session is None or self._input_name is None:
            raise LocalCropDetectorUnavailable("Local NudeNet detector did not initialize.")

        try:
            (
                preprocessed_image,
                x_pad,
                y_pad,
                x_ratio,
                y_ratio,
                image_original_width,
                image_original_height,
            ) = _read_image_metadata(str(image_path), self.inference_resolution, nudenet_impl)
            outputs = self._session.run(None, {self._input_name: preprocessed_image})
            detections = nudenet_impl._postprocess(
                outputs,
                x_pad,
                y_pad,
                x_ratio,
                y_ratio,
                image_original_width,
                image_original_height,
                self.inference_resolution,
                self.inference_resolution,
            )
        except Exception as exc:
            raise LocalCropDetectorUnavailable(f"Local NudeNet detection failed: {exc}") from exc

        parsed: list[LocalCropDetection] = []
        for detection in detections:
            label = str(detection.get("class", ""))
            score = float(detection.get("score", 0.0))
            raw_box = detection.get("box")
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                continue
            try:
                x, y, width, height = [int(value) for value in raw_box]
            except (TypeError, ValueError):
                continue
            if width <= 0 or height <= 0:
                continue
            parsed.append(LocalCropDetection(label=label, score=score, box=(x, y, width, height), provider=self._provider))
        return parsed

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                import nudenet
            except Exception as exc:
                raise LocalCropDetectorUnavailable("NudeNet and ONNX Runtime are required for local AI crop scanning.") from exc

            model_path = Path(nudenet.__file__).with_name("320n.onnx")
            if not model_path.is_file():
                raise LocalCropDetectorUnavailable(f"NudeNet model file is missing: {model_path}")

            available = ort.get_available_providers()
            providers = _preferred_providers(available)
            session_options = ort.SessionOptions()
            if "DmlExecutionProvider" in providers:
                session_options.enable_mem_pattern = False
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                self._session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
            except Exception as exc:
                raise LocalCropDetectorUnavailable(f"Unable to start local NudeNet detector: {exc}") from exc

            model_inputs = self._session.get_inputs()
            self._input_name = model_inputs[0].name
            self._provider = self._session.get_providers()[0] if self._session.get_providers() else "unknown"


def _preferred_providers(available: list[str]) -> list[str]:
    ordered = [provider for provider in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
    return ordered or available


def _read_image_metadata(image_path: str, inference_resolution: int, nudenet_impl):
    (
        preprocessed_image,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    ) = nudenet_impl._read_image(image_path, inference_resolution)
    return preprocessed_image, x_pad, y_pad, x_ratio, y_ratio, image_original_width, image_original_height


_DETECTOR: LocalNudeNetCropDetector | None = None
_DETECTOR_LOCK = Lock()


def local_crop_detector() -> LocalNudeNetCropDetector:
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = LocalNudeNetCropDetector()
        return _DETECTOR
