from pathlib import Path

from PIL import Image

from local_model_studio.face_safety import FaceScanResult, OpenCvFaceDetector


def test_face_safety_default_detector_has_stable_interface(tmp_path: Path) -> None:
    image_path = tmp_path / "simple.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image_path)

    result = OpenCvFaceDetector().scan(image_path)

    assert isinstance(result, FaceScanResult)
    assert isinstance(result.face_count, int)
    assert isinstance(result.detector_available, bool)
