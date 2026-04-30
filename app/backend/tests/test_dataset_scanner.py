from pathlib import Path

import pytest
from PIL import Image

from local_model_studio.dataset_scanner import scan_dataset
from local_model_studio.face_safety import FaceScanResult
from local_model_studio.training_schemas import DatasetScanRequest


class NoFaceDetector:
    def scan(self, image_path: Path) -> FaceScanResult:
        return FaceScanResult(face_count=0, detector_available=True)


class FaceDetector:
    def scan(self, image_path: Path) -> FaceScanResult:
        return FaceScanResult(face_count=1, detector_available=True)


class UnavailableDetector:
    def scan(self, image_path: Path) -> FaceScanResult:
        return FaceScanResult(
            face_count=0,
            detector_available=False,
            warning="face detector unavailable",
        )


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), color).save(path)


def request(source_folder: Path, dataset_type: str = "style", **kwargs: object) -> DatasetScanRequest:
    return DatasetScanRequest(
        name="training set",
        source_folder=source_folder,
        dataset_type=dataset_type,
        **kwargs,
    )


def test_scan_accepts_unique_face_free_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "one.jpg", (255, 0, 0))
    make_image(source / "two.png", (0, 255, 0))
    (source / "notes.txt").write_text("ignored", encoding="utf-8")

    report = scan_dataset(
        request(source, "style", tags=["editorial"]),
        output_root=tmp_path / "datasets",
        detector=NoFaceDetector(),
    )

    assert report.accepted_count == 2
    assert report.rejected_count == 0
    assert report.ignored_count == 1
    assert all("training set" not in image.caption for image in report.accepted)
    assert all("style" in image.caption for image in report.accepted)
    assert all(Path(image.stored_path or "").with_suffix(".txt").exists() for image in report.accepted)


def test_scan_counts_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "one.jpg", (255, 0, 0))
    (source / "copy.jpg").write_bytes((source / "one.jpg").read_bytes())

    report = scan_dataset(
        request(source, "pose"),
        output_root=tmp_path / "datasets",
        detector=NoFaceDetector(),
    )

    assert report.accepted_count == 1
    assert report.duplicate_count == 1


def test_scan_rejects_detected_faces(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "face.jpg", (255, 0, 0))

    report = scan_dataset(
        request(source, "style"),
        output_root=tmp_path / "datasets",
        detector=FaceDetector(),
    )

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert "face" in report.rejected[0].reason
    assert Path(report.rejected[0].stored_path or "").exists()


def test_body_part_crop_policy_is_invalid_for_style_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "face.jpg", (255, 0, 0))

    with pytest.raises(ValueError, match="body_part_crops_only"):
        scan_dataset(
            request(source, "style", face_policy="body_part_crops_only"),
            output_root=tmp_path / "datasets",
            detector=FaceDetector(),
        )


def test_body_shape_rejects_detected_faces_with_default_policy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "face.jpg", (255, 0, 0))

    report = scan_dataset(
        request(source, "body_shape"),
        output_root=tmp_path / "datasets",
        detector=FaceDetector(),
    )

    assert report.accepted_count == 0
    assert report.rejected_count == 1


def test_unavailable_detector_fails_closed_for_style_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "one.jpg", (255, 0, 0))

    with pytest.raises(ValueError, match="detector unavailable"):
        scan_dataset(
            request(source, "style"),
            output_root=tmp_path / "datasets",
            detector=UnavailableDetector(),
        )


def test_body_part_can_scan_when_detector_unavailable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "crop.jpg", (255, 0, 0))

    report = scan_dataset(
        request(source, "body_part"),
        output_root=tmp_path / "datasets",
        detector=UnavailableDetector(),
    )

    assert report.accepted_count == 1
    assert report.warnings == ["face detector unavailable"]


def test_body_part_crop_policy_can_scan_when_detector_unavailable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "crop.jpg", (255, 0, 0))

    report = scan_dataset(
        request(source, "body_part", face_policy="body_part_crops_only"),
        output_root=tmp_path / "datasets",
        detector=UnavailableDetector(),
    )

    assert report.accepted_count == 1
    assert report.rejected_count == 0


def test_fictional_face_identity_requires_character_and_rights(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "face.jpg", (255, 0, 0))

    with pytest.raises(ValueError, match="character_id"):
        scan_dataset(
            request(source, "fictional_face_identity"),
            output_root=tmp_path / "datasets",
            detector=FaceDetector(),
        )


def test_fictional_face_identity_accepts_faces_with_synthetic_rights(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_image(source / "face.jpg", (255, 0, 0))

    report = scan_dataset(
        request(
            source,
            "fictional_face_identity",
            character_id="character-123",
            source_rights="synthetic",
        ),
        output_root=tmp_path / "datasets",
        detector=FaceDetector(),
    )

    assert report.accepted_count == 1
    assert report.accepted[0].caption == "fictional adult character face reference, synthetic source, character:character-123"
