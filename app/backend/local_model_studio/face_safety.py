from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .training_schemas import FaceScanResult


class FaceDetector(Protocol):
    def scan(self, image_path: Path) -> FaceScanResult:
        ...


class OpenCvFaceDetector:
    def scan(self, image_path: Path) -> FaceScanResult:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return FaceScanResult(
                face_count=0,
                detector_available=False,
                warning="OpenCV face detector unavailable; install opencv-python to enable face scans.",
            )

        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            return FaceScanResult(
                face_count=0,
                detector_available=False,
                warning="OpenCV face detector cascade unavailable.",
            )

        image = cv2.imread(str(image_path))
        if image is None:
            return FaceScanResult(
                face_count=0,
                detector_available=False,
                warning="OpenCV could not read image for face scan.",
            )

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5)
        return FaceScanResult(face_count=len(faces), detector_available=True)
