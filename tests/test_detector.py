import numpy as np
import pytest

from src.detection.detector import Detection, PersonDetector


def test_detection_dataclass_properties() -> None:
    detection = Detection(bbox=(10, 20, 110, 220), confidence=0.9, class_id=0)

    assert detection.xywh == (10, 20, 100, 200)
    assert detection.center == (60, 120)
    assert detection.area == 20000


@pytest.mark.slow
def test_detector_loads() -> None:
    detector = PersonDetector()

    assert hasattr(detector, "model")
    assert detector.conf_threshold == 0.5


@pytest.mark.slow
def test_detector_runs_on_blank_image() -> None:
    detector = PersonDetector()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert isinstance(detections, list)
    assert all(isinstance(detection, Detection) for detection in detections)
