import numpy as np

from src.detection import Detection
from src.tracking import DeepSORTTracker


class FakeExtractor:
    embedding_dim = 8

    def extract(self, frame, bboxes):
        n = len(bboxes)
        feats = np.zeros((n, self.embedding_dim), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            feats[i] = [
                np.sin(cx * 0.01),
                np.cos(cx * 0.01),
                np.sin(cy * 0.01),
                np.cos(cy * 0.01),
                np.sin(cx * 0.005),
                np.cos(cx * 0.005),
                np.sin(cy * 0.005),
                np.cos(cy * 0.005),
            ]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-8)


def make_det(x1, y1, x2, y2):
    return Detection(bbox=(x1, y1, x2, y2), confidence=0.9, class_id=0)


def test_alive_tracks_in_history():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=30, n_init=3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for f in range(1, 6):
        tracker.update(frame, [make_det(100, 100, 200, 300)], frame_idx=f)
    data = tracker.get_all_track_data()
    assert len(data) == 1
    assert 0 in data
    assert data[0]["first_frame"] == 1
    assert data[0]["last_frame"] == 5
    assert data[0]["n_observations"] == 5
    assert data[0]["gallery"].shape[1] == 8


def test_deleted_tracks_in_history():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=2, n_init=3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for f in range(1, 4):
        tracker.update(frame, [make_det(100, 100, 200, 300)], frame_idx=f)
    for f in range(4, 8):
        tracker.update(frame, [], frame_idx=f)
    data = tracker.get_all_track_data()
    assert 0 in data
    assert data[0]["n_observations"] == 3
    assert data[0]["gallery"].shape[0] >= 1


def test_multiple_tracks_history():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=30, n_init=3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for f in range(1, 6):
        dets = [
            make_det(100, 100, 200, 300),
            make_det(500, 100, 600, 300),
        ]
        tracker.update(frame, dets, frame_idx=f)
    data = tracker.get_all_track_data()
    assert len(data) == 2
    for _, d in data.items():
        assert d["first_frame"] == 1
        assert d["last_frame"] == 5
        assert d["n_observations"] == 5
