import numpy as np

from src.detection import Detection
from src.tracking import DeepSORTTracker


def make_det(x1, y1, x2, y2, conf=0.9):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=0)


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
        norms = np.maximum(norms, 1e-8)
        return feats / norms


def test_single_track_persistence():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=30, n_init=3, iou_gate_threshold=0.0, appearance_threshold=0.5)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    seen_ids = set()
    for fidx in range(1, 11):
        x_offset = fidx
        tracks = tracker.update(frame, [make_det(100 + x_offset, 100, 200 + x_offset, 300)], frame_idx=fidx)
        for t in tracks:
            seen_ids.add(t.track_id)
    assert len(seen_ids) == 1, f"Expected 1 ID, got {seen_ids}"


def test_two_tracks_no_id_switch():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=30, n_init=3, iou_gate_threshold=0.0, appearance_threshold=0.5)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ids_left, ids_right = set(), set()
    for fidx in range(1, 11):
        dets = [make_det(100, 100, 200, 300), make_det(900, 100, 1000, 300)]
        tracks = tracker.update(frame, dets, frame_idx=fidx)
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) / 2
            (ids_left if cx < 600 else ids_right).add(t.track_id)
    assert len(ids_left) == 1
    assert len(ids_right) == 1
    assert ids_left != ids_right


def test_tentative_track_dropped_quickly():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=30, n_init=3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker.update(frame, [make_det(100, 100, 200, 300)], frame_idx=1)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].is_tentative()
    tracker.update(frame, [], frame_idx=2)
    assert len(tracker.tracks) == 0


def test_confirmed_track_survives_brief_occlusion():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor(), max_age=10, n_init=3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for f in range(1, 4):
        tracker.update(frame, [make_det(100, 100, 200, 300)], frame_idx=f)
    assert tracker.tracks[0].is_confirmed()
    for f in range(4, 9):
        tracker.update(frame, [], frame_idx=f)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].is_confirmed()
    tracker.update(frame, [make_det(105, 100, 205, 300)], frame_idx=9)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 0


def test_motion_only_mode():
    tracker = DeepSORTTracker(feature_extractor=None, max_age=30, n_init=3, iou_threshold_fallback=0.3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    seen_ids = set()
    for fidx in range(1, 11):
        x_offset = fidx
        tracks = tracker.update(frame, [make_det(100 + x_offset, 100, 200 + x_offset, 300)], frame_idx=fidx)
        for t in tracks:
            seen_ids.add(t.track_id)
    assert len(seen_ids) == 1


def test_reset():
    tracker = DeepSORTTracker(feature_extractor=FakeExtractor())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker.update(frame, [make_det(100, 100, 200, 300)], frame_idx=1)
    tracker.reset()
    assert tracker.frame_count == 0
    assert len(tracker.tracks) == 0
    assert tracker._next_id == 0