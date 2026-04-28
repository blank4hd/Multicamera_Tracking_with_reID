import pytest

from src.detection import Detection
from src.tracking.sort import SORTTracker


def make_det(x1, y1, x2, y2, conf=0.9):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=0)


def test_single_track_persistence():
    """A detection that stays put should produce a single confirmed track."""
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    track_ids_seen = set()
    # Run 10 frames with the same detection
    for frame_idx in range(1, 11):
        tracks = tracker.update([make_det(100, 100, 200, 300)], frame_idx=frame_idx)
        for t in tracks:
            track_ids_seen.add(t.track_id)
    assert len(track_ids_seen) == 1


def test_two_tracks_no_id_switch():
    """Two well-separated boxes should not swap IDs."""
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    ids_per_box = [set(), set()]
    for frame_idx in range(1, 11):
        dets = [
            make_det(100, 100, 200, 300),  # left person
            make_det(500, 100, 600, 300),  # right person, well separated
        ]
        tracks = tracker.update(dets, frame_idx=frame_idx)
        for t in tracks:
            # Bucket by box position
            cx = (t.bbox[0] + t.bbox[2]) / 2
            ids_per_box[0 if cx < 300 else 1].add(t.track_id)
    # Each box should have produced exactly one track ID
    assert len(ids_per_box[0]) == 1
    assert len(ids_per_box[1]) == 1
    assert ids_per_box[0] != ids_per_box[1]


def test_track_dies_after_max_age():
    """A track should be removed after max_age frames of no updates."""
    tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    # Establish a track
    for f in range(1, 4):
        tracker.update([make_det(100, 100, 200, 300)], frame_idx=f)
    assert len(tracker.trackers) == 1
    # No detections for max_age+1 frames
    for f in range(4, 4 + 7):
        tracker.update([], frame_idx=f)
    assert len(tracker.trackers) == 0


def test_reset():
    tracker = SORTTracker()
    tracker.update([make_det(0, 0, 10, 10)], frame_idx=1)
    tracker.reset()
    assert tracker.frame_count == 0
    assert len(tracker.trackers) == 0
