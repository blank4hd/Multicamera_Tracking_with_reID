"""SORT tracker implementation for single-camera tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import associate_detections_to_tracks
from .kalman import KalmanBoxTracker


@dataclass
class TrackedBox:
    """Single tracked output emitted for a frame."""

    bbox: tuple[float, float, float, float]
    track_id: int
    confidence: float
    frame_idx: int


class SORTTracker:
    """Simple Online and Realtime Tracking (SORT)."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self) -> None:
        """Reset tracker state and ID counter."""
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: list, frame_idx: int | None = None) -> list[TrackedBox]:
        """Update tracker state from detections and return active tracked boxes."""
        self.frame_count += 1
        if frame_idx is None:
            frame_idx = self.frame_count

        if detections:
            det_bboxes = np.asarray([det.bbox for det in detections], dtype=float)
            det_confidences = np.asarray(
                [float(det.confidence) for det in detections],
                dtype=float,
            )
        else:
            det_bboxes = np.zeros((0, 4), dtype=float)
            det_confidences = np.zeros((0,), dtype=float)

        predicted_boxes: list[np.ndarray] = []
        valid_trackers: list[KalmanBoxTracker] = []
        for tracker in self.trackers:
            predicted = tracker.predict()
            if np.all(np.isfinite(predicted)):
                valid_trackers.append(tracker)
                predicted_boxes.append(predicted)

        self.trackers = valid_trackers
        track_bboxes = (
            np.asarray(predicted_boxes, dtype=float)
            if predicted_boxes
            else np.zeros((0, 4), dtype=float)
        )

        matches, unmatched_dets, _ = associate_detections_to_tracks(
            det_bboxes,
            track_bboxes,
            iou_threshold=self.iou_threshold,
        )

        for det_idx, trk_idx in matches:
            self.trackers[int(trk_idx)].update(
                det_bboxes[int(det_idx)],
                confidence=float(det_confidences[int(det_idx)]),
            )

        for det_idx in unmatched_dets:
            tracker = KalmanBoxTracker(det_bboxes[int(det_idx)])
            tracker.last_confidence = float(det_confidences[int(det_idx)])
            self.trackers.append(tracker)

        outputs: list[TrackedBox] = []
        for tracker in self.trackers:
            if tracker.time_since_update == 0 and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                bbox = tracker.get_state()
                outputs.append(
                    TrackedBox(
                        bbox=(
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ),
                        track_id=tracker.id,
                        confidence=float(tracker.last_confidence),
                        frame_idx=int(frame_idx),
                    )
                )

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        outputs.sort(key=lambda t: t.track_id)
        return outputs
