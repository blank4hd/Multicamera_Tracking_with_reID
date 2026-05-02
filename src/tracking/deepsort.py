from collections import deque
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from .assignment import iou_batch
from .kalman import KalmanBoxTracker
from .sort import TrackedBox


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class DeepSORTTrack:
    """
    A single track with motion (Kalman) + appearance (embedding gallery) state.
    """

    def __init__(self, bbox, feature, track_id, n_init=3, max_age=30, gallery_size=30):
        self.kalman = KalmanBoxTracker(bbox)
        self.kalman.id = track_id
        self.track_id = track_id
        self.gallery_size = gallery_size
        self.gallery = deque(maxlen=gallery_size)
        if feature is not None:
            self.gallery.append(np.asarray(feature, dtype=np.float32).reshape(-1))
        self.last_confidence = 1.0
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.n_init = n_init
        self.max_age = max_age
        self.time_since_update = 0
        self.age = 0

    def predict(self):
        bbox = self.kalman.predict()
        self.time_since_update = self.kalman.time_since_update
        self.age = self.kalman.age
        return bbox

    def update(self, bbox, feature, confidence: float = 1.0):
        self.kalman.update(np.asarray(bbox), confidence=confidence)
        self.last_confidence = confidence
        if feature is not None:
            self.gallery.append(np.asarray(feature, dtype=np.float32).reshape(-1))
        self.time_since_update = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.kalman.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        return self.state == TrackState.DELETED

    def is_tentative(self):
        return self.state == TrackState.TENTATIVE

    def get_state(self):
        return self.kalman.get_state()

    def appearance_distance(self, features: np.ndarray) -> np.ndarray:
        if len(self.gallery) == 0 or len(features) == 0:
            return np.full(len(features), 1e6, dtype=np.float32)
        gallery = np.stack(list(self.gallery), axis=0)
        sim = features @ gallery.T
        dist = 1.0 - sim
        return dist.min(axis=1).astype(np.float32)


def _gated_appearance_cost(
    tracks: list,
    track_indices: list[int],
    detections_features: np.ndarray,
    detections_bboxes: np.ndarray,
    predicted_bboxes: np.ndarray,
    detection_indices: list[int],
    appearance_threshold: float,
    iou_gate_threshold: float,
) -> np.ndarray:
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)

    track_bboxes_subset = predicted_bboxes[track_indices]
    det_bboxes_subset = detections_bboxes[detection_indices]
    det_features_subset = detections_features[detection_indices]

    iou_matrix = iou_batch(track_bboxes_subset, det_bboxes_subset)

    cost = np.full((len(track_indices), len(detection_indices)), 1e6, dtype=np.float32)
    for ti_local, ti_global in enumerate(track_indices):
        track = tracks[ti_global]
        app_dist = track.appearance_distance(det_features_subset)
        for di_local in range(len(detection_indices)):
            if iou_matrix[ti_local, di_local] < iou_gate_threshold:
                continue
            if app_dist[di_local] > appearance_threshold:
                continue
            cost[ti_local, di_local] = app_dist[di_local]

    return cost


def _hungarian_match(cost_matrix: np.ndarray, max_cost: float = 1e5):
    if cost_matrix.size == 0:
        return (
            np.zeros((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < max_cost:
            matches.append([r, c])
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    matched_rows = set(matches[:, 0].tolist()) if len(matches) else set()
    matched_cols = set(matches[:, 1].tolist()) if len(matches) else set()
    unmatched_rows = np.array([r for r in range(cost_matrix.shape[0]) if r not in matched_rows], dtype=int)
    unmatched_cols = np.array([c for c in range(cost_matrix.shape[1]) if c not in matched_cols], dtype=int)
    return matches, unmatched_rows, unmatched_cols


class DeepSORTTracker:
    def __init__(
        self,
        feature_extractor=None,
        max_age: int = 30,
        n_init: int = 3,
        iou_gate_threshold: float = 0.0,
        appearance_threshold: float = 0.2,
        iou_threshold_fallback: float = 0.3,
        gallery_size: int = 30,
    ):
        self.feature_extractor = feature_extractor
        self.max_age = max_age
        self.n_init = n_init
        self.iou_gate_threshold = iou_gate_threshold
        self.appearance_threshold = appearance_threshold
        self.iou_threshold_fallback = iou_threshold_fallback
        self.gallery_size = gallery_size
        self.tracks: list[DeepSORTTrack] = []
        self.frame_count = 0
        self._next_id = 0
        self._track_history: dict[int, dict] = {}

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        self._next_id = 0
        self._track_history = {}
        KalmanBoxTracker.count = 0

    def _new_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, frame, detections, frame_idx: int | None = None) -> list[TrackedBox]:
        self.frame_count += 1
        if frame_idx is None:
            frame_idx = self.frame_count

        if detections:
            det_bboxes = np.asarray([det.bbox for det in detections], dtype=np.float32)
            det_confs = np.asarray([det.confidence for det in detections], dtype=np.float32)
        else:
            det_bboxes = np.zeros((0, 4), dtype=np.float32)
            det_confs = np.zeros((0,), dtype=np.float32)

        if self.feature_extractor is not None and len(detections) > 0:
            det_features = self.feature_extractor.extract(frame, det_bboxes)
        else:
            embedding_dim = getattr(self.feature_extractor, "embedding_dim", 256)
            det_features = np.zeros((len(detections), embedding_dim), dtype=np.float32)

        predicted_bboxes = np.zeros((len(self.tracks), 4), dtype=np.float32)
        for i, track in enumerate(self.tracks):
            predicted_bboxes[i] = track.predict()

        valid_mask = np.isfinite(predicted_bboxes).all(axis=1)
        invalid_indices = np.where(~valid_mask)[0].tolist()
        for index in invalid_indices:
            self.tracks[index].state = TrackState.DELETED
        valid_indices = np.where(valid_mask)[0].tolist()

        confirmed_indices = [i for i in valid_indices if self.tracks[i].is_confirmed()]
        tentative_indices = [i for i in valid_indices if self.tracks[i].is_tentative()]

        unmatched_dets = list(range(len(detections)))
        all_matches: list[tuple[int, int]] = []

        if self.feature_extractor is not None and len(unmatched_dets) > 0:
            for level in range(self.max_age + 1):
                if len(unmatched_dets) == 0:
                    break
                level_track_indices = [i for i in confirmed_indices if self.tracks[i].time_since_update == level]
                if len(level_track_indices) == 0:
                    continue

                cost = _gated_appearance_cost(
                    self.tracks,
                    level_track_indices,
                    det_features,
                    det_bboxes,
                    predicted_bboxes,
                    unmatched_dets,
                    self.appearance_threshold,
                    self.iou_gate_threshold,
                )
                matches, _, _ = _hungarian_match(cost, max_cost=1e5)

                for r_local, c_local in matches:
                    global_track = level_track_indices[r_local]
                    global_det = unmatched_dets[c_local]
                    all_matches.append((global_track, global_det))

                matched_dets_this_level = {unmatched_dets[c_local] for _, c_local in matches}
                unmatched_dets = [d for d in unmatched_dets if d not in matched_dets_this_level]

        matched_track_global = {t for t, _ in all_matches}
        cascade_remaining_tracks = [i for i in confirmed_indices if i not in matched_track_global]
        stage2_track_indices = tentative_indices + cascade_remaining_tracks

        if len(stage2_track_indices) > 0 and len(unmatched_dets) > 0:
            iou_mat = iou_batch(predicted_bboxes[stage2_track_indices], det_bboxes[unmatched_dets])
            cost = 1.0 - iou_mat
            cost[iou_mat < self.iou_threshold_fallback] = 1e6
            matches, _, _ = _hungarian_match(cost, max_cost=1e5)
            for r_local, c_local in matches:
                global_track = stage2_track_indices[r_local]
                global_det = unmatched_dets[c_local]
                all_matches.append((global_track, global_det))
            matched_dets_this_stage = {unmatched_dets[c_local] for _, c_local in matches}
            unmatched_dets = [d for d in unmatched_dets if d not in matched_dets_this_stage]

        matched_track_global = {t for t, _ in all_matches}
        for tidx, didx in all_matches:
            track = self.tracks[tidx]
            track.update(
                det_bboxes[didx],
                det_features[didx] if self.feature_extractor is not None else None,
                confidence=float(det_confs[didx]),
            )

        for tidx in valid_indices:
            if tidx not in matched_track_global:
                self.tracks[tidx].mark_missed()

        for didx in unmatched_dets:
            new_track = DeepSORTTrack(
                bbox=det_bboxes[didx],
                feature=det_features[didx] if self.feature_extractor is not None else None,
                track_id=self._new_track_id(),
                n_init=self.n_init,
                max_age=self.max_age,
                gallery_size=self.gallery_size,
            )
            new_track.last_confidence = float(det_confs[didx])
            self.tracks.append(new_track)
            new_track._first_frame = self.frame_count

        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        for t in deleted_tracks:
            tid = int(t.track_id)
            if len(t.gallery) > 0 and tid not in self._track_history:
                self._track_history[tid] = {
                    "gallery": np.stack(list(t.gallery), axis=0).astype(np.float32),
                    "first_frame": getattr(t, "_first_frame", self.frame_count),
                    "last_frame": self.frame_count,
                    "n_observations": int(t.hits),
                }

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        outputs = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                bbox = track.get_state()
                outputs.append(
                    TrackedBox(
                        bbox=tuple(float(v) for v in bbox),
                        track_id=int(track.track_id),
                        confidence=float(track.last_confidence),
                        frame_idx=frame_idx,
                    )
                )
        outputs.sort(key=lambda t: t.track_id)
        return outputs

    def get_track_galleries(self) -> dict[int, np.ndarray]:
        """
        Return a snapshot of each currently-alive track's gallery embeddings.

        Returns:
            Dict mapping track_id -> (G, D) numpy array of L2-normalized embeddings,
            where G is the current gallery size (1 to gallery_size, depending on
            track age). Includes both confirmed and tentative tracks.
        """
        result = {}
        for track in self.tracks:
            if len(track.gallery) > 0:
                result[int(track.track_id)] = np.stack(list(track.gallery), axis=0).astype(np.float32)
        return result

    def get_all_track_data(self) -> dict[int, dict]:
        """
        Return data for ALL tracks ever created during this run:
        both currently-alive tracks AND tracks that have been deleted.

        Returns:
            Dict mapping track_id -> {
                "gallery": (G, D) numpy array of L2-normalized embeddings,
                "first_frame": int (1-indexed frame when track was born),
                "last_frame": int (1-indexed frame when track was last updated),
                "n_observations": int,
            }
        """
        result = {}
        for tid, data in self._track_history.items():
            result[tid] = dict(data)
        for track in self.tracks:
            tid = int(track.track_id)
            if len(track.gallery) > 0:
                result[tid] = {
                    "gallery": np.stack(list(track.gallery), axis=0).astype(np.float32),
                    "first_frame": getattr(track, "_first_frame", 1),
                    "last_frame": self.frame_count,
                    "n_observations": int(track.hits),
                }
        return result
