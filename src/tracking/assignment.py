"""Data association helpers for SORT tracking."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two xyxy box arrays."""
    a = np.asarray(bboxes_a, dtype=float)
    b = np.asarray(bboxes_b, dtype=float)

    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)

    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.zeros((n, m), dtype=float)
    valid = union > 0.0
    iou[valid] = inter[valid] / union[valid]
    return iou


def associate_detections_to_tracks(
    detections: np.ndarray,
    tracks: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate detection boxes to track boxes using Hungarian assignment."""
    dets = np.asarray(detections, dtype=float)
    trks = np.asarray(tracks, dtype=float)

    m = dets.shape[0]
    n = trks.shape[0]

    if m == 0 and n == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )
    if m == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(n, dtype=int),
        )
    if n == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(m, dtype=int),
            np.empty((0,), dtype=int),
        )

    iou_matrix = iou_batch(dets, trks)
    cost_matrix = -iou_matrix

    det_indices, trk_indices = linear_sum_assignment(cost_matrix)

    matches_list: list[list[int]] = []
    unmatched_detections = set(range(m))
    unmatched_tracks = set(range(n))

    for det_idx, trk_idx in zip(det_indices, trk_indices):
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            continue
        matches_list.append([int(det_idx), int(trk_idx)])
        unmatched_detections.discard(int(det_idx))
        unmatched_tracks.discard(int(trk_idx))

    matches = (
        np.asarray(matches_list, dtype=int)
        if matches_list
        else np.empty((0, 2), dtype=int)
    )
    unmatched_d = np.asarray(sorted(unmatched_detections), dtype=int)
    unmatched_t = np.asarray(sorted(unmatched_tracks), dtype=int)

    return matches, unmatched_d, unmatched_t
