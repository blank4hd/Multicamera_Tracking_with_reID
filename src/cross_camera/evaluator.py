"""Cross-camera identity evaluation against Wildtrack ground truth."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .wildtrack_io import WildtrackBBox, load_all_annotations


@dataclass
class Prediction:
    """A single per-(camera, frame) bbox emission with its predicted global ID."""

    camera_id: int
    frame_idx: int
    global_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    local_track_id: int = -1


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Single-pair IoU, xyxy."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)


def load_predictions_from_mot_files(
    per_camera_dir: str | Path,
    assignments: dict[tuple[int, int], int],
    num_cameras: int = 7,
) -> list[Prediction]:
    """
    Read per-camera MOT prediction files (C1.txt ... C7.txt) and produce a flat list
    of Prediction objects with global IDs from the matcher assignments.

    MOT format: <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1
    Note: id in MOT files was written 1-indexed (offset +1 by write_mot_results).
    The matcher assignments use the original 0-indexed track_ids.
    So when reading the MOT file, we need to subtract 1 from id to recover the
    original local_track_id used in the assignments dict.
    """

    per_camera_dir = Path(per_camera_dir)
    predictions: list[Prediction] = []
    for cam_idx in range(num_cameras):
        path = per_camera_dir / f"C{cam_idx + 1}.txt"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                frame_idx = int(parts[0])
                mot_id = int(parts[1])
                local_id = mot_id - 1
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                key = (cam_idx, local_id)
                if key not in assignments:
                    continue
                gid = assignments[key]
                predictions.append(
                    Prediction(
                        camera_id=cam_idx,
                        frame_idx=frame_idx,
                        global_id=int(gid),
                        x1=x,
                        y1=y,
                        x2=x + w,
                        y2=y + h,
                        local_track_id=local_id,
                    )
                )
    return predictions


def match_predictions_to_gt(
    predictions: list[Prediction],
    gt_bboxes: list[WildtrackBBox],
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[float]]:
    """
    For each prediction, find the best-matching GT bbox in the same (camera, frame)
    with IoU >= threshold. Return parallel lists:
      pred_to_gt_pid: list of length len(predictions); the matched GT person_id, or -1 if no match.
      pred_to_gt_iou: list of length len(predictions); the IoU score (0.0 if no match).

    Per (camera, frame), we use Hungarian on a local IoU cost matrix to ensure
    one-to-one matching (so two preds in the same cell don't both grab the same GT).
    """

    gt_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for gi, g in enumerate(gt_bboxes):
        gt_by_cell[(g.camera_id, g.frame_idx)].append(gi)

    pred_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pi, p in enumerate(predictions):
        pred_by_cell[(p.camera_id, p.frame_idx)].append(pi)

    pred_to_gt_pid = [-1] * len(predictions)
    pred_to_gt_iou = [0.0] * len(predictions)

    for cell, pred_indices in pred_by_cell.items():
        gt_indices = gt_by_cell.get(cell, [])
        if not gt_indices:
            continue

        n_p, n_g = len(pred_indices), len(gt_indices)
        iou_mat = np.zeros((n_p, n_g), dtype=np.float32)
        for i, pi in enumerate(pred_indices):
            p = predictions[pi]
            for j, gi in enumerate(gt_indices):
                g = gt_bboxes[gi]
                iou_mat[i, j] = _bbox_iou((p.x1, p.y1, p.x2, p.y2), (g.x1, g.y1, g.x2, g.y2))

        cost = -iou_mat
        row_idx, col_idx = linear_sum_assignment(cost)
        for ri, ci in zip(row_idx, col_idx):
            iou = iou_mat[ri, ci]
            if iou < iou_threshold:
                continue
            pi = pred_indices[ri]
            gi = gt_indices[ci]
            pred_to_gt_pid[pi] = gt_bboxes[gi].person_id
            pred_to_gt_iou[pi] = float(iou)

    return pred_to_gt_pid, pred_to_gt_iou


def compute_identity_metrics(
    predictions: list[Prediction],
    gt_bboxes: list[WildtrackBBox],
    pred_to_gt_pid: list[int],
) -> dict:
    """
    Compute cross-camera identity metrics:
      IDP = ID Precision
      IDR = ID Recall
      IDF1 = harmonic mean
      Plus diagnostic counts.

    Algorithm:
      1. Build contingency matrix C[i, j] = number of predictions where
         predicted global ID = i AND matched GT person ID = j.
      2. Use Hungarian on -C to find optimal one-to-one mapping between
         predicted IDs and GT IDs that maximizes total true-positives.
      3. From the mapping:
         - True positives (TP) = sum of C[i, mapping(i)] over matched pairs
         - False positives (FP) = total predictions - TP
         - False negatives (FN) = total GT bbox appearances - TP
         - IDP = TP / (TP + FP)
         - IDR = TP / (TP + FN)
         - IDF1 = 2 * IDP * IDR / (IDP + IDR)
    """

    predicted_ids = sorted({p.global_id for p in predictions})
    gt_ids = sorted({g.person_id for g in gt_bboxes})

    pid_to_row = {gid: i for i, gid in enumerate(predicted_ids)}
    gt_to_col = {pid: j for j, pid in enumerate(gt_ids)}

    n_pred = len(predicted_ids)
    n_gt = len(gt_ids)

    contingency = np.zeros((n_pred, n_gt), dtype=np.int64)
    for pi, p in enumerate(predictions):
        gt_pid = pred_to_gt_pid[pi]
        if gt_pid < 0:
            continue
        if p.global_id not in pid_to_row or gt_pid not in gt_to_col:
            continue
        contingency[pid_to_row[p.global_id], gt_to_col[gt_pid]] += 1

    if n_pred == 0 or n_gt == 0:
        TP = 0
    else:
        size = max(n_pred, n_gt)
        cost = np.zeros((size, size), dtype=np.int64)
        cost[:n_pred, :n_gt] = -contingency
        row_idx, col_idx = linear_sum_assignment(cost)
        TP = 0
        for ri, ci in zip(row_idx, col_idx):
            if ri < n_pred and ci < n_gt:
                TP += int(contingency[ri, ci])

    total_predictions = len(predictions)
    total_gt = len(gt_bboxes)
    FP = total_predictions - TP
    FN = total_gt - TP

    IDP = TP / max(total_predictions, 1) if total_predictions > 0 else 0.0
    IDR = TP / max(total_gt, 1) if total_gt > 0 else 0.0
    if IDP + IDR > 0:
        IDF1 = 2 * IDP * IDR / (IDP + IDR)
    else:
        IDF1 = 0.0

    return {
        "IDF1": IDF1 * 100.0,
        "IDP": IDP * 100.0,
        "IDR": IDR * 100.0,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "n_predicted_ids": n_pred,
        "n_gt_ids": n_gt,
        "total_predictions": total_predictions,
        "total_gt": total_gt,
    }


def evaluate_wildtrack(
    per_camera_dir: str | Path,
    assignments: dict[tuple[int, int], int],
    annotations_root: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """
    End-to-end Wildtrack evaluation given:
      - directory with per-camera MOT files (C1.txt..C7.txt)
      - assignments dict from CrossCameraMatcher.match()
      - root of Wildtrack annotations_positions/

    Returns metrics dict.
    """

    predictions = load_predictions_from_mot_files(per_camera_dir, assignments)
    gt_bboxes = load_all_annotations(annotations_root)
    pred_to_gt_pid, pred_to_gt_iou = match_predictions_to_gt(predictions, gt_bboxes, iou_threshold)
    metrics = compute_identity_metrics(predictions, gt_bboxes, pred_to_gt_pid)
    metrics["iou_threshold"] = iou_threshold
    metrics["matched_predictions"] = sum(1 for p in pred_to_gt_pid if p >= 0)
    metrics["unmatched_predictions"] = sum(1 for p in pred_to_gt_pid if p < 0)
    metrics["mean_matched_iou"] = float(np.mean([iou for iou in pred_to_gt_iou if iou > 0.0])) if any(iou > 0.0 for iou in pred_to_gt_iou) else 0.0
    return metrics