import numpy as np
import pytest

from src.cross_camera import Prediction
from src.cross_camera.evaluator import (
    _bbox_iou,
    match_predictions_to_gt,
    compute_identity_metrics,
)
from src.cross_camera.wildtrack_io import WildtrackBBox


def test_bbox_iou_identical():
    a = (10, 10, 50, 50)
    assert _bbox_iou(a, a) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert _bbox_iou(a, b) == 0.0


def test_bbox_iou_half_overlap():
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    assert _bbox_iou(a, b) == pytest.approx(50 / 150)


def make_pred(cam, frame, gid, x1, y1, x2, y2, local=0):
    return Prediction(camera_id=cam, frame_idx=frame, global_id=gid, x1=x1, y1=y1, x2=x2, y2=y2, local_track_id=local)


def make_gt(cam, frame, pid, x1, y1, x2, y2):
    return WildtrackBBox(camera_id=cam, person_id=pid, frame_idx=frame, x1=x1, y1=y1, x2=x2, y2=y2)


def test_perfect_single_camera_single_person():
    """1 prediction, 1 GT, perfect overlap, perfect IDF1."""
    preds = [make_pred(0, 1, 100, 0, 0, 10, 10)]
    gt = [make_gt(0, 1, 5, 0, 0, 10, 10)]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    assert pred_to_gt_pid == [5]
    metrics = compute_identity_metrics(preds, gt, pred_to_gt_pid)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["IDF1"] == pytest.approx(100.0)


def test_two_predictions_one_gt_per_cell():
    """Two predictions in same cell, one GT. Hungarian picks the better-IoU prediction."""
    preds = [
        make_pred(0, 1, 100, 0, 0, 10, 10),
        make_pred(0, 1, 200, 0, 0, 5, 5),
    ]
    gt = [make_gt(0, 1, 5, 0, 0, 10, 10)]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    assert pred_to_gt_pid[0] == 5
    assert pred_to_gt_pid[1] == -1


def test_below_iou_threshold():
    """A prediction below IoU threshold is unmatched."""
    preds = [make_pred(0, 1, 100, 0, 0, 10, 10)]
    gt = [make_gt(0, 1, 5, 100, 100, 110, 110)]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    assert pred_to_gt_pid == [-1]


def test_perfect_two_persons_two_cameras():
    """
    Person A in cam 0 and cam 1: 2 GT bboxes, 2 predictions, both labeled global_id=10.
    Person B in cam 0 and cam 1: 2 GT bboxes, 2 predictions, both labeled global_id=20.
    Perfect matching: IDF1 = 100%.
    """
    preds = [
        make_pred(0, 1, 10, 0, 0, 10, 10),
        make_pred(1, 1, 10, 0, 0, 10, 10),
        make_pred(0, 1, 20, 100, 100, 110, 110),
        make_pred(1, 1, 20, 100, 100, 110, 110),
    ]
    gt = [
        make_gt(0, 1, 100, 0, 0, 10, 10),
        make_gt(1, 1, 100, 0, 0, 10, 10),
        make_gt(0, 1, 200, 100, 100, 110, 110),
        make_gt(1, 1, 200, 100, 100, 110, 110),
    ]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    metrics = compute_identity_metrics(preds, gt, pred_to_gt_pid)
    assert metrics["TP"] == 4
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["IDF1"] == pytest.approx(100.0)


def test_id_swap_costs_recall():
    """
    Same setup as above BUT global IDs are swapped between cam 0 and cam 1.
    The Hungarian step will match one of {pred_id=10, pred_id=20} to person A,
    other to person B. So 2 of 4 predictions count as TP; the rest are FP/FN.
    """
    preds = [
        make_pred(0, 1, 10, 0, 0, 10, 10),
        make_pred(1, 1, 20, 0, 0, 10, 10),
        make_pred(0, 1, 20, 100, 100, 110, 110),
        make_pred(1, 1, 10, 100, 100, 110, 110),
    ]
    gt = [
        make_gt(0, 1, 100, 0, 0, 10, 10),
        make_gt(1, 1, 100, 0, 0, 10, 10),
        make_gt(0, 1, 200, 100, 100, 110, 110),
        make_gt(1, 1, 200, 100, 100, 110, 110),
    ]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    metrics = compute_identity_metrics(preds, gt, pred_to_gt_pid)
    assert metrics["TP"] == 2
    assert metrics["FP"] == 2
    assert metrics["FN"] == 2
    assert metrics["IDF1"] == pytest.approx(50.0)


def test_unmatched_prediction_counts_as_FP():
    """A prediction with no GT match should count as FP."""
    preds = [
        make_pred(0, 1, 10, 0, 0, 10, 10),
        make_pred(0, 1, 20, 500, 500, 510, 510),
    ]
    gt = [make_gt(0, 1, 100, 0, 0, 10, 10)]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    metrics = compute_identity_metrics(preds, gt, pred_to_gt_pid)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0


def test_missed_gt_counts_as_FN():
    """A GT with no prediction in that cell -> FN."""
    preds = [make_pred(0, 1, 10, 0, 0, 10, 10)]
    gt = [
        make_gt(0, 1, 100, 0, 0, 10, 10),
        make_gt(0, 1, 200, 100, 100, 110, 110),
    ]
    pred_to_gt_pid, _ = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
    metrics = compute_identity_metrics(preds, gt, pred_to_gt_pid)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0
    assert metrics["FN"] == 1