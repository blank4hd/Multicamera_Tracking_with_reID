import numpy as np

from src.tracking.assignment import associate_detections_to_tracks, iou_batch


def test_iou_identical_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_batch(a, b)
    assert iou.shape == (1, 1)
    np.testing.assert_allclose(iou[0, 0], 1.0)


def test_iou_no_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    iou = iou_batch(a, b)
    assert iou[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    iou = iou_batch(a, b)
    # Intersection: 5*10=50, Union: 100+100-50=150, IoU = 50/150 = 1/3
    np.testing.assert_allclose(iou[0, 0], 1.0 / 3.0)


def test_iou_empty():
    a = np.zeros((0, 4))
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_batch(a, b)
    assert iou.shape == (0, 1)


def test_assignment_perfect_match():
    detections = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    tracks = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    matches, unmatched_d, unmatched_t = associate_detections_to_tracks(detections, tracks)
    assert len(matches) == 2
    assert len(unmatched_d) == 0
    assert len(unmatched_t) == 0


def test_assignment_no_overlap():
    detections = np.array([[0.0, 0.0, 10.0, 10.0]])
    tracks = np.array([[100.0, 100.0, 110.0, 110.0]])
    matches, unmatched_d, unmatched_t = associate_detections_to_tracks(detections, tracks)
    assert len(matches) == 0
    assert len(unmatched_d) == 1
    assert len(unmatched_t) == 1


def test_assignment_empty_tracks():
    detections = np.array([[0.0, 0.0, 10.0, 10.0]])
    tracks = np.zeros((0, 4))
    matches, unmatched_d, unmatched_t = associate_detections_to_tracks(detections, tracks)
    assert len(matches) == 0
    assert len(unmatched_d) == 1
    assert len(unmatched_t) == 0
