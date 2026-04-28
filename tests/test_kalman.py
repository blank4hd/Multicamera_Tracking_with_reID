import numpy as np
import pytest

from src.tracking.kalman import KalmanBoxTracker


@pytest.fixture(autouse=True)
def reset_kbt_count():
    KalmanBoxTracker.count = 0
    yield


def test_initialization():
    bbox = np.array([100.0, 100.0, 200.0, 300.0])  # 100x200 box
    kbt = KalmanBoxTracker(bbox)
    assert kbt.id == 0
    assert kbt.hits == 0
    assert kbt.time_since_update == 0
    assert kbt.age == 0


def test_unique_ids():
    kbt1 = KalmanBoxTracker(np.array([0.0, 0.0, 10.0, 10.0]))
    kbt2 = KalmanBoxTracker(np.array([0.0, 0.0, 10.0, 10.0]))
    assert kbt1.id == 0
    assert kbt2.id == 1


def test_predict_returns_bbox_close_to_init():
    bbox = np.array([100.0, 100.0, 200.0, 300.0])
    kbt = KalmanBoxTracker(bbox)
    pred = kbt.predict()
    # No velocity, prediction should match init within tight tolerance
    np.testing.assert_allclose(pred, bbox, atol=1e-3)


def test_update_after_constant_motion():
    # Simulate a box moving 10 pixels right per frame
    kbt = KalmanBoxTracker(np.array([100.0, 100.0, 200.0, 200.0]))
    for i in range(1, 6):
        kbt.predict()
        new_bbox = np.array([100.0 + i * 10, 100.0, 200.0 + i * 10, 200.0])
        kbt.update(new_bbox)
    # Predict next: should extrapolate ~10 px right
    pred = kbt.predict()
    # x1 should be roughly 100 + 6*10 = 160; allow tolerance because Kalman smooths
    assert 150 < pred[0] < 170
    assert 250 < pred[2] < 270


def test_predict_increments_age():
    kbt = KalmanBoxTracker(np.array([0.0, 0.0, 10.0, 10.0]))
    for _ in range(5):
        kbt.predict()
    assert kbt.age == 5
