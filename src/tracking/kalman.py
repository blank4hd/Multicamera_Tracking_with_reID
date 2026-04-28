"""Kalman filter tracker for SORT-style single-box tracking."""

from __future__ import annotations

import numpy as np
from scipy.linalg import inv

_EPS = 1e-6


def _xyxy_to_uvsr(bbox: np.ndarray) -> np.ndarray:
    """Convert xyxy box to [u, v, s, r] representation."""
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float).reshape(-1)[:4]
    w = max(x2 - x1, _EPS)
    h = max(y2 - y1, _EPS)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = max(w * h, _EPS)
    r = max(w / h, _EPS)
    return np.array([u, v, s, r], dtype=float)


def _uvsr_to_xyxy(state: np.ndarray) -> np.ndarray:
    """Convert [u, v, s, r] (or full state) to xyxy box."""
    vec = np.asarray(state, dtype=float).reshape(-1)
    u, v, s, r = vec[:4]
    s = max(float(s), _EPS)
    r = max(float(r), _EPS)
    w = np.sqrt(s * r)
    h = s / max(w, _EPS)
    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


class KalmanBoxTracker:
    """Track a single bounding box with a constant-velocity Kalman filter."""

    count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        p: np.ndarray | None = None,
    ) -> None:
        measurement = _xyxy_to_uvsr(np.asarray(bbox, dtype=float))

        self.x = np.zeros((7, 1), dtype=float)
        self.x[:4, 0] = measurement

        self.F = np.eye(7, dtype=float)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7), dtype=float)
        self.H[:4, :4] = np.eye(4, dtype=float)

        self.Q = (
            np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]).astype(float)
            if q is None
            else np.asarray(q, dtype=float)
        )
        self.R = (
            np.diag([1.0, 1.0, 10.0, 10.0]).astype(float)
            if r is None
            else np.asarray(r, dtype=float)
        )
        self.P = (
            np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0]).astype(float)
            if p is None
            else np.asarray(p, dtype=float)
        )

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history: list[np.ndarray] = []
        self.last_confidence = 1.0

    def predict(self) -> np.ndarray:
        """Advance one time-step and return predicted xyxy box."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if self.x[2, 0] <= _EPS:
            self.x[2, 0] = _EPS

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        predicted = _uvsr_to_xyxy(self.x[:4, 0])
        self.history.append(predicted)
        return predicted

    def update(self, bbox: np.ndarray, confidence: float = 1.0) -> None:
        """Correct state with a new xyxy observation."""
        z = _xyxy_to_uvsr(np.asarray(bbox, dtype=float)).reshape(4, 1)

        y = z - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ inv(s)

        self.x = self.x + (k @ y)
        identity = np.eye(7, dtype=float)
        self.P = (identity - (k @ self.H)) @ self.P

        if self.x[2, 0] <= _EPS:
            self.x[2, 0] = _EPS

        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.last_confidence = float(confidence)

    def get_state(self) -> np.ndarray:
        """Return current state as xyxy box."""
        return _uvsr_to_xyxy(self.x[:4, 0])
