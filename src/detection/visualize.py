"""Visualization helpers for person detection outputs."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw detections on a copy of the input frame and return the annotated copy."""
    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))

        cv2.rectangle(annotated, pt1, pt2, color, thickness)

        if show_conf:
            text = f"person {detection.confidence:.2f}"
            text_org = (pt1[0], max(0, pt1[1] - 10))
            cv2.putText(
                annotated,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

    return annotated
