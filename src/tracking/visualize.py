"""Visualization helpers for tracked boxes."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .sort import TrackedBox


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministically map a track ID to a vivid BGR color."""
    hue = (float(track_id) * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackedBox],
    thickness: int = 2,
) -> np.ndarray:
    """Draw tracked bounding boxes and ID labels on a frame copy."""
    annotated = frame.copy()

    for track in tracks:
        color = _color_for_id(track.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {track.track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        bg_x1 = x1
        bg_y2 = y1
        bg_y1 = max(0, bg_y2 - text_h - baseline - 4)
        bg_x2 = bg_x1 + text_w + 6

        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        text_x = bg_x1 + 3
        text_y = bg_y2 - baseline - 2
        if text_y <= bg_y1:
            text_y = bg_y1 + text_h
        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated
