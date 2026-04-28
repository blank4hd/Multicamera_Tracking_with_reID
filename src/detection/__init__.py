"""Detection package exports."""

from .detector import Detection, PersonDetector
from .visualize import draw_detections

__all__ = ["PersonDetector", "Detection", "draw_detections"]

