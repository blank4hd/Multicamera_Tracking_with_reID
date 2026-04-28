"""Person detection utilities built on top of Ultralytics YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from src.utils.device import get_device


@dataclass
class Detection:
    """Single detection result for one object in one frame."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    frame_idx: int | None = None

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Return box as (x_top_left, y_top_left, width, height)."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        """Return center point of the detection box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Return area of the detection box in pixels squared."""
        _, _, w, h = self.xywh
        return w * h


class PersonDetector:
    """Ultralytics YOLOv8-based detector for class-filtered person detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str | torch.device | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        person_class_id: int = 0,
        image_size: int = 640,
    ) -> None:
        self.model_name = model_name
        self.device = get_device() if device is None else device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.image_size = image_size

        self.model = YOLO(self.model_name)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame and return person detections."""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            classes=[self.person_class_id],
            verbose=False,
        )
        return self._results_to_detections(results[0])

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run batched detection on a list of BGR frames."""
        if not frames:
            return []

        results = self.model(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            classes=[self.person_class_id],
            verbose=False,
        )
        return [self._results_to_detections(result) for result in results]

    @staticmethod
    def _results_to_detections(result: object) -> list[Detection]:
        """Convert one Ultralytics result object into Detection instances."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for bbox, confidence, class_id in zip(xyxy, conf, cls):
            detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(confidence),
                    class_id=int(class_id),
                )
            )
        return detections

    def __repr__(self) -> str:
        return (
            "PersonDetector("
            f"model_name={self.model_name!r}, "
            f"device={self.device!r}, "
            f"conf_threshold={self.conf_threshold}, "
            f"iou_threshold={self.iou_threshold}, "
            f"person_class_id={self.person_class_id}, "
            f"image_size={self.image_size}"
            ")"
        )
