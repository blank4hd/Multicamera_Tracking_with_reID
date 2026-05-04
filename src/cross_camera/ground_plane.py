from dataclasses import dataclass
from pathlib import Path

from .calibration import CameraCalibration, load_all_calibrations


@dataclass
class ZoneBounds:
    """Rectangular ground-plane bounds (cm), inclusive."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, X: float, Y: float) -> bool:
        return (self.x_min <= X <= self.x_max and
                self.y_min <= Y <= self.y_max)


# Default Wildtrack zone bounds, derived empirically from GT projection across
# 20 sampled frames using 1st-99th percentile + ~50cm safety margin.
DEFAULT_WILDTRACK_ZONE = ZoneBounds(
    x_min=-280.0,
    x_max=940.0,
    y_min=-650.0,
    y_max=2380.0,
)


class GroundPlaneFilter:
    """
    Filters per-camera bbox detections by projecting their foot point onto the
    ground plane and checking against an annotated-zone rectangle.

    Detections whose foot projects outside the zone are dropped.

    Args:
        calibrations_root: path to Wildtrack calibrations folder.
        zone: ZoneBounds. Defaults to DEFAULT_WILDTRACK_ZONE.
        num_cameras: number of cameras to load. Defaults to 7.
    """

    def __init__(self,
                 calibrations_root: str | Path,
                 zone: ZoneBounds | None = None,
                 num_cameras: int = 7):
        self.calibrations = load_all_calibrations(calibrations_root, num_cameras)
        self.zone = zone if zone is not None else DEFAULT_WILDTRACK_ZONE
        self.num_cameras = num_cameras

    def is_in_zone(self, camera_id: int, bbox: tuple[float, float, float, float]) -> bool:
        """Check if a bbox's foot point projects to the ground-plane zone."""
        if camera_id not in self.calibrations:
            return True  # unknown camera; don't filter
        calib = self.calibrations[camera_id]
        X, Y = calib.bbox_foot_to_ground(bbox)
        return self.zone.contains(X, Y)

    def filter_detections(self, camera_id: int, detections: list) -> list:
        """
        Return a new list containing only detections whose foot is in the zone.

        detections: list of objects with a `.bbox` attribute (xyxy tuple).
                    Compatible with Detection from src.detection.
        """
        return [d for d in detections if self.is_in_zone(camera_id, d.bbox)]