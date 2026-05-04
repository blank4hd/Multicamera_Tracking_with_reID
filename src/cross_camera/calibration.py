import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Camera index (viewNum 0-6) to filename stem in calibration files
CAMERA_NAME_BY_ID = {
    0: 'CVLab1', 1: 'CVLab2', 2: 'CVLab3', 3: 'CVLab4',
    4: 'IDIAP1', 5: 'IDIAP2', 6: 'IDIAP3',
}


@dataclass
class CameraCalibration:
    """Camera intrinsics + extrinsics for one Wildtrack camera."""

    K: np.ndarray   # 3x3 intrinsic
    R: np.ndarray   # 3x3 rotation (world -> camera)
    t: np.ndarray   # 3x1 translation (world -> camera)

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """
        Project a pixel (u, v) onto the ground plane (z=0).
        Returns (X, Y) in centimeters in the calibration's world coordinate frame.
        """
        # Ray in camera frame
        pt = np.linalg.inv(self.K) @ np.array([float(u), float(v), 1.0])
        # Camera position in world coords: -R^T @ t
        cam_pos_world = -self.R.T @ self.t
        # Ray direction in world coords
        ray_world = self.R.T @ pt
        # Intersect with z=0 plane: cam_pos[2] + s * ray[2] = 0
        rz = float(ray_world[2])
        if abs(rz) < 1e-9:
            # Ray is parallel to ground plane (camera looking straight ahead).
            # Return a sentinel far outside any zone.
            return (1e9, 1e9)
        s = -float(cam_pos_world[2, 0]) / rz
        X = float(cam_pos_world[0, 0]) + s * float(ray_world[0])
        Y = float(cam_pos_world[1, 0]) + s * float(ray_world[1])
        return X, Y

    def bbox_foot_to_ground(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """
        Project the foot point (bottom-center of bbox in xyxy format) to the ground plane.
        bbox: (x1, y1, x2, y2)
        Returns (X, Y) in cm.
        """
        x1, _, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = y2
        return self.pixel_to_ground(u, v)


def _parse_xml_matrix(path: str | Path, tag: str) -> np.ndarray:
    """
    Parse an OpenCV-style XML matrix or vector. Handles both formats:
      - Structured matrix with <rows>, <cols>, <data> children (e.g. camera_matrix)
      - Vector with text content directly in the tag (e.g. rvec, tvec)
    """
    root = ET.parse(path).getroot()
    elem = root.find(tag)
    if elem is None:
        raise ValueError(f"Tag '{tag}' not found in {path}")
    rows_elem = elem.find('rows')
    if rows_elem is not None:
        rows = int(rows_elem.text)
        cols = int(elem.find('cols').text)
        data_text = elem.find('data').text.strip().split()
        return np.array([float(x) for x in data_text]).reshape(rows, cols)
    # Direct text content (rvec, tvec)
    return np.array([float(x) for x in elem.text.strip().split()])


def load_camera_calibration(
    calibrations_root: str | Path,
    camera_id: int,
) -> CameraCalibration:
    """
    Load one camera's calibration from Wildtrack's standard layout:
      calibrations_root/intrinsic_zero/intr_<name>.xml
      calibrations_root/extrinsic/extr_<name>.xml
    where <name> is determined by CAMERA_NAME_BY_ID[camera_id].
    """
    if camera_id not in CAMERA_NAME_BY_ID:
        raise ValueError(f"Unknown camera_id {camera_id}; expected 0-6")
    name = CAMERA_NAME_BY_ID[camera_id]
    root = Path(calibrations_root)
    intr_path = root / 'intrinsic_zero' / f'intr_{name}.xml'
    extr_path = root / 'extrinsic' / f'extr_{name}.xml'
    K = _parse_xml_matrix(intr_path, 'camera_matrix')
    rvec = _parse_xml_matrix(extr_path, 'rvec').reshape(3, 1)
    tvec = _parse_xml_matrix(extr_path, 'tvec').reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return CameraCalibration(K=K.astype(np.float64), R=R.astype(np.float64), t=tvec.astype(np.float64))


def load_all_calibrations(
    calibrations_root: str | Path,
    num_cameras: int = 7,
) -> dict[int, CameraCalibration]:
    """Load calibrations for all cameras 0..num_cameras-1."""
    return {c: load_camera_calibration(calibrations_root, c) for c in range(num_cameras)}