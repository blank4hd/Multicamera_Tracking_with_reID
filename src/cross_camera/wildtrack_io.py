"""Wildtrack data loading and annotation parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WILDTRACK_NUM_CAMERAS = 7
# Frame alignment relies on sorted PNG filenames matching sorted annotation JSON files.
# Wildtrack stores pre-subsampled images (every 5th frame of the original 60 fps stream)
# in Image_subsets/; no additional step is applied when iterating frames here.


@dataclass
class WildtrackBBox:
    camera_id: int
    person_id: int
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def list_camera_frames(image_subsets_root: str | Path, camera_id: int) -> list[Path]:
    """
    Return sorted list of frame paths for a given camera (0-indexed).
    image_subsets_root: e.g. data/Wildtrack/Image_subsets
    camera_id: 0-6 -> directory C1-C7
    """
    cam_dir = Path(image_subsets_root) / f"C{camera_id + 1}"
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
    frames = sorted(cam_dir.glob("*.png"))
    return frames


def list_annotation_files(annotations_root: str | Path) -> list[Path]:
    """Sorted list of annotation JSON paths."""
    return sorted(Path(annotations_root).glob("*.json"))


def parse_annotation_file(path: str | Path, frame_idx: int) -> list[WildtrackBBox]:
    """
    Parse one Wildtrack annotation JSON file.
    Returns list of WildtrackBBox for all (person, camera) pairs where the person
    is visible (i.e., bbox not -1,-1,-1,-1).

    frame_idx: the 1-indexed sequential frame number to attach to each bbox.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    bboxes: list[WildtrackBBox] = []
    for entry in data:
        pid = int(entry["personID"])
        for view in entry["views"]:
            cam_id = int(view["viewNum"])
            xmin, ymin, xmax, ymax = view["xmin"], view["ymin"], view["xmax"], view["ymax"]
            if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                continue
            bboxes.append(
                WildtrackBBox(
                    camera_id=cam_id,
                    person_id=pid,
                    frame_idx=frame_idx,
                    x1=float(xmin),
                    y1=float(ymin),
                    x2=float(xmax),
                    y2=float(ymax),
                )
            )
    return bboxes


def load_all_annotations(annotations_root: str | Path) -> list[WildtrackBBox]:
    """
    Load all Wildtrack annotations as a flat list of WildtrackBBox.
    Frames are numbered 1-indexed in order of sorted JSON files.
    """
    files = list_annotation_files(annotations_root)
    all_bboxes: list[WildtrackBBox] = []
    for i, path in enumerate(files, start=1):
        all_bboxes.extend(parse_annotation_file(path, frame_idx=i))
    return all_bboxes


def annotations_to_per_camera_per_frame(
    bboxes: list[WildtrackBBox],
    num_cameras: int = WILDTRACK_NUM_CAMERAS,
) -> dict[int, dict[int, list[WildtrackBBox]]]:
    """
    Reorganize a flat bbox list into:
      camera_id -> frame_idx -> list of WildtrackBBox
    """
    result: dict[int, dict[int, list[WildtrackBBox]]] = {c: {} for c in range(num_cameras)}
    for bb in bboxes:
        result[bb.camera_id].setdefault(bb.frame_idx, []).append(bb)
    return result
