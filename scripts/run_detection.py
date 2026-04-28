"""CLI for running YOLOv8 person detection on images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from src.detection import PersonDetector, draw_detections


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the detection demo."""
    parser = argparse.ArgumentParser(description="Run YOLOv8 person detection on images.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/detection_demo",
        help="Directory where annotated frames are saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated frames; only print counts.",
    )
    return parser.parse_args()


def resolve_input_paths(input_path: Path) -> list[Path]:
    """Resolve a single image path or all image files in a directory."""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png"}
        image_paths = [
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in valid_exts
        ]
        return sorted(image_paths)

    return []


def main() -> None:
    """Run person detection and optionally save annotated outputs."""
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    frame_paths = resolve_input_paths(input_path)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    if not frame_paths:
        raise FileNotFoundError(f"No input images found at: {input_path}")

    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    detector = PersonDetector(model_name=args.model, conf_threshold=args.conf)

    total_detections = 0
    total_frames = 0
    start_time = time.perf_counter()

    for idx, frame_path in enumerate(tqdm(frame_paths, desc="Detecting", unit="frame"), start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        detections = detector.detect(frame)
        num_detections = len(detections)

        total_frames += 1
        total_detections += num_detections

        print(f"frame {idx}: {num_detections} detections")

        if not args.no_save:
            annotated = draw_detections(frame, detections)
            out_path = output_dir / frame_path.name
            cv2.imwrite(str(out_path), annotated)

    total_time = time.perf_counter() - start_time
    avg_per_frame = (total_detections / total_frames) if total_frames else 0.0
    fps = (total_frames / total_time) if total_time > 0 else 0.0

    print(f"total frames: {total_frames}")
    print(f"total detections: {total_detections}")
    print(f"average detections/frame: {avg_per_frame:.2f}")
    print(f"total time (s): {total_time:.3f}")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
