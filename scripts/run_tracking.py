"""Run single-camera tracking on frame folders."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from src.detection import PersonDetector
from src.evaluation import write_mot_results
from src.reid import ReIDFeatureExtractor
from src.tracking import DeepSORTTracker, SORTTracker, draw_tracks
from src.utils.device import get_device


def _collect_frame_paths(input_dir: Path) -> list[Path]:
    valid_exts = {".jpg", ".jpeg", ".png"}
    frame_paths = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_exts]
    return sorted(frame_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tracking on a directory of image frames")
    parser.add_argument("--input", required=True, help="Path to input frame directory")
    parser.add_argument("--output-dir", default="outputs/tracking_demo", help="Directory where annotated frames are written")
    parser.add_argument("--output-mot", default="outputs/tracking_demo/results.txt", help="Path to output MOT-format .txt results file")
    parser.add_argument("--tracker", choices=["sort", "deepsort"], default="sort", help="Tracking mode to run")
    parser.add_argument("--reid-checkpoint", default="outputs/reid/best.pth", help="Re-ID checkpoint for DeepSORT")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Re-ID embedding dimension")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model path/name")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="SORT IoU association threshold / DeepSORT fallback IoU threshold")
    parser.add_argument("--iou-gate", type=float, default=0.0, help="DeepSORT cascade IoU gate threshold")
    parser.add_argument("--appearance-thresh", type=float, default=0.2, help="DeepSORT cosine distance threshold")
    parser.add_argument("--max-age", type=int, default=30, help="Max frames without update before deletion")
    parser.add_argument("--min-hits", type=int, default=3, help="Consecutive matches needed for confirmation")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no-save-frames", action="store_true", help="Skip saving annotated frames")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    frame_paths = _collect_frame_paths(input_dir)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {input_dir}")

    output_dir = Path(args.output_dir)
    if not args.no_save_frames:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    detector = PersonDetector(model_name=args.model, conf_threshold=args.conf)

    if args.tracker == "deepsort":
        checkpoint = Path(args.reid_checkpoint)
        print(f"Loading Re-ID model from {checkpoint}...")
        feature_extractor = ReIDFeatureExtractor(checkpoint, embedding_dim=args.embedding_dim, device=device)
        tracker = DeepSORTTracker(
            feature_extractor=feature_extractor,
            max_age=args.max_age,
            n_init=args.min_hits,
            iou_gate_threshold=args.iou_gate,
            appearance_threshold=args.appearance_thresh,
            iou_threshold_fallback=args.iou,
        )
    else:
        tracker = SORTTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou)

    tracks_per_frame: dict[int, list] = {}
    unique_track_ids: set[int] = set()
    total_emitted_tracks = 0

    start = time.perf_counter()
    total_frames = 0

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Tracking"), start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        detections = detector.detect(frame)
        if args.tracker == "deepsort":
            tracks = tracker.update(frame, detections, frame_idx=frame_idx)
        else:
            tracks = tracker.update(detections, frame_idx=frame_idx)

        tracks_per_frame[frame_idx] = tracks
        total_emitted_tracks += len(tracks)
        unique_track_ids.update(track.track_id for track in tracks)
        total_frames += 1

        if not args.no_save_frames:
            annotated = draw_tracks(frame, tracks)
            cv2.imwrite(str(output_dir / frame_path.name), annotated)

    write_mot_results(args.output_mot, tracks_per_frame)

    elapsed = time.perf_counter() - start
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    print(f"Total frames processed: {total_frames}")
    print(f"Total unique track IDs: {len(unique_track_ids)}")
    print(f"Total emitted tracks: {total_emitted_tracks}")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
