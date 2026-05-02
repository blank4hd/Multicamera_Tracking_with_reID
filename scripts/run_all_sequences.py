"""Run tracking on MOT17 train sequences and save MOT-format predictions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from src.detection import PersonDetector
from src.evaluation import write_mot_results
from src.reid import ReIDFeatureExtractor
from src.tracking import DeepSORTTracker, SORTTracker
from src.utils.device import get_device

MOT17_TRAIN_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tracking on all MOT17 train sequences")
    parser.add_argument("--mot17-dir", type=Path, default=Path("data/MOT17/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions/SORT"))
    parser.add_argument("--tracker", choices=["sort", "deepsort"], default="sort")
    parser.add_argument("--reid-checkpoint", type=Path, default=Path("outputs/reid/best.pth"))
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.45,
                        help="YOLO NMS IoU threshold for detection (suppresses overlapping boxes).")
    parser.add_argument("--assoc-iou", type=float, default=0.30,
                        help="Tracker association IoU threshold (SORT gate / DeepSORT IoU fallback).")
    parser.add_argument("--iou-gate", type=float, default=0.0)
    parser.add_argument("--appearance-thresh", type=float, default=0.2)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--sequences", nargs="+", default=MOT17_TRAIN_SEQUENCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    detector = PersonDetector(model_name=args.model, device=device,
                              conf_threshold=args.conf, iou_threshold=args.nms_iou)

    feature_extractor = None
    if args.tracker == "deepsort":
        print(f"Loading Re-ID model from {args.reid_checkpoint}...")
        feature_extractor = ReIDFeatureExtractor(args.reid_checkpoint, embedding_dim=args.embedding_dim, device=device)

    total_frames = 0
    total_start = time.time()

    for seq_name in args.sequences:
        print(f"\n=== Running on {seq_name} ===")
        if args.tracker == "deepsort":
            tracker = DeepSORTTracker(
                feature_extractor=feature_extractor,
                max_age=args.max_age,
                n_init=args.min_hits,
                iou_gate_threshold=args.iou_gate,
                appearance_threshold=args.appearance_thresh,
                iou_threshold_fallback=args.assoc_iou,
            )
        else:
            tracker = SORTTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.assoc_iou)

        img_dir = args.mot17_dir / seq_name / "img1"
        frame_paths = sorted(img_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found for sequence '{seq_name}' in {img_dir}")

        tracks_per_frame: dict[int, list] = {}
        seen_track_ids: set[int] = set()

        seq_start = time.time()
        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=seq_name), start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")

            detections = detector.detect(frame)
            if args.tracker == "deepsort":
                tracks = tracker.update(frame, detections, frame_idx=frame_idx)
            else:
                tracks = tracker.update(detections, frame_idx=frame_idx)

            tracks_per_frame[frame_idx] = tracks
            seen_track_ids.update(track.track_id for track in tracks)

        seq_time = time.time() - seq_start
        num_frames = len(frame_paths)
        total_frames += num_frames
        fps = (num_frames / seq_time) if seq_time > 0 else 0.0

        output_file = args.output_dir / f"{seq_name}.txt"
        write_mot_results(output_file, tracks_per_frame)

        print(f"{seq_name}: frames={num_frames}, unique_ids={len(seen_track_ids)}, fps={fps:.2f}")

    total_time = time.time() - total_start
    overall_fps = (total_frames / total_time) if total_time > 0 else 0.0
    print(f"\nFinished all sequences: total_frames={total_frames}, total_time={total_time:.2f}s, overall_fps={overall_fps:.2f}")


if __name__ == "__main__":
    main()
