"""Run per-camera detection and DeepSORT tracking on Wildtrack."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.cross_camera.wildtrack_io import WILDTRACK_NUM_CAMERAS, list_camera_frames
from src.detection import PersonDetector
from src.evaluation import write_mot_results
from src.reid import ReIDFeatureExtractor
from src.tracking import DeepSORTTracker
from src.utils.device import get_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run per-camera Wildtrack tracking with embeddings.")
    parser.add_argument("--wildtrack-dir", default="data/Wildtrack")
    parser.add_argument("--reid-checkpoint", default="outputs/reid/best.pth")
    parser.add_argument("--output-dir", default="outputs/wildtrack/per_camera")
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        help=(
            "YOLO model name. Default yolov8m.pt; yolo26l.pt is also supported but "
            "requires a lower conf threshold (~0.2) to match recall."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--iou-gate", type=float, default=0.0)
    parser.add_argument("--appearance-thresh", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--min-track-length", type=int, default=10)
    parser.add_argument(
        "--ground-plane-filter",
        action="store_true",
        help="Filter detections to those whose foot projects onto the Wildtrack annotated zone.",
    )
    parser.add_argument(
        "--calibrations-dir",
        default="data/Wildtrack/calibrations",
        help="Path to Wildtrack calibrations root.",
    )
    parser.add_argument(
        "--zone-x-min",
        type=float,
        default=None,
        help="Override zone X minimum (cm). Defaults to DEFAULT_WILDTRACK_ZONE.",
    )
    parser.add_argument(
        "--zone-x-max",
        type=float,
        default=None,
        help="Override zone X maximum (cm). Defaults to DEFAULT_WILDTRACK_ZONE.",
    )
    parser.add_argument(
        "--zone-y-min",
        type=float,
        default=None,
        help="Override zone Y minimum (cm). Defaults to DEFAULT_WILDTRACK_ZONE.",
    )
    parser.add_argument(
        "--zone-y-max",
        type=float,
        default=None,
        help="Override zone Y maximum (cm). Defaults to DEFAULT_WILDTRACK_ZONE.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        type=int,
        default=list(range(WILDTRACK_NUM_CAMERAS)),
        help="Subset of camera IDs to run (0-based).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    wildtrack_dir = Path(args.wildtrack_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    detector = PersonDetector(model_name=args.model, device=device, conf_threshold=args.conf, iou_threshold=args.iou)
    feature_extractor = ReIDFeatureExtractor(
        checkpoint_path=args.reid_checkpoint,
        embedding_dim=args.embedding_dim,
        device=device,
    )

    gp_filter = None
    if args.ground_plane_filter:
        from src.cross_camera import GroundPlaneFilter, ZoneBounds, DEFAULT_WILDTRACK_ZONE

        zone = ZoneBounds(
            x_min=args.zone_x_min if args.zone_x_min is not None else DEFAULT_WILDTRACK_ZONE.x_min,
            x_max=args.zone_x_max if args.zone_x_max is not None else DEFAULT_WILDTRACK_ZONE.x_max,
            y_min=args.zone_y_min if args.zone_y_min is not None else DEFAULT_WILDTRACK_ZONE.y_min,
            y_max=args.zone_y_max if args.zone_y_max is not None else DEFAULT_WILDTRACK_ZONE.y_max,
        )
        gp_filter = GroundPlaneFilter(args.calibrations_dir, zone=zone)
        print(
            f"Ground-plane filtering enabled. Zone: "
            f"X=[{zone.x_min:.0f}, {zone.x_max:.0f}], "
            f"Y=[{zone.y_min:.0f}, {zone.y_max:.0f}] cm"
        )

    total_tracks = 0
    image_subsets_root = wildtrack_dir / "Image_subsets"

    for camera_id in args.cameras:
        print(f"=== Camera C{camera_id + 1} ===")
        tracker = DeepSORTTracker(
            feature_extractor=feature_extractor,
            max_age=args.max_age,
            n_init=args.min_hits,
            iou_gate_threshold=args.iou_gate,
            appearance_threshold=args.appearance_thresh,
            iou_threshold_fallback=args.iou,
            gallery_size=30,
        )

        frames = list_camera_frames(image_subsets_root, camera_id)
        tracks_per_frame: dict[int, list] = {}

        for fidx, frame_path in enumerate(tqdm(frames, desc=f"C{camera_id + 1}"), start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Failed to read frame: {frame_path}")
            detections = detector.detect(frame)
            if gp_filter is not None:
                n_before = len(detections)
                detections = gp_filter.filter_detections(camera_id, detections)
                # if fidx <= 3:
                #     print(f"  cam {camera_id} frame {fidx}: {n_before} -> {len(detections)} detections after gp filter")
            tracks = tracker.update(frame, detections, frame_idx=fidx)
            if tracks:
                tracks_per_frame[fidx] = tracks

        all_track_data = tracker.get_all_track_data()
        filtered_track_ids = {
            tid for tid, data in all_track_data.items() if data["n_observations"] >= args.min_track_length
        }

        mot_path = output_dir / f"C{camera_id + 1}.txt"
        filtered_tracks_per_frame = {
            fidx: [track for track in track_list if track.track_id in filtered_track_ids]
            for fidx, track_list in tracks_per_frame.items()
        }
        filtered_tracks_per_frame = {frame_idx: track_list for frame_idx, track_list in filtered_tracks_per_frame.items() if track_list}
        write_mot_results(mot_path, filtered_tracks_per_frame)

        ids_arr = np.array(sorted(filtered_track_ids), dtype=np.int64)
        gallery_list = [all_track_data[int(track_id)]["gallery"] for track_id in ids_arr]
        first_frames = np.array([all_track_data[int(track_id)]["first_frame"] for track_id in ids_arr], dtype=np.int64)
        last_frames = np.array([all_track_data[int(track_id)]["last_frame"] for track_id in ids_arr], dtype=np.int64)
        n_obs = np.array([all_track_data[int(track_id)]["n_observations"] for track_id in ids_arr], dtype=np.int64)

        gallery_means = np.zeros((len(ids_arr), args.embedding_dim), dtype=np.float32)
        for idx, gallery in enumerate(gallery_list):
            mean = gallery.mean(axis=0)
            mean = mean / max(np.linalg.norm(mean), 1e-12)
            gallery_means[idx] = mean.astype(np.float32)

        npz_path = output_dir / f"C{camera_id + 1}.npz"
        np.savez(
            npz_path,
            track_ids=ids_arr,
            gallery_means=gallery_means,
            first_frames=first_frames,
            last_frames=last_frames,
            n_observations=n_obs,
            galleries=np.array(gallery_list, dtype=object),
        )

        total_tracks += len(ids_arr)
        suffix = " (gp-filtered)" if gp_filter is not None else ""
        print(f"C{camera_id + 1}: saved {len(ids_arr)} tracks to {mot_path.name} and {npz_path.name}{suffix}")

    elapsed = time.time() - start_time
    print(f"Finished. Saved {total_tracks} tracks across {len(args.cameras)} cameras in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
