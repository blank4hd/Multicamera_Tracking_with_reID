"""Render a multi-camera grid video showing cross-camera global IDs for Wildtrack.

Usage: python scripts/make_wildtrack_video.py [options]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from src.cross_camera import (
    CrossCameraMatcher, load_camera_tracks,
    draw_global_id_boxes, make_grid_2x4,
    draw_global_id_boxes_with_palette, HIGHLIGHT_PALETTE_BGR,
    list_camera_frames,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render Wildtrack cross-camera grid video")
    p.add_argument("--wildtrack-dir", default="data/Wildtrack")
    p.add_argument("--per-camera-dir", default="outputs/wildtrack/per_camera")
    p.add_argument("--output", default="outputs/wildtrack/cross_camera_demo.mp4")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--tile-width", type=int, default=640)
    p.add_argument("--tile-height", type=int, default=360)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--num-cameras", type=int, default=7)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument(
        "--cross-camera-only",
        action="store_true",
        default=False,
        help="Only draw bboxes for global IDs that appear in 2+ cameras",
    )
    p.add_argument(
        "--highlight-top-n",
        type=int,
        default=0,
        help="If > 0, draw only the top N global IDs by cross-camera presence, using a curated color palette. Overrides --cross-camera-only.",
    )
    p.add_argument(
        "--highlight-gids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of global IDs to highlight. Overrides --highlight-top-n.",
    )
    return p


def _load_all_tracks(per_camera_dir: Path, num_cameras: int) -> list:
    all_tracks = []
    for cam_idx in range(num_cameras):
        npz_path = per_camera_dir / f"C{cam_idx + 1}.npz"
        if not npz_path.exists():
            continue
        all_tracks.extend(load_camera_tracks(npz_path, camera_id=cam_idx))
    return all_tracks


def _load_prediction_boxes(per_camera_dir: Path, assignments: dict[tuple[int, int], int], num_cameras: int):
    """Return dict[cam][frame] -> list of boxes dicts with x1,y1,x2,y2,global_id"""
    per_camera_dir = Path(per_camera_dir)
    boxes_by_cam_frame: dict[int, dict[int, list]] = {c: {} for c in range(num_cameras)}
    for cam_idx in range(num_cameras):
        path = per_camera_dir / f"C{cam_idx + 1}.txt"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                frame_idx = int(parts[0])
                mot_id = int(parts[1])
                local_id = mot_id - 1
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                key = (cam_idx, local_id)
                if key not in assignments:
                    continue
                gid = int(assignments[key])
                box = {"x1": x, "y1": y, "x2": x + w, "y2": y + h, "global_id": gid}
                boxes_by_cam_frame.setdefault(cam_idx, {}).setdefault(frame_idx, []).append(box)
    return boxes_by_cam_frame


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    per_camera_dir = Path(args.per_camera_dir)
    wildtrack_dir = Path(args.wildtrack_dir)
    output_path = Path(args.output)

    all_tracks = _load_all_tracks(per_camera_dir, args.num_cameras)
    matcher = CrossCameraMatcher(threshold=args.threshold)
    assignments = matcher.match(all_tracks)
    gid_to_cameras = defaultdict(set)
    for (cam_id, local_id), gid in assignments.items():
        gid_to_cameras[gid].add(cam_id)
    cross_camera_gids = {gid for gid, cams in gid_to_cameras.items() if len(cams) >= 2}
    print(f"Cross-camera GIDs: {len(cross_camera_gids)} (out of {len(gid_to_cameras)} total)")
    highlight_gid_to_color: dict[int, tuple[int, int, int]] = {}
    if args.highlight_gids is not None and len(args.highlight_gids) > 0:
        # Explicit list of GIDs to highlight (overrides auto-ranking)
        for i, gid in enumerate(args.highlight_gids):
            color = HIGHLIGHT_PALETTE_BGR[i % len(HIGHLIGHT_PALETTE_BGR)]
            highlight_gid_to_color[gid] = color
        print(f"Highlight reel: drawing {len(highlight_gid_to_color)} "
              f"explicit GIDs:")
        for gid in highlight_gid_to_color:
            cams = sorted(gid_to_cameras.get(gid, []))
            print(f"  GID {gid}: cameras {cams}")
    elif args.highlight_top_n > 0:
        # Score each GID by: number of distinct cameras * log(1 + total observations)
        import math
        gid_observation_count: dict[int, int] = {}
        gid_observation_count_default = 0
        # Count observations per GID by reading per-camera MOT files.
        # This requires the MOT files to be loaded; we'll do a quick pass here.
        # boxes_by_cam_frame may already be populated below; if so, count from there.
        # For clarity, recount using the assignments and per-camera prediction files:
        for cam_idx in range(args.num_cameras):
            mot_path = Path(args.per_camera_dir) / f"C{cam_idx + 1}.txt"
            if not mot_path.exists():
                continue
            with open(mot_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if not line.strip():
                        continue
                    mot_id = int(parts[1])
                    local_id = mot_id - 1
                    key = (cam_idx, local_id)
                    if key not in assignments:
                        continue
                    gid = assignments[key]
                    gid_observation_count[gid] = gid_observation_count.get(gid, 0) + 1

        scores: list[tuple[int, float]] = []
        for gid, cams in gid_to_cameras.items():
            n_cams = len(cams)
            n_obs = gid_observation_count.get(gid, 0)
            if n_cams < 2:
                continue  # only cross-camera GIDs are eligible
            score = n_cams * math.log(1 + n_obs)
            scores.append((gid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:args.highlight_top_n]
        for i, (gid, _score) in enumerate(top):
            color = HIGHLIGHT_PALETTE_BGR[i % len(HIGHLIGHT_PALETTE_BGR)]
            highlight_gid_to_color[gid] = color
        print(f"Highlight reel: drawing top {len(highlight_gid_to_color)} GIDs:")
        for gid in highlight_gid_to_color:
            cams = sorted(gid_to_cameras[gid])
            print(f"  GID {gid}: cameras {cams}, "
                  f"observations={gid_observation_count.get(gid, 0)}")
    summary = matcher.cluster_summary(all_tracks, assignments)
    print(f"Total tracks: {summary['n_tracks']}, global IDs: {summary['n_clusters']}, cross-camera clusters: {summary['cross_camera_clusters']}")

    boxes_by_cam_frame = _load_prediction_boxes(per_camera_dir, assignments, args.num_cameras)

    # List frames per camera (Image_subsets/C1..C7)
    frames_per_cam = []
    for cam_idx in range(args.num_cameras):
        frames = list_camera_frames(wildtrack_dir / "Image_subsets", cam_idx)
        frames_per_cam.append(frames)
    total_frames = min(len(frames) for frames in frames_per_cam)

    rows = 2
    cols = 4
    pad = 4
    tile_w = args.tile_width
    tile_h = args.tile_height
    grid_h = rows * tile_h + (rows - 1) * pad
    grid_w = cols * tile_w + (cols - 1) * pad

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(args.fps), (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}. Check codecs and output path.")

    labels = [f"C{i+1}" for i in range(args.num_cameras)] + [""]

    emitted = 0
    start = args.start_frame
    end = args.start_frame + total_frames
    for fidx in tqdm(range(start, end), desc="Frames"):
        if args.max_frames is not None and emitted >= args.max_frames:
            break
        camera_tiles: list[np.ndarray | None] = []
        for cam_idx in range(args.num_cameras):
            cam_frames = frames_per_cam[cam_idx]
            # MOT frames are 1-indexed; list_camera_frames returns 0-indexed list
            try:
                frame_path = cam_frames[fidx - 1]
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    # corrupt/missing -> black placeholder in expected resolution
                    frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            except Exception:
                frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

            bboxes = boxes_by_cam_frame.get(cam_idx, {}).get(fidx, [])
            if len(highlight_gid_to_color) > 0:
                annotated = draw_global_id_boxes_with_palette(
                    frame, bboxes, highlight_gid_to_color
                )
            elif args.cross_camera_only:
                bboxes = [b for b in bboxes if b["global_id"] in cross_camera_gids]
                annotated = draw_global_id_boxes(frame, bboxes)
            else:
                annotated = draw_global_id_boxes(frame, bboxes)
            camera_tiles.append(annotated)

        grid = make_grid_2x4(camera_tiles, tile_w, tile_h, pad=pad, labels=labels)
        writer.write(grid)
        emitted += 1

    writer.release()
    written = emitted
    try:
        size = output_path.stat().st_size
    except Exception:
        size = -1
    print(f"Wrote {written} frames to {output_path} ({size} bytes)")


if __name__ == "__main__":
    main()
