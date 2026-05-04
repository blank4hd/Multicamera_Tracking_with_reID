import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.cross_camera.visualize import color_for_global_id


def parse_seqinfo(seqinfo_path: Path) -> tuple[int, int, int]:
    """Parse frameRate, imWidth, and imHeight from a MOT seqinfo.ini file."""
    values: dict[str, str] = {}
    in_sequence_section = False
    with open(seqinfo_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                in_sequence_section = line == "[Sequence]"
                continue
            if not in_sequence_section or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    try:
        frame_rate = int(values["frameRate"])
        im_width = int(values["imWidth"])
        im_height = int(values["imHeight"])
    except KeyError as exc:
        raise KeyError(f"Missing required key in {seqinfo_path}: {exc.args[0]}") from exc
    return frame_rate, im_width, im_height


def draw_tracks_on_frame(frame, boxes, thickness=3, font_scale=0.7):
    """Draw track bboxes with per-ID coloring."""
    out = frame.copy()
    for b in boxes:
        tid = int(b["track_id"])
        color = color_for_global_id(tid)
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_overlay(frame, text, font_scale):
    """Draw a semi-opaque overlay banner in the top-left."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(frame, (8, 8), (8 + tw + 12, 8 + th + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, 8 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a MOT17 single-camera tracking demo video")
    parser.add_argument(
        "--sequence",
        type=str,
        default="MOT17-04-FRCNN",
        help="MOT17 sequence name (e.g., MOT17-04-FRCNN)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/predictions/DeepSORT_v3",
        help="Directory with MOT-format prediction files",
    )
    parser.add_argument(
        "--mot-root",
        type=str,
        default="data/MOT17/train",
        help="Root of MOT17 train sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output mp4 path. Defaults to outputs/mot17_demo_<sequence>.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate. Defaults to sequence's native fps from seqinfo.ini",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap output to this many frames (for testing)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Bbox border thickness",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.7,
        help="Label font scale",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seq_dir = Path(args.mot_root) / args.sequence
    img_dir = seq_dir / "img1"
    seqinfo_path = seq_dir / "seqinfo.ini"
    pred_path = Path(args.predictions) / f"{args.sequence}.txt"
    output_path = Path(args.output) if args.output else Path("outputs") / f"mot17_demo_{args.sequence}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    native_fps, im_width, im_height = parse_seqinfo(seqinfo_path)
    fps = float(args.fps) if args.fps is not None else float(native_fps)

    boxes_by_frame = defaultdict(list)
    unique_track_ids: set[int] = set()
    with open(pred_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            boxes_by_frame[frame].append({
                "track_id": track_id,
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
            })
            unique_track_ids.add(track_id)

    image_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    width = im_width
    height = im_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {output_path}. Try installing ffmpeg or use a different codec."
        )

    n_emitted = 0
    total_frames = len(image_files)
    for fidx in tqdm(range(1, total_frames + 1), desc="Frames"):
        if args.max_frames and n_emitted >= args.max_frames:
            break
        img_path = image_files[fidx - 1]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: failed to read {img_path}, skipping")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        boxes = boxes_by_frame.get(fidx, [])
        annotated = draw_tracks_on_frame(
            frame,
            boxes,
            thickness=args.thickness,
            font_scale=args.font_scale,
        )
        overlay = f"Frame {fidx}/{total_frames} | Active tracks: {len(boxes)}"
        draw_overlay(annotated, overlay, args.font_scale)
        writer.write(annotated)
        n_emitted += 1

    writer.release()

    try:
        size_mb = output_path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = -1.0

    print(f"Output: {output_path}")
    print(f"Frames written: {n_emitted}")
    if size_mb >= 0:
        print(f"File size: {size_mb:.2f} MB")
    else:
        print("File size: unavailable")
    print(f"Unique track IDs: {len(unique_track_ids)}")


if __name__ == "__main__":
    main()