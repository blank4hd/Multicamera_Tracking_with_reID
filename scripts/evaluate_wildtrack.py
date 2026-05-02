"""CLI for offline cross-camera evaluation on Wildtrack."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cross_camera import CrossCameraMatcher, evaluate_wildtrack, load_camera_tracks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Wildtrack cross-camera identity matching.")
    parser.add_argument("--per-camera-dir", default="outputs/wildtrack/per_camera")
    parser.add_argument("--annotations", default="data/Wildtrack/annotations_positions")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-cameras", type=int, default=7)
    parser.add_argument("--output", default="outputs/wildtrack/eval_results.txt")
    parser.add_argument("--sweep", nargs="+", type=float, default=None)
    return parser


def _load_all_tracks(per_camera_dir: Path, num_cameras: int) -> list:
    all_tracks = []
    for cam_idx in range(num_cameras):
        npz_path = per_camera_dir / f"C{cam_idx + 1}.npz"
        if not npz_path.exists():
            continue
        all_tracks.extend(load_camera_tracks(npz_path, camera_id=cam_idx))
    return all_tracks


def _write_text_output(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    per_camera_dir = Path(args.per_camera_dir)
    annotations_root = Path(args.annotations)
    output_path = Path(args.output)

    all_tracks = _load_all_tracks(per_camera_dir, args.num_cameras)
    thresholds = args.sweep if args.sweep is not None else [args.threshold]

    if args.sweep is not None:
        lines = ["threshold | IDF1 | IDP | IDR | TP | FP | FN | n_pred_ids | n_gt_ids"]
        for threshold in thresholds:
            matcher = CrossCameraMatcher(threshold=threshold)
            assignments = matcher.match(all_tracks)
            metrics = evaluate_wildtrack(
                per_camera_dir=per_camera_dir,
                assignments=assignments,
                annotations_root=annotations_root,
                iou_threshold=args.iou_threshold,
            )
            line = (
                f"{threshold:.4f} | {metrics['IDF1']:.2f} | {metrics['IDP']:.2f} | {metrics['IDR']:.2f} | "
                f"{metrics['TP']} | {metrics['FP']} | {metrics['FN']} | {metrics['n_predicted_ids']} | {metrics['n_gt_ids']}"
            )
            print(line)
            lines.append(line)
        _write_text_output(output_path, lines)
        return

    matcher = CrossCameraMatcher(threshold=args.threshold)
    assignments = matcher.match(all_tracks)
    metrics = evaluate_wildtrack(
        per_camera_dir=per_camera_dir,
        assignments=assignments,
        annotations_root=annotations_root,
        iou_threshold=args.iou_threshold,
    )
    print(metrics)
    _write_text_output(output_path, [str(metrics)])


if __name__ == "__main__":
    main()