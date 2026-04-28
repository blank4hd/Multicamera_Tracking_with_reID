"""Evaluate MOT17 predictions with TrackEval and print/save summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import format_results_table, prepare_trackeval_layout, run_trackeval

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
    parser = argparse.ArgumentParser(description="Evaluate MOT17 predictions via TrackEval")
    parser.add_argument("--mot17-dir", type=Path, default=Path("data/MOT17/train"))
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions/SORT"),
    )
    parser.add_argument("--tracker-name", type=str, default="SORT")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/trackeval"))
    parser.add_argument("--sequences", nargs="+", default=MOT17_TRAIN_SEQUENCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Preparing TrackEval folder layout...")
    trackeval_root = prepare_trackeval_layout(
        mot17_train_dir=args.mot17_dir,
        predictions_dir=args.predictions_dir,
        sequences=args.sequences,
        tracker_name=args.tracker_name,
        output_root=args.output_root,
    )

    print("Running TrackEval...")
    results = run_trackeval(trackeval_root=trackeval_root, tracker_name=args.tracker_name)

    table = format_results_table(results)
    print(table)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / f"{args.tracker_name}_results.txt"
    report_path.write_text(table + "\n", encoding="utf-8")
    print(f"Saved summary to: {report_path}")


if __name__ == "__main__":
    main()
