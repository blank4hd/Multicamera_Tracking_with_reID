"""TrackEval integration helpers for MOT17 evaluation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

# NumPy 1.24+ removed np.float, np.int, np.bool, np.object aliases.
# TrackEval still references them; patch only for the duration of the import,
# then remove the aliases so the rest of the process sees clean NumPy.
_np_patches: list[str] = []
for _alias, _target in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)
        _np_patches.append(_alias)

import trackeval  # noqa: E402

for _alias in _np_patches:
    try:
        delattr(np, _alias)
    except AttributeError:
        pass
del _np_patches, _alias, _target


def prepare_trackeval_layout(
    mot17_train_dir: Path,
    predictions_dir: Path,
    sequences: list[str],
    tracker_name: str,
    output_root: Path,
) -> Path:
    """Create TrackEval-required folder layout under output_root."""
    gt_root = output_root / "gt" / "mot_challenge"
    gt_split_root = gt_root / "MOT17-train"
    seqmaps_dir = gt_root / "seqmaps"
    trackers_data_dir = (
        output_root
        / "trackers"
        / "mot_challenge"
        / "MOT17-train"
        / tracker_name
        / "data"
    )

    gt_split_root.mkdir(parents=True, exist_ok=True)
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    trackers_data_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sequences:
        src_seq_dir = mot17_train_dir / seq_name
        dst_seq_gt_dir = gt_split_root / seq_name / "gt"
        dst_seq_gt_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_seq_dir / "gt" / "gt.txt", dst_seq_gt_dir / "gt.txt")
        shutil.copy2(src_seq_dir / "seqinfo.ini", gt_split_root / seq_name / "seqinfo.ini")

        src_pred_file = predictions_dir / f"{seq_name}.txt"
        if not src_pred_file.exists():
            raise FileNotFoundError(
                f"Missing prediction file for sequence '{seq_name}': {src_pred_file}"
            )
        shutil.copy2(src_pred_file, trackers_data_dir / f"{seq_name}.txt")

    seqmap_path = seqmaps_dir / "MOT17-train.txt"
    seqmap_lines = ["name", *sequences]
    seqmap_path.write_text("\n".join(seqmap_lines) + "\n", encoding="utf-8")

    return output_root


def _to_python_number(value: object) -> float | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return float(value)


def _extract_sequence_metrics(seq_metrics: dict, enabled_metrics: set[str]) -> dict[str, float | int]:
    """Pull scalar values from TrackEval's nested per-metric dicts."""
    extracted: dict[str, float | int] = {}

    if "HOTA" in enabled_metrics:
        hota_block = seq_metrics.get("HOTA", {})
        # HOTA returns arrays over alpha thresholds; average for a single scalar.
        for key in ["HOTA", "DetA", "AssA", "LocA"]:
            vals = hota_block.get(key)
            if vals is not None:
                extracted[key] = float(np.mean(np.asarray(vals, dtype=float)))

    if "CLEAR" in enabled_metrics:
        clear_block = seq_metrics.get("CLEAR", {})
        # Map TrackEval's CLR_* names to clean, report-friendly names.
        clear_map = {
            "MOTA": "MOTA",
            "MOTP": "MOTP",
            "IDSW": "IDSW",
            "CLR_FP": "FP",
            "CLR_FN": "FN",
            "CLR_TP": "TP",
            "MT": "MT",
            "ML": "ML",
            "Frag": "Frag",
        }
        for src_key, out_key in clear_map.items():
            if src_key in clear_block:
                extracted[out_key] = _to_python_number(clear_block[src_key])

    if "Identity" in enabled_metrics:
        identity_block = seq_metrics.get("Identity", {})
        for key in ["IDF1", "IDR", "IDP"]:
            if key in identity_block:
                extracted[key] = _to_python_number(identity_block[key])

    # Convert fraction metrics to percentages (TrackEval reports them as 0-1).
    for pct_key in ["HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDR", "IDP"]:
        if pct_key in extracted:
            extracted[pct_key] = float(extracted[pct_key]) * 100.0

    return extracted


def run_trackeval(
    trackeval_root: Path,
    tracker_name: str,
    benchmark: str = "MOT17",
    split: str = "train",
    metrics: list[str] | None = None,
) -> dict:
    """Run TrackEval and return parsed per-sequence and combined results."""
    if metrics is None:
        metrics = ["HOTA", "CLEAR", "Identity"]
    enabled_metrics = set(metrics)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False
    eval_config["OUTPUT_SUMMARY"] = False
    eval_config["OUTPUT_DETAILED"] = False
    eval_config["PLOT_CURVES"] = False
    eval_config["USE_PARALLEL"] = False

    ds_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_config["GT_FOLDER"] = os.fspath(trackeval_root / "gt" / "mot_challenge")
    ds_config["TRACKERS_FOLDER"] = os.fspath(trackeval_root / "trackers" / "mot_challenge")
    ds_config["BENCHMARK"] = benchmark
    ds_config["SPLIT_TO_EVAL"] = split
    ds_config["TRACKERS_TO_EVAL"] = [tracker_name]
    ds_config["DO_PREPROC"] = True
    ds_config["PRINT_CONFIG"] = False

    metric_classes = []
    if "HOTA" in enabled_metrics:
        metric_classes.append(trackeval.metrics.HOTA())
    if "CLEAR" in enabled_metrics:
        metric_classes.append(trackeval.metrics.CLEAR())
    if "Identity" in enabled_metrics:
        metric_classes.append(trackeval.metrics.Identity())

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_config)
    raw_results, _ = evaluator.evaluate([dataset], metric_classes)

    dataset_results = raw_results.get("MotChallenge2DBox")
    if dataset_results is None:
        raise RuntimeError("TrackEval results missing 'MotChallenge2DBox' dataset key")

    tracker_results = dataset_results.get(tracker_name)
    if tracker_results is None:
        raise RuntimeError(f"TrackEval results missing tracker key '{tracker_name}'")

    # # === TEMPORARY DEBUG: dump structure of one sequence's metrics ===
    # first_seq = next(iter(tracker_results))
    # sample = tracker_results[first_seq].get("pedestrian", {})
    # print(f"\n=== TrackEval result keys (debug) for {first_seq} ===")
    # for metric_name, metric_dict in sample.items():
    #     if isinstance(metric_dict, dict):
    #         print(f"  {metric_name}: {list(metric_dict.keys())}")
    #     else:
    #         print(f"  {metric_name}: {type(metric_dict).__name__}")
    # print("=== end debug ===\n")
    # # === END DEBUG ===

    parsed: dict[str, dict[str, float | int]] = {}
    for seq_name, seq_result in tracker_results.items():
        pedestrian_metrics = seq_result.get("pedestrian")
        if pedestrian_metrics is None:
            continue
        parsed[seq_name] = _extract_sequence_metrics(pedestrian_metrics, enabled_metrics)

    return parsed


def format_results_table(results: dict) -> str:
    """Format evaluation results into a fixed-width plain-text table."""

    def row_values(seq_name: str, metrics: dict[str, float | int]) -> list[str]:
        def fmt_pct(key: str) -> str:
            return f"{float(metrics.get(key, 0.0)):.1f}"

        def fmt_int(key: str) -> str:
            return f"{int(round(float(metrics.get(key, 0.0))))}"

        return [
            seq_name,
            fmt_pct("HOTA"),
            fmt_pct("MOTA"),
            fmt_pct("IDF1"),
            fmt_int("IDSW"),
            fmt_int("FP"),
            fmt_int("FN"),
            fmt_int("MT"),
            fmt_int("ML"),
            fmt_int("Frag"),
        ]

    headers = ["Sequence", "HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN", "MT", "ML", "Frag"]

    rows: list[list[str]] = []
    for seq_name in sorted(name for name in results if name != "COMBINED_SEQ"):
        rows.append(row_values(seq_name, results[seq_name]))

    combined_row = None
    if "COMBINED_SEQ" in results:
        combined_row = row_values("OVERALL", results["COMBINED_SEQ"])

    all_rows = [headers, *rows]
    if combined_row is not None:
        all_rows.append(combined_row)

    widths = [max(len(row[idx]) for row in all_rows) for idx in range(len(headers))]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows)

    if combined_row is not None:
        lines.append("-+-".join("-" * w for w in widths))
        lines.append(fmt_row(combined_row))

    return "\n".join(lines)
