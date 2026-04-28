"""MOT Challenge result file I/O utilities."""

from __future__ import annotations

from pathlib import Path

from src.tracking.sort import TrackedBox


def write_mot_results(path: str | Path, tracks_per_frame: dict[int, list[TrackedBox]]) -> None:
    """Write tracked outputs to MOT Challenge CSV format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for frame_idx in sorted(tracks_per_frame):
        tracks = sorted(tracks_per_frame[frame_idx], key=lambda t: t.track_id)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            w = x2 - x1
            h = y2 - y1
            mot_track_id = track.track_id + 1
            lines.append(
                f"{int(frame_idx)},{mot_track_id},{x1:.6f},{y1:.6f},{w:.6f},{h:.6f},{track.confidence:.6f},-1,-1,-1"
            )

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_mot_results(path: str | Path) -> dict[int, list[dict]]:
    """Read MOT Challenge CSV format into frame-indexed track dictionaries."""
    input_path = Path(path)
    content = input_path.read_text(encoding="utf-8")

    parsed: dict[int, list[dict]] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 7:
            continue

        frame_idx = int(float(parts[0]))
        track_id = int(float(parts[1])) - 1
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        conf = float(parts[6])
        bbox = (x, y, x + w, y + h)

        parsed.setdefault(frame_idx, []).append(
            {
                "track_id": track_id,
                "bbox": bbox,
                "conf": conf,
            }
        )

    return parsed
