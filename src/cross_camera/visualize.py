import colorsys
from pathlib import Path
import numpy as np
import cv2


# Hand-picked palette of 8 visually distinct colors in BGR.
# Used by the highlight-reel visualization to make individual GIDs easy to track.
HIGHLIGHT_PALETTE_BGR = [
    (0, 0, 255),      # bright red
    (0, 255, 0),      # bright green
    (255, 0, 0),      # bright blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (240, 240, 240),  # off-white
    (147, 20, 255),   # hot pink
]


def color_for_global_id(global_id: int) -> tuple[int, int, int]:
    """
    Deterministic color from a global ID, returning a BGR tuple for OpenCV.
    Uses golden-ratio hue spread for good visual distinguishability.
    """
    hue = (global_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    # Convert to BGR uint8
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_global_id_boxes(
    frame: np.ndarray,
    boxes: list[dict],
    thickness: int = 4,
    font_scale: float = 1.0,
) -> np.ndarray:
    """
    Draw bboxes colored by global ID.
    boxes: list of dicts with keys 'x1', 'y1', 'x2', 'y2', 'global_id'.
    Returns annotated copy of frame.
    """
    out = frame.copy()
    for b in boxes:
        gid = int(b["global_id"])
        color = color_for_global_id(gid)
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"GID {gid}"
        # Filled background for label legibility
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_global_id_boxes_with_palette(
    frame: np.ndarray,
    boxes: list[dict],
    gid_to_color: dict[int, tuple[int, int, int]],
    thickness: int = 5,
    font_scale: float = 1.2,
) -> np.ndarray:
    """
    Like draw_global_id_boxes, but uses a caller-provided color mapping rather
    than the deterministic golden-ratio hash. Boxes whose global_id is not in
    gid_to_color are skipped entirely.

    Used by highlight-reel mode where only a small set of GIDs is rendered
    using a curated palette.
    """
    out = frame.copy()
    for b in boxes:
        gid = int(b["global_id"])
        if gid not in gid_to_color:
            continue
        color = gid_to_color[gid]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"GID {gid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(out, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
        # Use black text on lighter colors (yellow, cyan, white) for legibility,
        # white text on darker colors. Approximate by checking the green channel.
        text_color = (0, 0, 0) if color[1] > 200 or color[2] > 200 and color[0] > 200 else (255, 255, 255)
        cv2.putText(out, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
    return out


def make_grid_2x4(camera_frames: list[np.ndarray | None],
                  tile_w: int, tile_h: int,
                  pad: int = 4,
                  bg_color: tuple[int, int, int] = (32, 32, 32),
                  labels: list[str] | None = None) -> np.ndarray:
    """
    Tile up to 8 camera frames into a 2-row, 4-column grid.
    None entries are rendered as solid background.
    Each input frame is resized to (tile_w, tile_h).
    Adds a small label in the top-left of each tile if labels provided.
    Returns the final (2*tile_h + pad, 4*tile_w + 3*pad, 3) image.
    """
    rows = 2
    cols = 4
    while len(camera_frames) < rows * cols:
        camera_frames.append(None)
    grid_h = rows * tile_h + (rows - 1) * pad
    grid_w = cols * tile_w + (cols - 1) * pad
    canvas = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
    for i, fr in enumerate(camera_frames[:rows * cols]):
        r, c = divmod(i, cols)
        y = r * (tile_h + pad)
        x = c * (tile_w + pad)
        if fr is None:
            continue
        resized = cv2.resize(fr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[y:y + tile_h, x:x + tile_w] = resized
        if labels is not None and i < len(labels) and labels[i]:
            cv2.putText(canvas, labels[i], (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, labels[i], (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas
