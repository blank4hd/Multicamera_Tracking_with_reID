# Codex Prompt — Fix All Audit Issues in Multicamera Tracking Project

Paste this entire prompt to Codex (or any coding agent). It describes every required change as an exact find-and-replace. Apply all changes together — this produces the final, corrected version of the project.

---

## Context

You are working on a Python project at the root directory `Multicamera_Tracking_with_reID`. The project is a multi-camera pedestrian tracking pipeline using YOLOv8 detection, SORT/DeepSORT tracking, ResNet-50 Re-ID, and cross-camera identity matching. Fix every issue listed below. Do not add extra features. Do not change logic that is not mentioned. Preserve all existing comments.

---

## Fix 1 — `pyproject.toml`: Add missing `trackeval` dep, move `faiss-cpu` to optional, remove unused `pyyaml`

**File:** `pyproject.toml`

Replace:
```toml
dependencies = [
    "torch",
    "torchvision",
    "ultralytics",
    "numpy",
    "scipy",
    "opencv-python",
    "pillow",
    "pyyaml",
    "tqdm",
    "faiss-cpu",
    "scikit-learn",
    "matplotlib",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "ipykernel",
    "jupyter",
]
```

With:
```toml
dependencies = [
    "torch",
    "torchvision",
    "ultralytics",
    "numpy",
    "scipy",
    "opencv-python",
    "pillow",
    "tqdm",
    "trackeval",
    "scikit-learn",
    "matplotlib",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "ipykernel",
    "jupyter",
]
faiss = [
    "faiss-cpu",
]
```

---

## Fix 2 — `configs/default.yaml`: Fix `embedding_dim`, fix threshold units, remove phantom key, use final detector

**File:** `configs/default.yaml`

Replace the entire file content with:
```yaml
device: "mps"

paths:
  market1501: "data/Market-1501-v15.09.15"
  mot17: "data/MOT17/train"
  wildtrack: "data/Wildtrack"
  checkpoints: "outputs/checkpoints"
  outputs: "outputs"

detection:
  model: "yolov8m.pt"
  conf_threshold: 0.5
  iou_threshold: 0.45
  person_class_id: 0

tracking:
  max_age: 30
  min_hits: 3
  assoc_iou_threshold: 0.3

reid:
  backbone: "resnet50"
  embedding_dim: 256
  batch_size: 64
  epochs: 60
  lr: 0.00035
  loss: "triplet"
  margin: 0.3

cross_camera:
  distance_threshold: 0.35
```

---

## Fix 3 — `src/reid/train.py`: Fix LR schedule milestones (epoch 70 is unreachable with 60-epoch training)

**File:** `src/reid/train.py`

Replace:
```python
	lr_steps: tuple = (40, 70)
```

With:
```python
	lr_steps: tuple = (30, 50)
```

---

## Fix 4 — `src/reid/feature_extractor.py`: Use `strict=True` so a mismatched checkpoint is caught immediately

**File:** `src/reid/feature_extractor.py`

Replace:
```python
        self.model = ReIDModel(num_classes=0, embedding_dim=embedding_dim, pretrained=False, last_stride=1)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
```

With:
```python
        self.model = ReIDModel(num_classes=0, embedding_dim=embedding_dim, pretrained=False, last_stride=1)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_dim = state.get("embedding_dim", embedding_dim)
        if ckpt_dim != embedding_dim:
            raise ValueError(
                f"Checkpoint embedding_dim={ckpt_dim} does not match requested embedding_dim={embedding_dim}. "
                "Update --embedding-dim or use the correct checkpoint."
            )
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()
```

---

## Fix 5 — `src/reid/utils.py`: Store `embedding_dim` in checkpoint so Fix 4 can validate it

**File:** `src/reid/utils.py`

Find the `save_checkpoint` function. Add `embedding_dim` to the saved dict. The function currently saves something like:

```python
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_rank1, extra=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_rank1": best_rank1,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
```

Change it to also save `embedding_dim` pulled from the model:

```python
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_rank1, extra=None):
    embedding_dim = getattr(model, "embedding_dim", None)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_rank1": best_rank1,
        "embedding_dim": embedding_dim,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
```

---

## Fix 6 — `src/reid/model.py`: Expose `embedding_dim` as an instance attribute so Fix 5 can read it

**File:** `src/reid/model.py`

In the `ReIDModel.__init__` method, after `self.embedding_dim = embedding_dim` (or wherever `embedding_dim` is first assigned as a constructor argument), ensure the line exists:

```python
self.embedding_dim = embedding_dim
```

If the line already exists, no change is needed. If the parameter is only used locally without being stored on `self`, add `self.embedding_dim = embedding_dim` at the top of `__init__`.

---

## Fix 7 — `src/tracking/kalman.py`: Clamp aspect ratio `r` to prevent nonphysical drift during occlusion

**File:** `src/tracking/kalman.py`

In the `predict` method, find:
```python
        if self.x[2, 0] <= _EPS:
            self.x[2, 0] = _EPS

        self.age += 1
```

Replace with:
```python
        if self.x[2, 0] <= _EPS:
            self.x[2, 0] = _EPS
        if self.x[3, 0] < 0.1:
            self.x[3, 0] = 0.1

        self.age += 1
```

---

## Fix 8 — `src/tracking/deepsort.py` (part A): Default `frame_idx` to `self.frame_count` instead of `None`

**File:** `src/tracking/deepsort.py`

Find:
```python
    def update(self, frame, detections, frame_idx: int | None = None) -> list[TrackedBox]:
        self.frame_count += 1
```

Replace with:
```python
    def update(self, frame, detections, frame_idx: int | None = None) -> list[TrackedBox]:
        self.frame_count += 1
        if frame_idx is None:
            frame_idx = self.frame_count
```

---

## Fix 9 — `src/tracking/deepsort.py` (part B): Reset `KalmanBoxTracker.count` when tracker is reset between sequences

**File:** `src/tracking/deepsort.py`

Find:
```python
    def reset(self):
        self.tracks = []
        self.frame_count = 0
        self._next_id = 0
        self._track_history = {}
```

Replace with:
```python
    def reset(self):
        self.tracks = []
        self.frame_count = 0
        self._next_id = 0
        self._track_history = {}
        KalmanBoxTracker.count = 0
```

---

## Fix 10 — `src/reid/dataset.py`: Fix `RandomIdentitySampler` to avoid pure duplicate-replacement for rare identities

**File:** `src/reid/dataset.py`

Find:
```python
                if len(indices) >= self.num_instances:
                    chosen = random.sample(indices, self.num_instances)
                else:
                    chosen = random.choices(indices, k=self.num_instances)
```

Replace with:
```python
                if len(indices) >= self.num_instances:
                    chosen = random.sample(indices, self.num_instances)
                else:
                    # Pad with random repeats only for the shortage — avoids
                    # sampling an identity as its own positive/negative pair.
                    shortage = self.num_instances - len(indices)
                    chosen = indices + random.choices(indices, k=shortage)
                    random.shuffle(chosen)
```

---

## Fix 11 — `src/evaluation/trackeval_runner.py`: Scope the NumPy monkey-patch so it only applies during the `import trackeval` statement

**File:** `src/evaluation/trackeval_runner.py`

Find:
```python
# NumPy 1.24+ removed np.float, np.int, np.bool, np.object aliases.
# TrackEval still uses np.float internally; restore the aliases before importing it.
for _alias, _target in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import trackeval
```

Replace with:
```python
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
```

---

## Fix 12 — `src/cross_camera/wildtrack_io.py`: Remove dead constant and document the frame-alignment assumption

**File:** `src/cross_camera/wildtrack_io.py`

Find:
```python
WILDTRACK_FRAME_STEP = 5
```

Replace with:
```python
# Frame alignment relies on sorted PNG filenames matching sorted annotation JSON files.
# Wildtrack stores pre-subsampled images (every 5th frame of the original 60 fps stream)
# in Image_subsets/; no additional step is applied when iterating frames here.
```

---

## Fix 13 — `src/evaluation/mot_io.py`: Add a clear comment documenting the internal 0-indexed / MOT 1-indexed convention

**File:** `src/evaluation/mot_io.py`

Find:
```python
def read_mot_results(path: str | Path) -> dict[int, list[dict]]:
    """Read MOT Challenge CSV format into frame-indexed track dictionaries."""
```

Replace with:
```python
def read_mot_results(path: str | Path) -> dict[int, list[dict]]:
    """Read MOT Challenge CSV format into frame-indexed track dictionaries.

    Convention: MOT files are 1-indexed (track_id >= 1).  This function
    converts to 0-indexed track IDs to match the internal tracker convention
    used by write_mot_results and all tracker classes.  Do not pass raw
    external MOT files here without first verifying they are 1-indexed.
    """
```

---

## Fix 14 — `scripts/run_all_sequences.py`: Split single `--iou` flag into separate `--nms-iou` (detector) and `--assoc-iou` (tracker)

**File:** `scripts/run_all_sequences.py`

In `parse_args()`, find:
```python
    parser.add_argument("--iou", type=float, default=0.3)
```

Replace with:
```python
    parser.add_argument("--nms-iou", type=float, default=0.45,
                        help="YOLO NMS IoU threshold for detection (suppresses overlapping boxes).")
    parser.add_argument("--assoc-iou", type=float, default=0.30,
                        help="Tracker association IoU threshold (SORT gate / DeepSORT IoU fallback).")
```

Then find the detector instantiation:
```python
    detector = PersonDetector(model_name=args.model, conf_threshold=args.conf, iou_threshold=args.iou)
```

Replace with:
```python
    detector = PersonDetector(model_name=args.model, device=device,
                              conf_threshold=args.conf, iou_threshold=args.nms_iou)
```

Then find every occurrence of `args.iou` that is used for the **tracker** (DeepSORT `iou_threshold_fallback` and SORT `iou_threshold`), which looks like:

```python
                iou_threshold_fallback=args.iou,
```

Replace with:
```python
                iou_threshold_fallback=args.assoc_iou,
```

And find:
```python
            tracker = SORTTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou)
```

Replace with:
```python
            tracker = SORTTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.assoc_iou)
```

---

## Fix 15 — `scripts/run_wildtrack_per_camera.py` (part A): Always store frames (even empty ones) to keep frame-index alignment

**File:** `scripts/run_wildtrack_per_camera.py`

Find:
```python
            tracks = tracker.update(frame, detections, frame_idx=fidx)
            if tracks:
                tracks_per_frame[fidx] = tracks
```

Replace with:
```python
            tracks = tracker.update(frame, detections, frame_idx=fidx)
            tracks_per_frame[fidx] = tracks  # store even when empty — preserves frame-index alignment
```

---

## Fix 16 — `scripts/run_wildtrack_per_camera.py` (part B): Also split `--iou` into `--nms-iou` and `--assoc-iou`

**File:** `scripts/run_wildtrack_per_camera.py`

In `build_arg_parser()`, find:
```python
    parser.add_argument("--iou", type=float, default=0.3)
```

Replace with:
```python
    parser.add_argument("--nms-iou", type=float, default=0.45,
                        help="YOLO NMS IoU threshold.")
    parser.add_argument("--assoc-iou", type=float, default=0.30,
                        help="Tracker association IoU threshold.")
```

Find:
```python
    detector = PersonDetector(model_name=args.model, device=device, conf_threshold=args.conf, iou_threshold=args.iou)
```

Replace with:
```python
    detector = PersonDetector(model_name=args.model, device=device,
                              conf_threshold=args.conf, iou_threshold=args.nms_iou)
```

Find:
```python
            iou_threshold_fallback=args.iou,
```

Replace with:
```python
            iou_threshold_fallback=args.assoc_iou,
```

---

## Fix 17 — `scripts/run_tracking.py`: Pass `device` explicitly to `PersonDetector`

**File:** `scripts/run_tracking.py`

Find the `PersonDetector(...)` instantiation. It will look like:
```python
    detector = PersonDetector(model_name=args.model, conf_threshold=args.conf, iou_threshold=args.iou)
```

Replace with:
```python
    detector = PersonDetector(model_name=args.model, device=device,
                              conf_threshold=args.conf, iou_threshold=args.iou)
```

If this script also has a single `--iou` flag used for both detector and tracker, apply the same `--nms-iou` / `--assoc-iou` split described in Fix 14.

---

## Fix 18 — `scripts/evaluate_reid.py`: Use `strict=True` for checkpoint loading

**File:** `scripts/evaluate_reid.py`

Find any line that contains:
```python
load_state_dict(state["model"], strict=False)
```

Replace with:
```python
load_state_dict(state["model"], strict=True)
```

---

## Verification checklist

After applying all fixes, confirm:

1. `pip install -e .` succeeds and `import trackeval` works without error.
2. `python scripts/run_all_sequences.py --help` shows `--nms-iou` and `--assoc-iou` as separate flags.
3. `python scripts/run_wildtrack_per_camera.py --help` shows `--nms-iou` and `--assoc-iou` as separate flags.
4. Training a new checkpoint saves `embedding_dim` in the `.pth` file.
5. Loading a checkpoint with wrong `embedding_dim` raises a `ValueError` immediately.
6. `DeepSORTTracker.reset()` resets `KalmanBoxTracker.count` to 0 — verify by checking `KalmanBoxTracker.count == 0` after calling `tracker.reset()`.
7. `configs/default.yaml` has no `use_faiss` key and no `pyyaml` or `faiss-cpu` in main deps.
8. All existing unit tests in `tests/` still pass: `pytest tests/ -v`.
