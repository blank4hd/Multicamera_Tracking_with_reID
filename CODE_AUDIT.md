# Code Audit — Multicamera Tracking with Person Re-ID
**CV Group 9 · Audit Date: May 2, 2026**

---

## Table of Contents
- [Critical](#critical)
- [High](#high)
- [Medium](#medium)
- [Low](#low)
- [Summary Table](#summary-table)

---

## Critical

> Issues that **will break the pipeline** on a clean install or evaluation run.

---

### 1. `trackeval` missing from `pyproject.toml`

| | |
|---|---|
| **File** | `pyproject.toml` |
| **Impact** | `scripts/evaluate.py` crashes on import |

`src/evaluation/trackeval_runner.py` imports `trackeval` at the top level, but the package is **never listed** in `[project.dependencies]`. A fresh `pip install -e .` succeeds silently, then crashes the moment MOT17 evaluation is attempted.

**Fix:**
```toml
# pyproject.toml — add to [project] dependencies
"trackeval",
```

---

### 2. Single `--iou` flag controls two completely different thresholds

| | |
|---|---|
| **File** | `scripts/run_all_sequences.py` (lines ~54, 72–76) |
| **Impact** | Tuning tracking IoU silently degrades detector recall/precision |

The `--iou` argument is passed to **both** `PersonDetector(iou_threshold=args.iou)` (YOLO NMS) **and** the SORT/DeepSORT track association threshold. These are independent concepts with different typical values (NMS ≈ 0.45, association ≈ 0.3). Changing one always corrupts the other.

**Fix:** Split into two flags:
```python
parser.add_argument("--nms-iou",   type=float, default=0.45)
parser.add_argument("--assoc-iou", type=float, default=0.30)
```

---

## High

> Issues that cause **incorrect results or silent failures** at runtime.

---

### 3. `strict=False` checkpoint loading — silently accepts wrong model

| | |
|---|---|
| **Files** | `src/reid/feature_extractor.py` (line ~41), `scripts/evaluate_reid.py` (lines ~34–35) |
| **Impact** | Wrong `embedding_dim` loads without error; produces garbage embeddings |

```python
# Current — dangerous
model.load_state_dict(torch.load(path), strict=False)

# Fix
model.load_state_dict(torch.load(path), strict=True)
```

Also add an explicit check that the checkpoint's saved `embedding_dim` matches the extractor's config before loading.

---

### 4. `embedding_dim` mismatch — `default.yaml` says 512, all code defaults to 256

| | |
|---|---|
| **Files** | `configs/default.yaml` (line ~22), `src/reid/train.py` (line ~29), all training scripts |
| **Impact** | Dimension mismatch causes runtime crash or (with `strict=False`) silent garbage |

`default.yaml`:
```yaml
reid:
  embedding_dim: 512   # ← conflicts with code
```

`src/reid/train.py`:
```python
@dataclass
class TrainConfig:
    embedding_dim: int = 256   # ← actual default used everywhere
```

**Fix:** Pick one value (512 is the better choice for Market-1501) and update **all** of: `default.yaml`, `TrainConfig`, `scripts/train_reid.py`, `scripts/evaluate_reid.py`, and `feature_extractor.py`.

---

### 5. `similarity_threshold` in `default.yaml` uses wrong units

| | |
|---|---|
| **File** | `configs/default.yaml` (lines ~30–32) |
| **Impact** | Plugging the YAML value directly into the matcher produces the opposite threshold |

The YAML says `similarity_threshold: 0.7` (similarity space), but `CrossCameraMatcher` thresholds on **distance** (`1 − similarity`), defaulting to ~`0.35`. These are equivalent when the formula is `distance = 1 - similarity`, but **the names and units are opposite** — a value of `0.7` from YAML would be interpreted as a distance, which would admit almost no matches.

**Fix:**
```yaml
cross_camera:
  distance_threshold: 0.35   # rename and correct the value
```

---

### 6. Empty frames silently dropped from Wildtrack MOT output

| | |
|---|---|
| **File** | `scripts/run_wildtrack_per_camera.py` (lines ~88–101) |
| **Impact** | Frame-index misalignment during cross-camera evaluation |

Frames with no confirmed tracks are simply **not written** to the `.txt` output. Standard MOT format expects all frame indices to appear (with zero rows for empty frames). This causes alignment errors in any evaluator that iterates by frame index.

**Fix:**
```python
# After the tracking loop, ensure all processed frames have an entry
for fidx in all_frame_indices:
    if fidx not in tracks_per_frame:
        tracks_per_frame[fidx] = []
```

---

### 7. `frame_idx=None` leaks into output tracks in DeepSORT

| | |
|---|---|
| **File** | `src/tracking/deepsort.py` (lines ~169–171, 284–294) |
| **Impact** | Downstream code expecting `int` frame index gets `None` |

`DeepSORTTracker.update` defaults `frame_idx=None` and passes it directly into `TrackedBox`. Unlike `SORTTracker` which falls back to `self.frame_count`, DeepSORT propagates `None`.

**Fix:**
```python
def update(self, detections, frame_idx=None):
    if frame_idx is None:
        frame_idx = self.frame_count   # mirror SORTTracker behaviour
    ...
```

---

## Medium

> Issues that cause **subtle wrong results** or accumulate over long runs.

---

### 8. LR step at epoch 70 never fires — training ends at epoch 60

| | |
|---|---|
| **File** | `src/reid/train.py` (lines ~26, 37–38) |
| **Impact** | The second LR decay is skipped; final 20 epochs run at the wrong LR |

```python
@dataclass
class TrainConfig:
    epochs: int = 60
    lr_steps: tuple = (40, 70)   # epoch 70 is unreachable
```

The Phase 1 report describes decay "at epoch 30 and epoch 50", which matches 60-epoch training correctly.

**Fix:**
```python
lr_steps: tuple = (30, 50)   # matches the 60-epoch schedule
```

---

### 9. Kalman aspect ratio `r` is never clamped

| | |
|---|---|
| **File** | `src/tracking/kalman.py` (`predict`, lines ~94–96) |
| **Impact** | Long occlusions drift `r` to nonphysical values, corrupting bounding boxes |

Scale `s` is clamped after prediction, but aspect ratio `r` is not. Over many frames without a detection update, `r` can go negative or extreme, causing `_uvsr_to_xyxy` to return garbage.

**Fix:**
```python
# After existing scale clamp:
self.x[2] = max(self.x[2], 1e-6)   # scale (existing)
self.x[3] = max(self.x[3], 0.1)    # aspect ratio — add this
```

---

### 10. `KalmanBoxTracker.count` not reset between sequences in DeepSORT

| | |
|---|---|
| **File** | `src/tracking/deepsort.py` (`reset`, lines ~158–162) |
| **Impact** | Track IDs from earlier sequences bleed into later ones, breaking MOT evaluation |

`SORTTracker.reset()` correctly resets the global `KalmanBoxTracker.count`, but `DeepSORTTracker.reset()` does not. When processing multiple MOT17 sequences in one process, IDs keep incrementing.

**Fix:**
```python
def reset(self):
    self.tracks.clear()
    self.frame_count = 0
    KalmanBoxTracker.count = 0   # add this line
```

---

### 11. Duplicate images in triplet batches for rare identities

| | |
|---|---|
| **File** | `src/reid/dataset.py` — `RandomIdentitySampler.__iter__` (lines ~136–138) |
| **Impact** | `random.choices` (with replacement) duplicates crops, violating batch-hard triplet assumptions |

For identities with fewer than `num_instances` images, the same image can appear twice in one batch. Batch-hard triplet loss loses its "distinct negative/positive pair" guarantee.

**Fix:**
```python
if len(indices) >= num_instances:
    selected = random.sample(indices, num_instances)
else:
    # pad with repeats only when necessary, avoid full replacement
    selected = indices + random.choices(indices, k=num_instances - len(indices))
```

---

### 12. Global NumPy monkey-patch in `trackeval_runner.py` is fragile

| | |
|---|---|
| **File** | `src/evaluation/trackeval_runner.py` (lines ~11–15) |
| **Impact** | Restoring deprecated `np.float` / `np.int` globally can conflict with other libraries in the same process |

**Fix:** Scope the patch tightly around the import, or isolate TrackEval in a subprocess:
```python
import subprocess
subprocess.run(["python", "-m", "trackeval", ...], check=True)
```

---

### 13. `WILDTRACK_FRAME_STEP = 5` is dead code and a hidden assumption

| | |
|---|---|
| **File** | `src/cross_camera/wildtrack_io.py` (line ~12) |
| **Impact** | Frame alignment relies silently on sorted PNG ≡ sorted JSON order; breaks if dataset is reindexed |

The constant is defined but never used in any code path. Real frame alignment depends on `enumerate(..., start=1)` in the script.

**Fix:** Either use the constant explicitly in `list_camera_frames()`, or delete it and add a comment:
```python
# Frames are enumerated from sorted PNG filenames; step-5 subsampling
# is assumed to already be reflected in the stored Image_subsets.
```

---

### 14. `read_mot_results` assumes its own `+1` write convention

| | |
|---|---|
| **File** | `src/evaluation/mot_io.py` (line ~46) |
| **Impact** | Reading any standard MOT Challenge file shifts every track ID by 1 |

```python
track_id = int(float(parts[1])) - 1   # only correct for this repo's writer
```

MOT Challenge files use 1-indexed IDs. This code re-indexes to 0-based only because `write_mot_results` adds `+1`. Any external file will be read with all IDs off by one.

**Fix:** Standardise on 1-indexed IDs throughout (the MOT convention), and remove the `−1` / `+1` conversions.

---

## Low

> Code quality, maintainability, and documentation issues.

---

### 15. `faiss-cpu` and `pyyaml` listed as dependencies but never used

| | |
|---|---|
| **File** | `pyproject.toml` |

Neither package is imported anywhere in `src/` or `scripts/`. They add unnecessary install size and version-conflict surface.

**Fix:**
```toml
# Move faiss-cpu to optional since it's a planned feature
[project.optional-dependencies]
faiss = ["faiss-cpu"]

# Remove pyyaml unless you wire default.yaml loading into the code
```

---

### 16. `use_faiss: false` in `default.yaml` is a phantom config key

| | |
|---|---|
| **File** | `configs/default.yaml` (lines ~30–32) |

No Python code reads or acts on this flag. It implies FAISS is switchable, but the cross-camera matcher only uses brute-force numpy clustering.

**Fix:** Either implement FAISS in `src/cross_camera/matcher.py` and wire it to this key, or remove it from the config entirely.

---

### 17. `device` not passed explicitly to `PersonDetector` in `run_tracking.py`

| | |
|---|---|
| **Files** | `scripts/run_tracking.py` (line ~65) vs `scripts/run_wildtrack_per_camera.py` (line ~57) |

`run_wildtrack_per_camera.py` correctly passes `device=device` to `PersonDetector`. `run_tracking.py` relies on the internal `get_device()` fallback. Inconsistent — will silently break if `get_device()` behaviour ever changes.

**Fix:**
```python
# run_tracking.py
detector = PersonDetector(model_name=args.model, device=device, ...)
```

---

## Summary Table

| # | File | Issue | Severity |
|---|------|-------|----------|
| 1 | `pyproject.toml` | `trackeval` not in deps — `evaluate.py` crashes on import | **Critical** |
| 2 | `run_all_sequences.py` | Single `--iou` flag conflates YOLO NMS and track association IoU | **Critical** |
| 3 | `feature_extractor.py`, `evaluate_reid.py` | `strict=False` — wrong checkpoint loaded silently | **High** |
| 4 | `default.yaml` + `train.py` + scripts | `embedding_dim` 512 (YAML) vs 256 (code) — dimension mismatch | **High** |
| 5 | `default.yaml` | `similarity_threshold: 0.7` in wrong units vs matcher distance API | **High** |
| 6 | `run_wildtrack_per_camera.py` | Empty frames dropped from MOT output — frame-index misalignment | **High** |
| 7 | `deepsort.py` | `frame_idx=None` leaks into output tracks | **High** |
| 8 | `train.py` | LR step epoch 70 never fires — training ends at 60 | **Medium** |
| 9 | `kalman.py` | Aspect ratio `r` unclamped — drifts nonphysical over occlusions | **Medium** |
| 10 | `deepsort.py` | `KalmanBoxTracker.count` not reset between sequences | **Medium** |
| 11 | `dataset.py` | `random.choices` allows duplicate images in triplet batches | **Medium** |
| 12 | `trackeval_runner.py` | Global NumPy monkey-patch — fragile across libraries | **Medium** |
| 13 | `wildtrack_io.py` | `WILDTRACK_FRAME_STEP = 5` is dead code — silent frame alignment assumption | **Medium** |
| 14 | `mot_io.py` | `read_mot_results` hardcodes this repo's +1 ID convention | **Medium** |
| 15 | `pyproject.toml` | `faiss-cpu` / `pyyaml` listed but never imported | **Low** |
| 16 | `default.yaml` | `use_faiss` is a phantom config key — not wired to any code | **Low** |
| 17 | `run_tracking.py` | `device` not passed explicitly to `PersonDetector` | **Low** |

---

## Recommended Fix Order (Before Demo)

1. **Fix #1** — add `trackeval` to `pyproject.toml` so `evaluate.py` actually runs
2. **Fix #4** — reconcile `embedding_dim` to 512 everywhere so checkpoints load correctly
3. **Fix #3** — switch to `strict=True` so a bad checkpoint is caught immediately
4. **Fix #10** — reset `KalmanBoxTracker.count` so multi-sequence MOT17 eval produces clean IDs
5. **Fix #8** — correct the LR schedule milestones to `(30, 50)` for any future retraining runs

---

*Generated from full read-through of all `src/`, `scripts/`, `configs/`, `tests/` files.*
