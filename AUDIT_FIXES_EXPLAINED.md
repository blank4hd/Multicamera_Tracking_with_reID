# Audit Fixes Explained

This document explains the changes applied to the multi-camera tracking project and why each change was made.

## Dependency and Configuration Fixes

### `pyproject.toml`

- Added `trackeval` to the main dependencies because `src/evaluation/trackeval_runner.py` imports it directly for MOT evaluation.
- Removed `pyyaml` from the direct dependency list because the project does not import it directly. It may still be installed indirectly by Ultralytics.
- Moved `faiss-cpu` into an optional `faiss` extra because FAISS is not required for the default pipeline.

Why: the main install should include only required runtime dependencies. Optional dependencies should not be forced on every environment.

### `configs/default.yaml`

- Updated dataset paths to the real expected directory names:
  - `data/Market-1501-v15.09.15`
  - `data/MOT17/train`
  - `data/Wildtrack`
- Changed the detector model from `yolov8n.pt` to `yolov8m.pt`.
- Changed Re-ID `embedding_dim` from `512` to `256`.
- Changed Re-ID learning rate to `0.00035`.
- Replaced the generic tracking IoU key with `assoc_iou_threshold`.
- Replaced cross-camera `similarity_threshold` and `use_faiss` with `distance_threshold`.

Why: these values now match the final pipeline assumptions: 256-dimensional embeddings, YOLOv8m detection, clear tracker association naming, and no phantom FAISS toggle in the default config.

## Re-ID Fixes

### `src/reid/train.py`

- Changed LR decay milestones from `(40, 70)` to `(30, 50)`.

Why: training runs for 60 epochs, so epoch 70 was unreachable and the second decay would never happen.

### `src/reid/model.py`

- Added `self.embedding_dim = embedding_dim`.

Why: checkpoint saving now needs to record the model embedding size.

### `src/reid/utils.py`

- `save_checkpoint()` now stores `embedding_dim` in the checkpoint.

Why: checkpoint loading can now verify that the requested embedding dimension matches the trained model.

### `src/reid/feature_extractor.py`

- Added checkpoint `embedding_dim` validation.
- Switched model loading to `strict=True`.
- Reconstructed the checkpoint classifier shape before strict loading when `classifier.weight` exists.

Why: mismatched checkpoints should fail immediately with a clear `ValueError`. The classifier reconstruction keeps strict loading compatible with normal training checkpoints that include classifier weights.

### `scripts/evaluate_reid.py`

- Switched checkpoint loading to `strict=True`.
- Reconstructed the classifier shape from the checkpoint before loading.

Why: evaluation should catch checkpoint/model mismatches instead of silently ignoring unexpected or missing weights.

### `src/reid/dataset.py`

- Updated `RandomIdentitySampler` so rare identities are padded only for the shortage instead of replacing the whole sample with duplicates.

Why: this keeps all available samples for rare identities and only repeats what is necessary to satisfy `num_instances`.

## Tracking Fixes

### `src/tracking/kalman.py`

- Clamped the Kalman aspect ratio state `r` to at least `0.1` during prediction.

Why: during occlusion, the aspect ratio can drift into nonphysical values. The clamp prevents invalid box geometry from spreading through tracking.

### `src/tracking/deepsort.py`

- Defaulted `frame_idx` to `self.frame_count` when no frame index is passed.
- Reset `KalmanBoxTracker.count` inside `DeepSORTTracker.reset()`.

Why: emitted tracks now always have a frame index, and resetting DeepSORT between sequences also resets the underlying Kalman tracker ID counter.

## Evaluation and Data I/O Fixes

### `src/evaluation/trackeval_runner.py`

- Scoped the NumPy alias monkey-patch to only the `import trackeval` step.

Why: TrackEval still references removed NumPy aliases such as `np.float`, but the rest of the process should not keep those compatibility aliases after TrackEval has imported.

### `src/evaluation/mot_io.py`

- Added a docstring note explaining that MOT files are 1-indexed and internal tracker IDs are 0-indexed.

Why: this prevents accidental misuse when reading or writing MOT Challenge result files.

### `src/cross_camera/wildtrack_io.py`

- Removed the dead `WILDTRACK_FRAME_STEP` constant.
- Added comments explaining that frame alignment relies on sorted PNG filenames matching sorted annotation JSON files.

Why: Wildtrack image subsets are already pre-subsampled. The loader should not imply that it applies an additional frame step.

### `src/cross_camera/__init__.py`

- Removed the stale `WILDTRACK_FRAME_STEP` export.

Why: the constant no longer exists, so keeping the export would break package imports.

## Script Fixes

### `scripts/run_all_sequences.py`

- Replaced the single `--iou` flag with:
  - `--nms-iou` for YOLO detection NMS
  - `--assoc-iou` for SORT/DeepSORT association
- Passed `device` explicitly to `PersonDetector`.

Why: detector NMS IoU and tracker association IoU control different parts of the pipeline and should not share one flag.

### `scripts/run_wildtrack_per_camera.py`

- Replaced the single `--iou` flag with `--nms-iou` and `--assoc-iou`.
- Passed detector NMS IoU separately from tracker association IoU.
- Stored every frame in `tracks_per_frame`, even when there are no tracks.

Why: separate thresholds make the CLI unambiguous, and storing empty frames preserves frame-index alignment for Wildtrack outputs.

### `scripts/run_tracking.py`

- Passed `device` explicitly to `PersonDetector`.

Why: detector execution now follows the same device selection path as the rest of the script.

## Verification Performed

All verification was run in the `cvmct` conda environment.

```powershell
conda run -n cvmct python -m pip install -e .
```

Result: passed.

```powershell
conda run -n cvmct python -c "import trackeval; print('trackeval import ok')"
```

Result: passed.

```powershell
conda run -n cvmct python scripts\run_all_sequences.py --help
conda run -n cvmct python scripts\run_wildtrack_per_camera.py --help
```

Result: both help outputs show `--nms-iou` and `--assoc-iou`.

```powershell
conda run -n cvmct python -m pytest tests\ -v --basetemp data\pytest-basetemp-codex-audit
```

Result: `68 passed, 3 skipped, 3 warnings`.

Additional manual checks confirmed:

- New checkpoints save `embedding_dim`.
- Loading with the correct `embedding_dim` succeeds.
- Loading with the wrong `embedding_dim` raises `ValueError`.
- `DeepSORTTracker.reset()` resets `KalmanBoxTracker.count` to `0`.

## Environment Note

Plain `pytest tests\ -v` initially failed because Windows blocked access to the default pytest temp path and Ultralytics settings file. The final successful test run used a workspace-local pytest base temp directory.
