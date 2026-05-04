---
license: mit
library_name: pytorch
tags:
  - person-reidentification
  - person-reid
  - computer-vision
  - pytorch
  - resnet
  - bnneck
  - triplet-loss
  - multi-camera-tracking
datasets:
  - Market-1501
metrics:
  - mAP
  - Rank-1
---

# MCTrack Re-ID Model

Person re-identification model trained on Market-1501 as part of the
**MSML 640 (Computer Vision) Final Project** at the University of Maryland.
This model is the appearance backbone used in our multi-camera object tracking
system.

## Model Variants

This repo contains two checkpoints:

- **best_60ep.pth** (primary) - trained for 60 epochs.
  Used as the deployed Re-ID model in our final cross-camera demos due to
  qualitatively cleaner cluster outputs.
- **best_120ep.pth** (ablation) - same recipe, trained for 120 epochs.
  Slightly higher Re-ID accuracy but only marginal improvement on downstream
  tasks (see "Performance" below).

Both checkpoints contain only the model state_dict. No optimizer or scheduler
state.

## Architecture

- **Backbone:** ResNet-50 (ImageNet-initialized; final stride changed from
  2 to 1 to retain higher-resolution features)
- **Pooling:** Global average pooling
- **Neck:** BNNeck (Batch Normalization neck) - separates triplet-loss
  features from classification features
- **Embedding dimension:** 256
- **Total parameters:** ~25M

## Training Recipe

| Setting          | Value                                                            |
| ---------------- | ---------------------------------------------------------------- |
| Dataset          | Market-1501 (12,936 train images, 751 train IDs)                 |
| Identity sampler | P=16 IDs x K=4 instances per batch                               |
| Batch size       | 64                                                               |
| Optimizer        | Adam, lr=3.5e-4, weight_decay=5e-4                               |
| LR schedule      | Linear warmup (10 epochs), step decay (x0.1 at epochs 40 and 70) |
| Loss             | Combined CE (label smoothing 0.1) + triplet (soft margin)        |
| Image size       | 256x128                                                          |
| Augmentation     | Random horizontal flip, random erasing                           |

## Performance

### Standalone Re-ID (Market-1501)

| Variant   | mAP   | Rank-1 | Rank-5 | Rank-10 |
| --------- | ----- | ------ | ------ | ------- |
| 60-epoch  | 73.73 | 89.32  | 96.04  | 97.74   |
| 120-epoch | 75.15 | 90.66  | 96.79  | 98.04   |

### Downstream - Single-camera tracking (MOT17, with DeepSORT)

| Variant   | HOTA | MOTA | IDF1 | IDSW |
| --------- | ---- | ---- | ---- | ---- |
| 60-epoch  | 41.1 | 36.5 | 48.6 | 260  |
| 120-epoch | 41.0 | 36.6 | 48.8 | 246  |

### Downstream - Cross-camera tracking (Wildtrack, with ground-plane filter)

| Variant   | IDF1  | IDP   | IDR   |
| --------- | ----- | ----- | ----- |
| 60-epoch  | 16.99 | 23.80 | 13.21 |
| 120-epoch | 18.72 | 26.41 | 14.50 |

## Usage

Download and load with `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download
import torch

ckpt_path = hf_hub_download(
    repo_id="blank4hd/mctrack-reid",
    filename="best_60ep.pth",
)

state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model_state = state["state_dict"]
```

To use this with the model architecture, you need the `ReIDModel` class from
the project repository. A minimal standalone loader (`load_reid.py`) is
provided alongside this model card with the architecture definition inlined,
so the model can be used without cloning the full project.

## Intended Use

This model produces 256-dimensional appearance embeddings for cropped person
images. Two crops of the same person are expected to produce embeddings with
high cosine similarity; crops of different people produce embeddings with low
similarity.

**Suitable for:**

- Person re-identification in research / academic settings
- Appearance feature extraction in tracking pipelines (e.g., DeepSORT)
- Educational demonstration of metric learning

**Not suitable for:**

- Surveillance applications without explicit consent
- Identification of individuals across populations (high false-positive rate
  in cross-domain settings)
- Any use where reliability is safety-critical

## Limitations

- **Domain gap.** Trained on Market-1501 (Tsinghua University campus, ~5
  surveillance cameras). Performance degrades on outdoor pedestrian-square
  scenes (e.g., Wildtrack), where IDF1 drops to ~17-19% in cross-camera
  matching.
- **Person crops only.** Expects the input to be a tightly-cropped person
  image. Whole scenes or non-person inputs produce meaningless embeddings.
- **Resolution sensitive.** Trained at 256x128. Significantly different input
  resolutions will degrade quality.
- **No fairness audit.** Not evaluated for performance disparities across
  demographic groups.

## Training Details (compute and time)

- **Hardware:** Apple M4 Pro (MPS backend)
- **Per-epoch time:** ~46 seconds
- **Total training time:** 60-epoch ~46 min; 120-epoch ~92 min
- **Memory usage:** ~3 GB unified memory at batch size 64

## Citation

This work was completed for the MSML 640 final project, Spring 2026.

```
Group 9 - MSML 640 Final Project
Multi-Camera Object Tracking with Person Re-Identification
```

## Acknowledgments

- Architecture inspired by Luo et al. ("Bag of Tricks and a Strong Baseline
  for Deep Person Re-Identification", CVPR Workshop 2019)
- BNNeck design from the same paper
- Triplet loss formulation from Hermans et al. ("In Defense of the Triplet
  Loss for Person Re-Identification", arXiv 2017)
- Market-1501 dataset from Zheng et al. ("Scalable Person Re-Identification:
  A Benchmark", ICCV 2015)
