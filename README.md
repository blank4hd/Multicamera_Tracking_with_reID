# Multi-Camera Object Tracking with Person Re-Identification

This project provides a scaffold for building a multi-camera object tracking pipeline with person re-identification. It is organized for modular development across detection, tracking, re-id, cross-camera association, and evaluation. The structure is intended to support experimentation, training, and reproducible benchmarking workflows.

## Setup

Install the project in editable mode:

```bash
pip install -e .
```

## Project Structure

```text
Multicamera_Tracking_with_reID/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── configs/
│   └── default.yaml
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   └── __init__.py
│   ├── tracking/
│   │   └── __init__.py
│   ├── reid/
│   │   └── __init__.py
│   ├── cross_camera/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── device.py
│   └── pipeline/
│       └── __init__.py
├── scripts/
│   ├── run_detection.py
│   ├── train_reid.py
│   ├── run_tracking.py
│   └── evaluate.py
├── notebooks/
│   └── .gitkeep
├── tests/
│   └── __init__.py
└── outputs/
    ├── checkpoints/
    │   └── .gitkeep
    ├── logs/
    │   └── .gitkeep
    └── videos/
        └── .gitkeep
```
