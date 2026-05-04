#!/usr/bin/env bash
set -e

echo "=================================================================="
echo "  MCTrack Re-ID Demo"
echo "  MSML 640 Final Project — Group 9"
echo "=================================================================="
echo ""

# Create demo virtual environment if it doesn't exist
if [ ! -d "demo_env" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv demo_env
else
    echo "[1/3] Reusing existing virtual environment..."
fi

# Activate the venv
source demo_env/bin/activate

# Install demo dependencies (idempotent — pip skips already-installed)
echo "[2/3] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    torchvision \
    huggingface_hub \
    Pillow \
    numpy

# Run the demo
echo "[3/3] Running Re-ID demo..."
echo ""
python3 demo_inference.py \
    --image-a examples/person_a.jpg \
    --image-b examples/person_b.jpg

echo ""
echo "=================================================================="
echo "  Demo complete. To run on your own images:"
echo "    source demo_env/bin/activate"
echo "    python3 demo_inference.py --image-a <path> --image-b <path>"
echo "=================================================================="
