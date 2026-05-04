"""Standalone Re-ID inference demo.

Downloads the trained MCTrack Re-ID model from Hugging Face and computes
the cosine similarity between two cropped pedestrian images. Same person
typically scores 0.5-0.85; different people typically score below 0.3.

Usage:
    python3 demo_inference.py --image-a path/A.jpg --image-b path/B.jpg
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


MODEL_REPO = "blank4hd/mctrack-reid"
MODEL_FILE = "best_60ep.pth"
EMBEDDING_DIM = 256
NUM_CLASSES_TRAIN = 751  # Market-1501 train identities


class GeneralizedMeanPooling(nn.Module):
    """Generalized mean pooling, learnable exponent p (BoT-style)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x


class ReIDModel(nn.Module):
    """ResNet-50 + BNNeck Re-ID model.

    Backbone: ResNet-50 with final stride 1 (BoT-style)
    Pooling: Generalized mean pooling
    Neck: BNNeck (BatchNorm after the linear bottleneck)
    Returns L2-normalized 256-dim embeddings during inference.
    """

class ReIDModel(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.gem = GeneralizedMeanPooling(p=3.0)
        self.bottleneck = nn.Linear(2048, embedding_dim, bias=False)
        self.bn_neck = nn.BatchNorm1d(embedding_dim)
        # No classifier — only needed at training time, not inference
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.gem(feat).flatten(1)
        feat = self.bottleneck(feat)
        feat = self.bn_neck(feat)
        feat = F.normalize(feat, dim=1, p=2)
        return feat


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a model-ready tensor."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return IMAGE_TRANSFORM(image).unsqueeze(0)


def load_model_from_hub(device: str = "cpu") -> ReIDModel:
    print(f"  Downloading model from {MODEL_REPO} (cached after first run)...")
    ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = ReIDModel(embedding_dim=EMBEDDING_DIM)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    # Drop classifier weights — they're only used at training time
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params / 1e6:.1f}M parameters, {EMBEDDING_DIM}-dim embeddings")
    return model


def compare_images(
    model: ReIDModel,
    image_a_path: str,
    image_b_path: str,
    device: str = "cpu",
) -> tuple[float, str]:
    """Run two images through the model and return (similarity, verdict)."""
    img_a = Image.open(image_a_path)
    img_b = Image.open(image_b_path)
    x_a = preprocess_image(img_a).to(device)
    x_b = preprocess_image(img_b).to(device)
    with torch.no_grad():
        emb_a = model(x_a)
        emb_b = model(x_b)
    similarity = float(F.cosine_similarity(emb_a, emb_b).item())
    if similarity > 0.78:
        verdict = "Likely SAME person"
    elif similarity > 0.68:
        verdict = "Uncertain — possibly same person"
    else:
        verdict = "Likely DIFFERENT people"
    return similarity, verdict


def main():
    parser = argparse.ArgumentParser(
        description="MCTrack Re-ID inference demo. Compares two pedestrian images.",
    )
    parser.add_argument(
        "--image-a", required=True,
        help="Path to first cropped pedestrian image",
    )
    parser.add_argument(
        "--image-b", required=True,
        help="Path to second cropped pedestrian image",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device (cpu, cuda, or mps)",
    )
    args = parser.parse_args()

    img_a = Path(args.image_a)
    img_b = Path(args.image_b)
    if not img_a.exists():
        print(f"Error: image not found: {img_a}", file=sys.stderr)
        sys.exit(1)
    if not img_b.exists():
        print(f"Error: image not found: {img_b}", file=sys.stderr)
        sys.exit(1)

    print("Loading Re-ID model...")
    model = load_model_from_hub(device=args.device)

    print(f"\nComparing:")
    print(f"  Image A: {img_a}")
    print(f"  Image B: {img_b}")

    similarity, verdict = compare_images(model, str(img_a), str(img_b), args.device)

    print(f"\n----- Result -----")
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  Verdict:           {verdict}")
    print(f"------------------")


if __name__ == "__main__":
    main()
