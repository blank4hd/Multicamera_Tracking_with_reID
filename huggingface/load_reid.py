from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


class ReIDModel(nn.Module):
    """Minimal Re-ID model matching the trained checkpoint architecture.

    ResNet-50 backbone (final stride 1) -> global average pooling -> BNNeck.
    Returns 256-dim L2-normalized embeddings during inference.
    """

    def __init__(self, num_classes: int = 751, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 2048
        self.bnneck = nn.BatchNorm1d(self.feature_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        self.embedding_layer = nn.Linear(self.feature_dim, embedding_dim, bias=False)
        nn.init.kaiming_normal_(self.embedding_layer.weight, mode="fan_out")

        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an L2-normalized embedding of shape (N, embedding_dim)."""
        feat_map = self.backbone(x)
        pooled = self.global_pool(feat_map).flatten(1)
        bn_features = self.bnneck(pooled)
        embedding = self.embedding_layer(bn_features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> ReIDModel:
    """Load the trained Re-ID model from a stripped checkpoint."""
    model = ReIDModel(num_classes=751, embedding_dim=256)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing_keys:
        print(f"Warning: missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Warning: unexpected keys: {len(unexpected_keys)} keys")
    model.eval()
    model.to(device)
    return model


_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image to model input format."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return _PREPROCESS(image).unsqueeze(0)


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """Cosine similarity between two L2-normalized embeddings, in [-1, 1]."""
    return float(F.cosine_similarity(emb_a, emb_b).item())


if __name__ == "__main__":
    import sys

    from huggingface_hub import hf_hub_download

    print("Downloading model from Hugging Face...")
    ckpt = hf_hub_download(repo_id="blank4hd/mctrack-reid", filename="best_60ep.pth")
    model = load_model(ckpt)
    print(f"Model loaded. Embedding dim: 256, parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if len(sys.argv) >= 3:
        img_a = Image.open(sys.argv[1])
        img_b = Image.open(sys.argv[2])
        x_a = preprocess_image(img_a)
        x_b = preprocess_image(img_b)
        with torch.no_grad():
            emb_a = model(x_a)
            emb_b = model(x_b)
        sim = cosine_similarity(emb_a, emb_b)
        print(f"Cosine similarity: {sim:.4f}")
        if sim > 0.7:
            print("Likely SAME person")
        elif sim > 0.4:
            print("Possibly same person (uncertain)")
        else:
            print("Likely DIFFERENT people")
    else:
        print("Usage: python load_reid.py <image_a.jpg> <image_b.jpg>")