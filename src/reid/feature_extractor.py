from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import REID_INPUT_HEIGHT, REID_INPUT_WIDTH, build_eval_transform
from .model import ReIDModel


class ReIDFeatureExtractor:
    """
    Extracts L2-normalized appearance embeddings from bbox crops.

    Loads a trained ReIDModel checkpoint, runs inference in eval mode,
    handles batched extraction with the standard Market-1501 preprocessing.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        embedding_dim: int = 256,
        device: torch.device | None = None,
        batch_size: int = 32,
    ):
        """
        Args:
            checkpoint_path: path to .pth file from training (e.g. outputs/reid/best.pth)
            embedding_dim: must match training setting
            device: defaults to get_device() from src.utils.device
            batch_size: max number of crops per forward pass
        """
        from src.utils.device import get_device

        self.device = device if device is not None else get_device()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_dim = state.get("embedding_dim", embedding_dim)
        if ckpt_dim != embedding_dim:
            raise ValueError(
                f"Checkpoint embedding_dim={ckpt_dim} does not match requested embedding_dim={embedding_dim}. "
                "Update --embedding-dim or use the correct checkpoint."
            )
        model_state = state["model"]
        classifier_weight = model_state.get("classifier.weight")
        ckpt_num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else 0
        self.model = ReIDModel(num_classes=ckpt_num_classes, embedding_dim=embedding_dim, pretrained=False, last_stride=1)
        self.model.load_state_dict(model_state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_eval_transform()

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: list | np.ndarray) -> np.ndarray:
        """
        Extract embeddings for a list of bboxes from a single frame.

        Args:
            frame: HxWx3 numpy array in BGR (OpenCV convention).
            bboxes: list of (x1, y1, x2, y2) tuples or array (N, 4) in pixel coords (xyxy).

        Returns:
            (N, embedding_dim) numpy array of L2-normalized features.
            Returns empty array shape (0, embedding_dim) if bboxes is empty.
        """
        if len(bboxes) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        height, width = frame.shape[:2]
        frame_rgb = frame[..., ::-1]

        tensors = []
        for x1, y1, x2, y2 in bboxes:
            xa = max(0, int(round(x1)))
            ya = max(0, int(round(y1)))
            xb = min(width, int(round(x2)))
            yb = min(height, int(round(y2)))
            if xb <= xa or yb <= ya:
                crop = np.zeros((REID_INPUT_HEIGHT, REID_INPUT_WIDTH, 3), dtype=np.uint8)
                pil = Image.fromarray(crop)
            else:
                crop = frame_rgb[ya:yb, xa:xb]
                pil = Image.fromarray(crop)
            tensors.append(self.transform(pil))

        all_features = []
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i : i + self.batch_size]).to(self.device)
            embedding, _ = self.model(batch)
            all_features.append(embedding.cpu().numpy())

        return np.concatenate(all_features, axis=0).astype(np.float32)
