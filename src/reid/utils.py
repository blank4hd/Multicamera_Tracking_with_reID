import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
	"""Set seeds for python random, numpy, and torch (CPU + MPS/CUDA)."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


class AverageMeter:
	"""Tracks running average of a scalar over an epoch."""

	def __init__(self):
		self.reset()

	def reset(self):
		self.sum = 0.0
		self.count = 0
		self.avg = 0.0

	def update(self, value: float, n: int = 1):
		self.sum += float(value) * n
		self.count += n
		self.avg = self.sum / max(self.count, 1)


def save_checkpoint(
	path: str | Path,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler,
	epoch: int,
	best_rank1: float,
	extra: dict | None = None,
) -> None:
	"""Save full training state to path."""
	embedding_dim = getattr(model, "embedding_dim", None)
	state = {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"scheduler": scheduler.state_dict() if scheduler is not None else None,
		"epoch": epoch,
		"best_rank1": best_rank1,
		"embedding_dim": embedding_dim,
		"extra": extra or {},
	}
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	torch.save(state, path)


def load_checkpoint(
	path: str | Path,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer | None = None,
	scheduler=None,
	map_location: str | torch.device = "cpu",
) -> dict:
	"""Load checkpoint into model (and optimizer/scheduler if provided). Returns full state dict."""
	state = torch.load(path, map_location=map_location, weights_only=False)
	model.load_state_dict(state["model"])
	if optimizer is not None and "optimizer" in state:
		optimizer.load_state_dict(state["optimizer"])
	if scheduler is not None and state.get("scheduler") is not None:
		scheduler.load_state_dict(state["scheduler"])
	return state
