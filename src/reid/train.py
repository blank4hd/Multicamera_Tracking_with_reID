import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
	Market1501Dataset,
	RandomIdentitySampler,
	build_eval_transform,
	build_train_transform,
)
from .evaluate import run_market1501_evaluation
from .losses import CombinedReIDLoss
from .model import ReIDModel
from .utils import AverageMeter, save_checkpoint


@dataclass
class TrainConfig:
	data_root: str = "data/Market-1501-v15.09.15"
	output_dir: str = "outputs/reid"
	epochs: int = 60
	batch_size: int = 64
	num_instances: int = 4
	embedding_dim: int = 256
	margin: float | None = None
	ce_weight: float = 1.0
	triplet_weight: float = 1.0
	label_smoothing: float = 0.1
	base_lr: float = 3.5e-4
	weight_decay: float = 5e-4
	warmup_epochs: int = 10
	warmup_factor: float = 0.01
	lr_steps: tuple = (40, 70)
	lr_gamma: float = 0.1
	eval_every: int = 5
	num_workers: int = 4
	seed: int = 42


def make_warmup_step_lr(optimizer, warmup_epochs, warmup_factor, lr_steps, lr_gamma):
	"""
	Linear warmup from base_lr * warmup_factor up to base_lr over warmup_epochs,
	then step decay by lr_gamma at each milestone in lr_steps.
	Implemented as a LambdaLR that returns a multiplier on base_lr.
	"""
	def lr_lambda(epoch: int) -> float:
		if epoch < warmup_epochs:
			alpha = epoch / max(warmup_epochs, 1)
			return warmup_factor + (1.0 - warmup_factor) * alpha
		mult = 1.0
		for step in lr_steps:
			if epoch >= step:
				mult *= lr_gamma
		return mult

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch_idx, total_epochs):
	"""Run one epoch. Returns dict of average losses for the epoch."""
	model.train()
	meters = {"total": AverageMeter(), "ce": AverageMeter(), "triplet": AverageMeter()}
	start = time.time()
	pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1}/{total_epochs}", leave=False)
	for imgs, labels, _ in pbar:
		imgs = imgs.to(device, non_blocking=False)
		labels = labels.to(device, non_blocking=False)

		embedding, triplet_features, logits = model(imgs, return_logits=True)
		loss, components = loss_fn(triplet_features, logits, labels)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		bs = imgs.size(0)
		meters["total"].update(components["total"], bs)
		meters["ce"].update(components["ce"], bs)
		meters["triplet"].update(components["triplet"], bs)
		pbar.set_postfix({"loss": f"{components['total']:.3f}", "ce": f"{components['ce']:.3f}", "tri": f"{components['triplet']:.3f}"})

	return {
		"loss_total": meters["total"].avg,
		"loss_ce": meters["ce"].avg,
		"loss_triplet": meters["triplet"].avg,
		"epoch_time_sec": time.time() - start,
	}


def train(config: TrainConfig, device: torch.device, resume_from: str | None = None) -> dict:
	"""Full training loop. Returns the best metrics dict."""
	from .utils import set_seed

	set_seed(config.seed)

	output_dir = Path(config.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	log_path = output_dir / "train_log.csv"

	train_ds = Market1501Dataset(config.data_root, split="train", transform=build_train_transform())
	query_ds = Market1501Dataset(config.data_root, split="query", transform=build_eval_transform())
	gallery_ds = Market1501Dataset(config.data_root, split="gallery", transform=build_eval_transform())

	sampler = RandomIdentitySampler(train_ds, batch_size=config.batch_size, num_instances=config.num_instances)
	pin = (device.type == "cuda")
	train_loader = DataLoader(
		train_ds,
		batch_size=config.batch_size,
		sampler=sampler,
		num_workers=config.num_workers,
		pin_memory=pin,
		drop_last=True,
	)

	model = ReIDModel(
		num_classes=train_ds.num_classes,
		embedding_dim=config.embedding_dim,
		pretrained=True,
		last_stride=1,
	).to(device)
	loss_fn = CombinedReIDLoss(
		num_classes=train_ds.num_classes,
		margin=config.margin,
		ce_weight=config.ce_weight,
		triplet_weight=config.triplet_weight,
		label_smoothing_epsilon=config.label_smoothing,
	).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)
	scheduler = make_warmup_step_lr(
		optimizer,
		config.warmup_epochs,
		config.warmup_factor,
		config.lr_steps,
		config.lr_gamma,
	)

	start_epoch = 0
	best_rank1 = 0.0
	best_metrics = {"mAP": 0.0, "Rank-1": 0.0, "Rank-5": 0.0, "Rank-10": 0.0}

	if resume_from is not None:
		from .utils import load_checkpoint

		state = load_checkpoint(resume_from, model, optimizer, scheduler, map_location=device)
		start_epoch = state["epoch"] + 1
		best_rank1 = state["best_rank1"]
		print(f"Resumed from epoch {start_epoch}, best Rank-1 so far: {best_rank1:.2f}")

	if not log_path.exists():
		with log_path.open("w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["epoch", "lr", "loss_total", "loss_ce", "loss_triplet", "epoch_time_sec", "mAP", "Rank-1", "Rank-5", "Rank-10"])

	for epoch in range(start_epoch, config.epochs):
		current_lr = optimizer.param_groups[0]["lr"]
		train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, config.epochs)
		scheduler.step()

		eval_metrics = {"mAP": "", "Rank-1": "", "Rank-5": "", "Rank-10": ""}
		do_eval = ((epoch + 1) % config.eval_every == 0) or (epoch + 1 == config.epochs)
		if do_eval:
			print("  [Validation] Running Market-1501 evaluation...")
			eval_metrics = run_market1501_evaluation(
				model,
				query_ds,
				gallery_ds,
				device,
				batch_size=128,
				num_workers=config.num_workers,
			)
			print(
				f"  Epoch {epoch + 1}: mAP={eval_metrics['mAP']:.2f}  "
				f"Rank-1={eval_metrics['Rank-1']:.2f}  Rank-5={eval_metrics['Rank-5']:.2f}"
			)

			if eval_metrics["Rank-1"] > best_rank1:
				best_rank1 = eval_metrics["Rank-1"]
				best_metrics = {k: eval_metrics[k] for k in ["mAP", "Rank-1", "Rank-5", "Rank-10"]}
				save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, best_rank1, extra={"metrics": best_metrics})
				print(f"  New best Rank-1: {best_rank1:.2f} -> saved best.pth")

		save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, best_rank1, extra={"metrics": best_metrics})

		with log_path.open("a", newline="") as f:
			w = csv.writer(f)
			w.writerow([
				epoch + 1,
				f"{current_lr:.2e}",
				f"{train_metrics['loss_total']:.4f}",
				f"{train_metrics['loss_ce']:.4f}",
				f"{train_metrics['loss_triplet']:.4f}",
				f"{train_metrics['epoch_time_sec']:.1f}",
				eval_metrics.get("mAP", "") if isinstance(eval_metrics.get("mAP", ""), str) else f"{eval_metrics['mAP']:.2f}",
				eval_metrics.get("Rank-1", "") if isinstance(eval_metrics.get("Rank-1", ""), str) else f"{eval_metrics['Rank-1']:.2f}",
				eval_metrics.get("Rank-5", "") if isinstance(eval_metrics.get("Rank-5", ""), str) else f"{eval_metrics['Rank-5']:.2f}",
				eval_metrics.get("Rank-10", "") if isinstance(eval_metrics.get("Rank-10", ""), str) else f"{eval_metrics['Rank-10']:.2f}",
			])

	return best_metrics