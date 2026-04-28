import numpy as np
import pytest
import torch

from src.reid import compute_distance_matrix, evaluate_market1501


def test_distance_matrix_shape():
	qf = torch.randn(5, 16)
	qf = qf / qf.norm(dim=1, keepdim=True)
	gf = torch.randn(20, 16)
	gf = gf / gf.norm(dim=1, keepdim=True)
	d = compute_distance_matrix(qf, gf)
	assert d.shape == (5, 20)
	assert (d >= -1e-5).all()
	assert (d <= 2.0 + 1e-5).all()


def test_perfect_evaluation():
	"""
	Construct a tiny dataset where each query has an identical gallery item
	(different cam) -> mAP and Rank-1 should be 100%.
	"""
	rng = np.random.RandomState(0)
	feats = torch.tensor(rng.randn(4, 16), dtype=torch.float32)
	feats = feats / feats.norm(dim=1, keepdim=True)
	qf = feats
	gf = feats.clone()
	q_pids = np.array([1, 2, 3, 4])
	g_pids = np.array([1, 2, 3, 4])
	q_camids = np.array([1, 1, 1, 1])
	g_camids = np.array([2, 2, 2, 2])
	metrics = evaluate_market1501(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=4)
	assert metrics["mAP"] > 99.0
	assert metrics["Rank-1"] > 99.0
	assert metrics["num_query"] == 4


def test_same_camera_filtered_out():
	"""
	Query and its only correct match are in same camera -> filtered out -> no valid query.
	"""
	feats = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
	qf = feats[:1]
	gf = feats[1:]
	q_pids = np.array([1])
	g_pids = np.array([1])
	q_camids = np.array([5])
	g_camids = np.array([5])
	metrics = evaluate_market1501(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=1)
	assert metrics["num_query"] == 0


def test_junk_gallery_excluded():
	"""
	pid == -1 in gallery is junk and shouldn't count as positive or negative.
	"""
	feats = torch.tensor(
		[
			[1.0, 0.0],
			[0.0, 1.0],
			[1.0, 0.0],
		],
		dtype=torch.float32,
	)
	qf = feats[:1]
	gf = feats[1:]
	q_pids = np.array([1])
	g_pids = np.array([-1, 1])
	q_camids = np.array([1])
	g_camids = np.array([2, 2])
	metrics = evaluate_market1501(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=2)
	assert metrics["Rank-1"] > 99.0


def test_warmup_scheduler():
	"""Warmup multiplier increases linearly then plateaus."""
	from src.reid import make_warmup_step_lr

	p = [torch.nn.Parameter(torch.zeros(1))]
	opt = torch.optim.Adam(p, lr=1.0)
	sched = make_warmup_step_lr(opt, warmup_epochs=10, warmup_factor=0.01, lr_steps=(40, 70), lr_gamma=0.1)
	assert abs(opt.param_groups[0]["lr"] - 0.01) < 1e-6
	sched.step()
	lr1 = opt.param_groups[0]["lr"]
	sched.step()
	lr2 = opt.param_groups[0]["lr"]
	assert lr2 > lr1
	for _ in range(8):
		sched.step()
	assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-3
	for _ in range(40):
		sched.step()
	assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-3