import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, dataloader, device):
	"""
	Run model in eval mode over dataloader, returning:
	  features: (N, D) tensor on CPU (already L2-normalized — model handles this)
	  pids:     (N,) numpy array of person IDs
	  camids:   (N,) numpy array of camera IDs
	"""
	model.eval()
	feats_chunks, pids_chunks, camids_chunks = [], [], []
	for imgs, pids, camids in tqdm(dataloader, desc="Extracting", leave=False):
		imgs = imgs.to(device, non_blocking=False)
		embedding, _ = model(imgs)
		feats_chunks.append(embedding.cpu())
		pids_chunks.append(pids.numpy() if torch.is_tensor(pids) else np.asarray(pids))
		camids_chunks.append(camids.numpy() if torch.is_tensor(camids) else np.asarray(camids))
	feats = torch.cat(feats_chunks, dim=0)
	pids = np.concatenate(pids_chunks)
	camids = np.concatenate(camids_chunks)
	return feats, pids, camids


def compute_distance_matrix(qf: torch.Tensor, gf: torch.Tensor) -> torch.Tensor:
	"""
	Cosine distance between query and gallery features.
	Both must already be L2-normalized.
	Returns (Q, G) tensor where smaller = more similar.
	distance = 1 - cosine_similarity = 1 - q . g (since both unit-norm).
	"""
	return 1.0 - qf @ gf.t()


def evaluate_market1501(
	qf: torch.Tensor, q_pids: np.ndarray, q_camids: np.ndarray,
	gf: torch.Tensor, g_pids: np.ndarray, g_camids: np.ndarray,
	max_rank: int = 50,
) -> dict:
	"""
	Standard Market-1501 evaluation protocol.

	For each query:
	  1. Compute distances to all gallery items.
	  2. Sort gallery indices by ascending distance.
	  3. Build a "valid" mask excluding:
	     - gallery items with same pid AND same camid as query (junk by protocol)
	     - gallery items with pid == -1 (junk class)
	  4. Compute AP over the valid ranked list (only positives = matching pid).
	  5. Compute CMC: 1 if any positive appears in top-k of valid list, else 0.
	Queries with zero valid positives are excluded from averages.

	Returns dict with:
	  - mAP: float (percentage)
	  - Rank-1, Rank-5, Rank-10: float (percentages)
	  - num_query: int (queries actually evaluated)
	"""
	distmat = compute_distance_matrix(qf, gf).numpy()
	num_q, num_g = distmat.shape
	if num_g < max_rank:
		max_rank = num_g

	indices = np.argsort(distmat, axis=1)
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

	all_cmc = []
	all_AP = []
	num_valid_q = 0

	for q_idx in range(num_q):
		q_pid = q_pids[q_idx]
		q_camid = q_camids[q_idx]

		order = indices[q_idx]
		remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
		remove |= (g_pids[order] == -1)
		keep = ~remove

		valid_matches = matches[q_idx][keep]
		if not valid_matches.any():
			continue

		cmc = valid_matches.cumsum()
		cmc[cmc > 1] = 1
		if len(cmc) < max_rank:
			cmc = np.pad(cmc, (0, max_rank - len(cmc)), mode="edge")
		all_cmc.append(cmc[:max_rank])

		num_rel = valid_matches.sum()
		tp_cumsum = valid_matches.cumsum()
		precision_at_k = tp_cumsum / (np.arange(len(valid_matches)) + 1.0)
		ap = (precision_at_k * valid_matches).sum() / num_rel
		all_AP.append(ap)
		num_valid_q += 1

	if num_valid_q == 0:
		return {"mAP": 0.0, "Rank-1": 0.0, "Rank-5": 0.0, "Rank-10": 0.0, "num_query": 0}

	all_cmc = np.asarray(all_cmc).astype(np.float32)
	cmc_curve = all_cmc.mean(axis=0)
	if len(cmc_curve) < 10:
		cmc_curve = np.pad(cmc_curve, (0, 10 - len(cmc_curve)), mode="edge")
	mAP = float(np.mean(all_AP))

	return {
		"mAP": mAP * 100.0,
		"Rank-1": float(cmc_curve[0]) * 100.0,
		"Rank-5": float(cmc_curve[4]) * 100.0,
		"Rank-10": float(cmc_curve[9]) * 100.0,
		"num_query": num_valid_q,
	}


def run_market1501_evaluation(
	model,
	query_dataset,
	gallery_dataset,
	device,
	batch_size: int = 128,
	num_workers: int = 4,
) -> dict:
	"""End-to-end: extract features for both splits, run evaluate_market1501."""
	pin = (device.type == "cuda")
	q_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False,
						  num_workers=num_workers, pin_memory=pin)
	g_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False,
						  num_workers=num_workers, pin_memory=pin)
	qf, q_pids, q_camids = extract_features(model, q_loader, device)
	gf, g_pids, g_camids = extract_features(model, g_loader, device)
	return evaluate_market1501(qf, q_pids, q_camids, gf, g_pids, g_camids)