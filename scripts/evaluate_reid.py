import argparse

import torch

from src.reid import (
	Market1501Dataset,
	ReIDModel,
	build_eval_transform,
	run_market1501_evaluation,
)
from src.utils.device import get_device


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate a trained Re-ID checkpoint on Market-1501.")
	parser.add_argument("--data-root", default="data/Market-1501-v15.09.15")
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--embedding-dim", type=int, default=256)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	device = get_device()
	print(f"Using device: {device}")

	query_ds = Market1501Dataset(args.data_root, split="query", transform=build_eval_transform())
	gallery_ds = Market1501Dataset(args.data_root, split="gallery", transform=build_eval_transform())
	state = torch.load(args.checkpoint, map_location=device, weights_only=False)
	model_state = state["model"]
	classifier_weight = model_state.get("classifier.weight")
	ckpt_num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else 0
	model = ReIDModel(num_classes=ckpt_num_classes, embedding_dim=args.embedding_dim, pretrained=False, last_stride=1)
	model.load_state_dict(model_state, strict=True)
	model.to(device)
	model.eval()

	results = run_market1501_evaluation(
		model,
		query_ds,
		gallery_ds,
		device,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	print(f"mAP: {results['mAP']:.2f}")
	print(f"Rank-1: {results['Rank-1']:.2f}")
	print(f"Rank-5: {results['Rank-5']:.2f}")
	print(f"Rank-10: {results['Rank-10']:.2f}")
	print(f"num_query: {results['num_query']}")


if __name__ == "__main__":
	main()
