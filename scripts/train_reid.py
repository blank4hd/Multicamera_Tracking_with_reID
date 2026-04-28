import argparse

import torch

from src.reid import TrainConfig, train
from src.utils.device import get_device


def parse_margin(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Re-ID model on Market-1501.")
    parser.add_argument("--data-root", default="data/Market-1501-v15.09.15")
    parser.add_argument("--output-dir", default="outputs/reid")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument(
        "--margin",
        default="none",
        help='Triplet margin as a float string, or "none"/empty for soft margin default.',
    )
    parser.add_argument("--base-lr", type=float, default=3.5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print("MPS note: fp32 is used (no autocast); num_workers above 4 may not help on M-series.")

    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        embedding_dim=args.embedding_dim,
        margin=parse_margin(args.margin),
        base_lr=args.base_lr,
        warmup_epochs=args.warmup_epochs,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("Training config:")
    for field_name, value in config.__dict__.items():
        print(f"  {field_name}: {value}")

    results = train(config, device, resume_from=args.resume)
    print("Final best metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
