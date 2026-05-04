import argparse
from pathlib import Path

import torch


def strip_checkpoint(in_path: Path, out_path: Path) -> dict:
    """Load a full checkpoint, strip to model weights only, and save it."""
    ck = torch.load(str(in_path), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f"Expected dict checkpoint at {in_path}, got {type(ck)}")
    if "model" not in ck:
        raise ValueError(f"Missing 'model' key in {in_path}")

    stripped = {"state_dict": ck["model"]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, str(out_path))

    meta = {
        "epoch": ck.get("epoch"),
        "best_rank1": ck.get("best_rank1"),
        "size_mb_in": in_path.stat().st_size / (1024 * 1024),
        "size_mb_out": out_path.stat().st_size / (1024 * 1024),
    }
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strip Re-ID training checkpoints to model-weights-only files for Hugging Face upload."
    )
    parser.add_argument("--primary-in", default="outputs/reid/best.pth")
    parser.add_argument("--ablation-in", default="outputs/reid_120ep/best.pth")
    parser.add_argument("--output-dir", default="huggingface")
    parser.add_argument("--skip-primary", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)

    if not args.skip_primary:
        meta = strip_checkpoint(Path(args.primary_in), out_dir / "best_60ep.pth")
        print(
            f"Primary (60-ep): {meta['size_mb_in']:.0f} MB -> {meta['size_mb_out']:.0f} MB | "
            f"epoch={meta['epoch']}, rank1={meta['best_rank1']:.2f}"
        )

    if not args.skip_ablation:
        meta = strip_checkpoint(Path(args.ablation_in), out_dir / "best_120ep.pth")
        print(
            f"Ablation (120-ep): {meta['size_mb_in']:.0f} MB -> {meta['size_mb_out']:.0f} MB | "
            f"epoch={meta['epoch']}, rank1={meta['best_rank1']:.2f}"
        )

    print(f"Stripped checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()