"""Plot Re-ID training curves from train_log.csv for the Phase 1 report."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(csv_path: Path):
    epochs, loss_total, loss_ce, loss_triplet = [], [], [], []
    eval_epochs, mAP, rank1 = [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            epochs.append(epoch)
            loss_total.append(float(row["loss_total"]))
            loss_ce.append(float(row["loss_ce"]))
            loss_triplet.append(float(row["loss_triplet"]))
            if row["mAP"]:
                eval_epochs.append(epoch)
                mAP.append(float(row["mAP"]))
                rank1.append(float(row["Rank-1"]))
    return epochs, loss_total, loss_ce, loss_triplet, eval_epochs, mAP, rank1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/reid/train_log.csv")
    ap.add_argument("--output", default="outputs/reid/training_curve.png")
    args = ap.parse_args()

    epochs, loss_total, loss_ce, loss_triplet, eval_epochs, mAP, rank1 = load_log(Path(args.log))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: training losses
    ax1.plot(epochs, loss_total, label="Total", linewidth=2)
    ax1.plot(epochs, loss_ce, label="Cross-Entropy", linestyle="--", alpha=0.8)
    ax1.plot(epochs, loss_triplet, label="Triplet", linestyle="--", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right: validation metrics
    ax2.plot(eval_epochs, mAP, marker="o", label="mAP", linewidth=2)
    ax2.plot(eval_epochs, rank1, marker="s", label="Rank-1", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Market-1501 Validation")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()