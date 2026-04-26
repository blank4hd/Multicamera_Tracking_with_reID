"""Utility to select the best available torch device."""

import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA, then MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


if __name__ == "__main__":
    print(get_device())
