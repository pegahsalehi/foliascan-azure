"""Reproducibility and device helpers for local training."""

from __future__ import annotations

import random

import numpy as np
import torch

from foliascan.training.config import DeviceName


class DeviceResolutionError(ValueError):
    """Raised when a requested torch device is not usable."""


def seed_everything(random_seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    if random_seed < 0:
        msg = "random_seed must not be negative."
        raise ValueError(msg)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_name: DeviceName) -> torch.device:
    """Resolve a configured device name to a concrete torch device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA was requested but is not available."
            raise DeviceResolutionError(msg)
        return torch.device("cuda")

    msg = f"Unsupported device: {device_name}"
    raise DeviceResolutionError(msg)

