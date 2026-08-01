import random

import numpy as np
import pytest
import torch

from foliascan.training.reproducibility import (
    DeviceResolutionError,
    resolve_device,
    seed_everything,
)


def test_seed_everything_is_deterministic_on_cpu() -> None:
    seed_everything(123)
    first_python = random.random()
    first_numpy = float(np.random.random())
    first_torch = torch.rand(3)

    seed_everything(123)
    second_python = random.random()
    second_numpy = float(np.random.random())
    second_torch = torch.rand(3)

    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch.equal(first_torch, second_torch)


def test_resolve_device_uses_cpu_when_requested() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_uses_cpu_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"


def test_resolve_device_rejects_unavailable_explicit_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceResolutionError, match="CUDA"):
        resolve_device("cuda")

