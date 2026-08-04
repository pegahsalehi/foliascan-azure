"""Training and validation epoch loops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class TrainingEngineError(ValueError):
    """Raised when an epoch cannot produce valid metrics."""


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Sample-weighted epoch metrics."""

    average_loss: float
    accuracy: float
    sample_count: int
    batch_count: int


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: Iterable[object],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Run one training epoch and return aggregate metrics."""

    _validate_max_batches(max_batches)
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    batch_count = 0

    for batch in _limited_batches(dataloader, max_batches):
        images, targets = _batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        correct_count += _correct_count(outputs, targets)
        sample_count += batch_size
        batch_count += 1

    return _metrics(total_loss, correct_count, sample_count, batch_count)


def evaluate_one_epoch(
    *,
    model: nn.Module,
    dataloader: Iterable[object],
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Run one validation epoch without updating model parameters."""

    _validate_max_batches(max_batches)
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    batch_count = 0

    with torch.inference_mode():
        for batch in _limited_batches(dataloader, max_batches):
            images, targets = _batch_tensors(batch, device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct_count += _correct_count(outputs, targets)
            sample_count += batch_size
            batch_count += 1

    return _metrics(total_loss, correct_count, sample_count, batch_count)


def _batch_tensors(batch: object, device: torch.device) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, list | tuple) or len(batch) != 2:
        msg = "Training batches must contain images and targets."
        raise TrainingEngineError(msg)

    images, targets = batch
    if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
        msg = "Training batches must contain tensor images and tensor targets."
        raise TrainingEngineError(msg)

    return images.to(device), targets.to(device)


def _limited_batches(
    dataloader: Iterable[object],
    max_batches: int | None,
) -> Iterable[object]:
    if max_batches is None:
        return dataloader
    return islice(dataloader, max_batches)


def _validate_max_batches(max_batches: int | None) -> None:
    if max_batches is None:
        return
    if max_batches <= 0:
        msg = "max_batches must be a positive integer when supplied."
        raise TrainingEngineError(msg)


def _correct_count(outputs: Tensor, targets: Tensor) -> int:
    predictions = outputs.argmax(dim=1)
    return int((predictions == targets).sum().item())


def _metrics(
    total_loss: float,
    correct_count: int,
    sample_count: int,
    batch_count: int,
) -> EpochMetrics:
    if sample_count <= 0 or batch_count <= 0:
        msg = "Cannot calculate epoch metrics from an empty dataloader."
        raise TrainingEngineError(msg)
    return EpochMetrics(
        average_loss=total_loss / sample_count,
        accuracy=correct_count / sample_count,
        sample_count=sample_count,
        batch_count=batch_count,
    )

