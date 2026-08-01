import math

import pytest
import torch
from torch import nn

from foliascan.training.engine import (
    TrainingEngineError,
    evaluate_one_epoch,
    train_one_epoch,
)


def test_train_one_epoch_updates_model_parameters() -> None:
    model = nn.Linear(2, 2)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    dataloader = [
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )
    ]

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    after = tuple(parameter.detach().clone() for parameter in model.parameters())
    assert model.training is True
    assert metrics.sample_count == 2
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, after, strict=True)
    )


def test_evaluate_one_epoch_does_not_update_model_parameters() -> None:
    model = nn.Linear(2, 2)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    dataloader = [
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )
    ]

    metrics = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    after = tuple(parameter.detach().clone() for parameter in model.parameters())
    assert model.training is False
    assert metrics.sample_count == 2
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_evaluate_one_epoch_calculates_sample_weighted_metrics() -> None:
    model = _ZeroLogitModel()
    loss_fn = nn.CrossEntropyLoss()
    dataloader = [
        (torch.ones((1, 2)), torch.tensor([0])),
        (torch.ones((2, 2)), torch.tensor([1, 0])),
    ]

    metrics = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    assert metrics.sample_count == 3
    assert metrics.batch_count == 2
    assert metrics.average_loss == pytest.approx(math.log(2))
    assert metrics.accuracy == pytest.approx(2 / 3)


def test_epoch_metrics_reject_empty_dataloader() -> None:
    with pytest.raises(TrainingEngineError, match="empty"):
        evaluate_one_epoch(
            model=_ZeroLogitModel(),
            dataloader=[],
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )


class _ZeroLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0], 2), device=inputs.device)
