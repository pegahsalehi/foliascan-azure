from torch import nn

from foliascan.training.model import ModelFactoryError, create_model


def test_create_model_replaces_resnet18_classifier() -> None:
    model = create_model(
        model_name="resnet18",
        num_classes=10,
        pretrained=False,
        freeze_backbone=False,
    )

    assert isinstance(model.fc, nn.Linear)  # type: ignore[attr-defined]
    assert model.fc.out_features == 10  # type: ignore[attr-defined]


def test_create_model_freezes_backbone_but_keeps_classifier_trainable() -> None:
    model = create_model(
        model_name="resnet18",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )

    frozen_parameters = [
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    ]
    classifier_parameters = [
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("fc.")
    ]

    assert frozen_parameters
    assert not any(frozen_parameters)
    assert classifier_parameters
    assert all(classifier_parameters)


def test_create_model_rejects_unsupported_model_name() -> None:
    try:
        create_model(
            model_name="mobilenet",
            num_classes=3,
            pretrained=False,
            freeze_backbone=False,
        )
    except ModelFactoryError as exc:
        assert "Unsupported model_name" in str(exc)
    else:
        raise AssertionError("Expected unsupported model name to fail.")

