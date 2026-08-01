import csv
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader

from foliascan.evaluation import evaluate as evaluate_module
from foliascan.evaluation.evaluate import (
    EvaluationError,
    EvaluationSummary,
    collect_test_predictions,
    create_test_dataloader,
    load_evaluation_checkpoint,
    prepare_evaluation_output_dir,
    run_evaluation,
)
from foliascan.training.dataset import (
    MANIFEST_COLUMNS,
    ClassMapping,
    ManifestImageDataset,
    ManifestRecord,
)
from foliascan.training.transforms import create_eval_transform


class TinyImageClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 16 * 16, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(inputs.flatten(start_dim=1))


def test_checkpoint_metadata_validation_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"epoch": 1}, checkpoint_path)

    with pytest.raises(EvaluationError, match="model_name"):
        load_evaluation_checkpoint(checkpoint_path, device=torch.device("cpu"))


def test_load_evaluation_checkpoint_restores_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_model = TinyImageClassifier(num_classes=2)
    with torch.no_grad():
        saved_model.classifier.weight.fill_(0.25)
        saved_model.classifier.bias.copy_(torch.tensor([1.0, -1.0]))
    checkpoint_path = _write_checkpoint(tmp_path, model=saved_model, epoch=7)

    created_model = TinyImageClassifier(num_classes=2)
    monkeypatch.setattr(
        evaluate_module,
        "create_model",
        lambda **kwargs: created_model,
    )

    checkpoint = load_evaluation_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert checkpoint.epoch == 7
    assert checkpoint.model_name == "resnet18"
    assert checkpoint.class_mapping.index_to_class == ("class_a", "class_b")
    assert torch.equal(
        created_model.classifier.weight,
        saved_model.classifier.weight,
    )
    assert torch.equal(created_model.classifier.bias, saved_model.classifier.bias)
    assert checkpoint.model is created_model
    assert not checkpoint.model.training


def test_run_evaluation_rejects_manifest_class_mapping_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_model = TinyImageClassifier(num_classes=2)
    checkpoint_path = _write_checkpoint(
        tmp_path,
        model=saved_model,
        class_to_index={"class_a": 0, "class_c": 1},
    )
    data_dir, manifest_path = _write_tiny_manifest_and_images(tmp_path)
    monkeypatch.setattr(
        evaluate_module,
        "create_model",
        lambda **kwargs: TinyImageClassifier(num_classes=2),
    )

    with pytest.raises(EvaluationError, match="does not match"):
        run_evaluation(
            manifest_path=manifest_path,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_dir=tmp_path / "evaluation",
            device_name="cpu",
        )


def test_create_test_dataloader_uses_only_test_records(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_tiny_manifest_and_images(tmp_path)
    records = evaluate_module.read_training_manifest(manifest_path)
    class_mapping = ClassMapping(
        class_to_index={"class_a": 0, "class_b": 1},
        index_to_class=("class_a", "class_b"),
    )

    dataloader = create_test_dataloader(
        records=records,
        data_dir=data_dir,
        class_mapping=class_mapping,
        checkpoint_config=_checkpoint_config(),
    )
    dataset = evaluate_module.dataset_from_loader(dataloader)

    assert [record.split for record in dataset.records] == ["test", "test"]
    assert [record.relative_path.as_posix() for record in dataset.records] == [
        "class_a/test_a.jpg",
        "class_b/test_b.jpg",
    ]


def test_collect_test_predictions_keeps_model_parameters_unchanged(
    tmp_path: Path,
) -> None:
    dataset = ManifestImageDataset(
        [
            ManifestRecord(
                Path("class_a/test_a.jpg"),
                "class_a",
                "test",
                "leaf",
                "test",
            )
        ],
        tmp_path,
        ClassMapping({"class_a": 0, "class_b": 1}, ("class_a", "class_b")),
        create_eval_transform(16),
    )
    image_path = tmp_path / "class_a" / "test_a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (18, 18), color=(10, 10, 10)).save(image_path)
    dataloader: DataLoader[tuple[Tensor, int]] = DataLoader(dataset, batch_size=1)
    model = TinyImageClassifier(num_classes=2)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    outputs = collect_test_predictions(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    after = model.state_dict()
    for name, value in before.items():
        assert torch.equal(value, after[name])
    assert outputs.logits.shape == (1, 2)
    assert outputs.probabilities.shape == (1, 2)
    assert torch.allclose(outputs.probabilities.sum(dim=1), torch.ones(1))
    assert len(outputs.predictions) == 1
    assert outputs.targets == (0,)
    assert outputs.relative_paths == ("class_a/test_a.jpg",)


def test_prepare_output_dir_rejects_non_empty_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(EvaluationError, match="not empty"):
        prepare_evaluation_output_dir(output_dir, overwrite=False)

    prepare_evaluation_output_dir(output_dir, overwrite=True)


def test_run_evaluation_rejects_unavailable_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA was requested"):
        run_evaluation(
            manifest_path=tmp_path / "manifest.csv",
            data_dir=tmp_path / "data",
            checkpoint_path=tmp_path / "checkpoint.pt",
            output_dir=tmp_path / "evaluation",
            device_name="cuda",
        )


def test_run_evaluation_completes_tiny_end_to_end_without_using_train_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path = _write_tiny_manifest_and_images(tmp_path)
    saved_model = TinyImageClassifier(num_classes=2)
    checkpoint_path = _write_checkpoint(tmp_path, model=saved_model, epoch=3)
    create_loader_calls: list[str] = []
    original_create_dataloader = evaluate_module.create_dataloader

    def recording_create_dataloader(**kwargs: object) -> DataLoader[tuple[Tensor, int]]:
        split = kwargs["split"]
        assert isinstance(split, str)
        create_loader_calls.append(split)
        return original_create_dataloader(**kwargs)

    monkeypatch.setattr(
        evaluate_module,
        "create_model",
        lambda **kwargs: TinyImageClassifier(num_classes=2),
    )
    monkeypatch.setattr(
        evaluate_module,
        "create_dataloader",
        recording_create_dataloader,
    )

    summary = run_evaluation(
        manifest_path=manifest_path,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "evaluation",
        device_name="cpu",
    )

    assert summary.checkpoint_epoch == 3
    assert summary.test_sample_count == 2
    assert summary.output_dir == tmp_path / "evaluation"
    assert summary.device == "cpu"
    assert summary.num_classes == 2
    assert create_loader_calls == ["test"]
    for filename in (
        "predictions.csv",
        "per_class_metrics.csv",
        "metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "misclassified.csv",
        "confusion_pairs.csv",
    ):
        assert (summary.output_dir / filename).exists()
    assert not (data_dir / "class_a" / "missing_train.jpg").exists()
    assert not (data_dir / "class_b" / "missing_validation.jpg").exists()


def test_evaluate_cli_parses_arguments_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_evaluation(
        *,
        manifest_path: Path,
        data_dir: Path,
        checkpoint_path: Path,
        output_dir: Path,
        device_name: str,
        overwrite: bool = False,
    ) -> EvaluationSummary:
        assert manifest_path == tmp_path / "dataset_manifest.csv"
        assert data_dir == tmp_path / "raw"
        assert checkpoint_path == tmp_path / "best_model.pt"
        assert output_dir == tmp_path / "evaluation"
        assert device_name == "cpu"
        assert overwrite is True
        return EvaluationSummary(
            checkpoint_epoch=7,
            test_sample_count=4,
            test_accuracy=0.75,
            macro_f1=0.7,
            weighted_f1=0.72,
            output_dir=output_dir,
            device="cpu",
            num_classes=2,
            misclassified_sample_count=1,
        )

    monkeypatch.setattr(evaluate_module, "run_evaluation", fake_run_evaluation)

    exit_status = evaluate_module.main(
        [
            "--manifest",
            str(tmp_path / "dataset_manifest.csv"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--checkpoint",
            str(tmp_path / "best_model.pt"),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--device",
            "cpu",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "FoliaScan final evaluation" in captured.out
    assert "Evaluation complete" in captured.out
    assert "test_accuracy: 0.750000" in captured.out


def test_evaluate_cli_reports_expected_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_evaluation(**kwargs: object) -> EvaluationSummary:
        raise EvaluationError("bad evaluation input")

    monkeypatch.setattr(evaluate_module, "run_evaluation", fake_run_evaluation)

    exit_status = evaluate_module.main(
        [
            "--manifest",
            str(tmp_path / "dataset_manifest.csv"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--checkpoint",
            str(tmp_path / "best_model.pt"),
            "--output-dir",
            str(tmp_path / "evaluation"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 2
    assert "error: bad evaluation input" in captured.err
    assert "Traceback" not in captured.err


def _write_tiny_manifest_and_images(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "raw"
    manifest_path = tmp_path / "dataset_manifest.csv"
    rows = [
        ("class_a/missing_train.jpg", "class_a", "train", "leaf_train", "train"),
        (
            "class_b/missing_validation.jpg",
            "class_b",
            "validation",
            "leaf_validation",
            "train",
        ),
        ("class_a/test_a.jpg", "class_a", "test", "leaf_test_a", "test"),
        ("class_b/test_b.jpg", "class_b", "test", "leaf_test_b", "test"),
    ]
    for relative_path, _, split, _, _ in rows:
        if split != "test":
            continue
        image_path = data_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (18, 18), color=(20, 40, 60)).save(image_path)

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for relative_path, class_name, split, leaf_id, source_split in rows:
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "split": split,
                    "leaf_id": leaf_id,
                    "source_split": source_split,
                }
            )

    return data_dir, manifest_path


def _write_checkpoint(
    tmp_path: Path,
    *,
    model: nn.Module,
    epoch: int = 2,
    class_to_index: Mapping[str, int] | None = None,
) -> Path:
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_name": "resnet18",
            "model_state_dict": model.state_dict(),
            "class_to_index": dict(class_to_index or {"class_a": 0, "class_b": 1}),
            "training_config": _checkpoint_config(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _checkpoint_config() -> dict[str, object]:
    return {
        "random_seed": 42,
        "image_size": 16,
        "batch_size": 2,
        "num_workers": 0,
    }
