"""Final test-set evaluation CLI."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from foliascan.evaluation.metrics import (
    EvaluationMetricsError,
    calculate_classification_metrics,
    confusion_pairs,
)
from foliascan.evaluation.plots import (
    write_confusion_matrix_csv,
    write_confusion_matrix_png,
)
from foliascan.evaluation.reports import (
    PredictionRecord,
    write_confusion_pairs_csv,
    write_metrics_json,
    write_misclassified_csv,
    write_per_class_metrics_csv,
    write_predictions_csv,
)
from foliascan.training.dataloaders import (
    DataLoaderError,
    create_dataloader,
    dataset_from_loader,
)
from foliascan.training.dataset import (
    ClassMapping,
    ManifestRecord,
    TrainingDataError,
    build_class_mapping,
    read_training_manifest,
    records_for_split,
)
from foliascan.training.model import ModelFactoryError, create_model
from foliascan.training.reproducibility import (
    DeviceResolutionError,
    resolve_device,
)

REQUIRED_CHECKPOINT_KEYS: tuple[str, ...] = (
    "epoch",
    "model_name",
    "model_state_dict",
    "class_to_index",
    "training_config",
)


class EvaluationError(ValueError):
    """Raised when final evaluation cannot complete."""


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Model and class metadata restored from a training checkpoint."""

    model: nn.Module
    epoch: int
    model_name: str
    class_mapping: ClassMapping
    training_config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EvaluationOutputs:
    """Raw predictions collected during test inference."""

    logits: Tensor
    probabilities: Tensor
    predictions: tuple[int, ...]
    targets: tuple[int, ...]
    relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Summary returned by final test evaluation."""

    checkpoint_epoch: int
    test_sample_count: int
    test_accuracy: float
    macro_f1: float
    weighted_f1: float
    output_dir: Path
    device: str
    num_classes: int
    misclassified_sample_count: int


def load_evaluation_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> LoadedCheckpoint:
    """Load and validate a training checkpoint for evaluation."""

    if not checkpoint_path.exists():
        msg = f"Checkpoint does not exist: {checkpoint_path}"
        raise EvaluationError(msg)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        msg = f"Checkpoint must contain a mapping: {checkpoint_path}"
        raise EvaluationError(msg)
    _validate_checkpoint_keys(checkpoint)

    epoch = _checkpoint_int(checkpoint, "epoch")
    model_name = _checkpoint_string(checkpoint, "model_name")
    class_mapping = _checkpoint_class_mapping(checkpoint)
    training_config = _checkpoint_training_config(checkpoint)
    model_state_dict = checkpoint["model_state_dict"]
    if not isinstance(model_state_dict, Mapping):
        msg = "Checkpoint field 'model_state_dict' must be a mapping."
        raise EvaluationError(msg)

    model = create_model(
        model_name=model_name,
        num_classes=class_mapping.num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        epoch=epoch,
        model_name=model_name,
        class_mapping=class_mapping,
        training_config=training_config,
    )


def run_evaluation(
    *,
    manifest_path: Path,
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
    overwrite: bool = False,
) -> EvaluationSummary:
    """Evaluate the selected checkpoint on the untouched test split."""

    device = _resolve_evaluation_device(device_name)
    checkpoint = load_evaluation_checkpoint(checkpoint_path, device=device)
    records = read_training_manifest(manifest_path)
    _validate_checkpoint_mapping_matches_manifest(checkpoint.class_mapping, records)
    test_loader = create_test_dataloader(
        records=records,
        data_dir=data_dir,
        class_mapping=checkpoint.class_mapping,
        checkpoint_config=checkpoint.training_config,
    )
    prepare_evaluation_output_dir(output_dir, overwrite=overwrite)
    if overwrite:
        reset_evaluation_artifacts(output_dir)

    outputs = collect_test_predictions(
        model=checkpoint.model,
        dataloader=test_loader,
        device=device,
    )
    class_names = checkpoint.class_mapping.index_to_class
    metrics = calculate_classification_metrics(
        targets=outputs.targets,
        predictions=outputs.predictions,
        class_names=class_names,
    )
    prediction_records = _prediction_records(outputs, class_names)
    pair_records = confusion_pairs(
        targets=outputs.targets,
        predictions=outputs.predictions,
        class_names=class_names,
    )

    write_predictions_csv(prediction_records, output_dir / "predictions.csv")
    write_per_class_metrics_csv(metrics, output_dir / "per_class_metrics.csv")
    write_metrics_json(
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint.epoch,
        class_to_index=dict(checkpoint.class_mapping.class_to_index),
        device=str(device),
        output_path=output_dir / "metrics.json",
        base_dir=Path.cwd(),
    )
    write_confusion_matrix_csv(
        matrix=metrics.confusion_matrix,
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.csv",
    )
    write_confusion_matrix_png(
        matrix=metrics.confusion_matrix,
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.png",
    )
    write_misclassified_csv(prediction_records, output_dir / "misclassified.csv")
    write_confusion_pairs_csv(pair_records, output_dir / "confusion_pairs.csv")

    return EvaluationSummary(
        checkpoint_epoch=checkpoint.epoch,
        test_sample_count=metrics.total_samples,
        test_accuracy=metrics.accuracy,
        macro_f1=metrics.macro.f1,
        weighted_f1=metrics.weighted.f1,
        output_dir=output_dir,
        device=str(device),
        num_classes=checkpoint.class_mapping.num_classes,
        misclassified_sample_count=sum(
            1 for record in prediction_records if not record.correct
        ),
    )


def create_test_dataloader(
    *,
    records: tuple[ManifestRecord, ...],
    data_dir: Path,
    class_mapping: ClassMapping,
    checkpoint_config: Mapping[str, object],
) -> DataLoader[tuple[Tensor, int]]:
    """Create the only DataLoader used during final evaluation."""

    test_records = records_for_split(records, "test")
    if not test_records:
        msg = "Training manifest contains no records for split: test"
        raise EvaluationError(msg)

    return create_dataloader(
        records=records,
        data_dir=data_dir,
        class_mapping=class_mapping,
        split="test",
        batch_size=_checkpoint_positive_int(checkpoint_config, "batch_size"),
        num_workers=_checkpoint_non_negative_int(checkpoint_config, "num_workers"),
        random_seed=_checkpoint_non_negative_int(checkpoint_config, "random_seed"),
        image_size=_checkpoint_positive_int(checkpoint_config, "image_size"),
    )


def collect_test_predictions(
    *,
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
) -> EvaluationOutputs:
    """Run test-set inference and collect logits, probabilities, and labels."""

    dataset = dataset_from_loader(dataloader)
    records = dataset.records
    logits: list[Tensor] = []
    probabilities: list[Tensor] = []
    predictions: list[int] = []
    targets: list[int] = []
    relative_paths: list[str] = []
    cursor = 0

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images, batch_targets = _batch_tensors(batch, device)
            batch_logits = model(images)
            batch_probabilities = torch.softmax(batch_logits, dim=1)
            batch_predictions = batch_probabilities.argmax(dim=1)
            batch_size = int(images.shape[0])
            batch_records = records[cursor : cursor + batch_size]
            cursor += batch_size

            logits.append(batch_logits.detach().cpu())
            probabilities.append(batch_probabilities.detach().cpu())
            predictions.extend(_tensor_ints(batch_predictions.cpu()))
            targets.extend(_tensor_ints(batch_targets.cpu()))
            relative_paths.extend(
                record.relative_path.as_posix() for record in batch_records
            )

    if cursor != len(records):
        msg = "Evaluation did not process every test record exactly once."
        raise EvaluationError(msg)
    if not targets:
        msg = "Cannot evaluate an empty test split."
        raise EvaluationError(msg)

    return EvaluationOutputs(
        logits=torch.cat(logits, dim=0),
        probabilities=torch.cat(probabilities, dim=0),
        predictions=tuple(predictions),
        targets=tuple(targets),
        relative_paths=tuple(relative_paths),
    )


def prepare_evaluation_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create or validate the evaluation output directory."""

    if output_dir.exists() and not output_dir.is_dir():
        msg = f"Evaluation output path is not a directory: {output_dir}"
        raise EvaluationError(msg)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        msg = (
            "Evaluation output directory is not empty; use --overwrite to write "
            f"into it: {output_dir}"
        )
        raise EvaluationError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)


def reset_evaluation_artifacts(output_dir: Path) -> None:
    """Remove report artifacts managed by evaluation when overwrite is enabled."""

    for filename in (
        "predictions.csv",
        "per_class_metrics.csv",
        "metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "misclassified.csv",
        "confusion_pairs.csv",
    ):
        artifact_path = output_dir / filename
        if artifact_path.exists():
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            else:
                artifact_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the final evaluation CLI and return a process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        manifest_path = _namespace_path(args, "manifest")
        data_dir = _namespace_path(args, "data_dir")
        checkpoint_path = _namespace_path(args, "checkpoint")
        output_dir = _namespace_path(args, "output_dir")
        device_name = _namespace_string(args, "device")
        print("FoliaScan final evaluation")
        print(f"manifest: {manifest_path}")
        print(f"data_dir: {data_dir}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"output_dir: {output_dir}")
        print(f"device: {device_name}")
        summary = run_evaluation(
            manifest_path=manifest_path,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device_name=device_name,
            overwrite=_namespace_bool(args, "overwrite"),
        )
    except (
        DataLoaderError,
        DeviceResolutionError,
        EvaluationError,
        EvaluationMetricsError,
        FileNotFoundError,
        ModelFactoryError,
        OSError,
        RuntimeError,
        TrainingDataError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_summary(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foliascan.evaluation.evaluate",
        description="Evaluate a selected FoliaScan checkpoint on the test split.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV FoliaScan dataset manifest path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Exported image dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Selected training checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Evaluation report output directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty evaluation output directory.",
    )
    return parser


def _print_summary(summary: EvaluationSummary) -> None:
    print("Evaluation complete")
    print(f"checkpoint_epoch: {summary.checkpoint_epoch}")
    print(f"test_samples: {summary.test_sample_count}")
    print(f"test_accuracy: {summary.test_accuracy:.6f}")
    print(f"macro_f1: {summary.macro_f1:.6f}")
    print(f"weighted_f1: {summary.weighted_f1:.6f}")
    print(f"misclassified: {summary.misclassified_sample_count}")
    print(f"classes: {summary.num_classes}")
    print(f"device: {summary.device}")
    print(f"output_dir: {summary.output_dir}")


def _validate_checkpoint_keys(checkpoint: Mapping[object, object]) -> None:
    for key in REQUIRED_CHECKPOINT_KEYS:
        if key not in checkpoint:
            msg = f"Checkpoint is missing required field: {key}"
            raise EvaluationError(msg)


def _checkpoint_int(checkpoint: Mapping[object, object], key: str) -> int:
    value = checkpoint[key]
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Checkpoint field '{key}' must be an integer."
        raise EvaluationError(msg)
    return value


def _checkpoint_string(checkpoint: Mapping[object, object], key: str) -> str:
    value = checkpoint[key]
    if not isinstance(value, str) or not value:
        msg = f"Checkpoint field '{key}' must be a non-empty string."
        raise EvaluationError(msg)
    return value


def _checkpoint_class_mapping(checkpoint: Mapping[object, object]) -> ClassMapping:
    raw_mapping = checkpoint["class_to_index"]
    if not isinstance(raw_mapping, Mapping):
        msg = "Checkpoint field 'class_to_index' must be a mapping."
        raise EvaluationError(msg)

    class_to_index: dict[str, int] = {}
    for class_name, class_index in raw_mapping.items():
        if not isinstance(class_name, str) or not class_name:
            msg = "Checkpoint class_to_index keys must be non-empty strings."
            raise EvaluationError(msg)
        if not isinstance(class_index, int) or isinstance(class_index, bool):
            msg = "Checkpoint class_to_index values must be integers."
            raise EvaluationError(msg)
        class_to_index[class_name] = class_index

    index_to_class = _index_to_class(class_to_index)
    return ClassMapping(
        class_to_index=MappingProxyType(class_to_index),
        index_to_class=index_to_class,
    )


def _index_to_class(class_to_index: Mapping[str, int]) -> tuple[str, ...]:
    if not class_to_index:
        msg = "Checkpoint class_to_index must not be empty."
        raise EvaluationError(msg)

    sorted_pairs = sorted(class_to_index.items(), key=lambda item: item[1])
    expected_indexes = tuple(range(len(sorted_pairs)))
    actual_indexes = tuple(class_index for _, class_index in sorted_pairs)
    if actual_indexes != expected_indexes:
        msg = "Checkpoint class_to_index must be zero-based and contiguous."
        raise EvaluationError(msg)
    return tuple(class_name for class_name, _ in sorted_pairs)


def _checkpoint_training_config(
    checkpoint: Mapping[object, object],
) -> Mapping[str, object]:
    raw_config = checkpoint["training_config"]
    if not isinstance(raw_config, Mapping):
        msg = "Checkpoint field 'training_config' must be a mapping."
        raise EvaluationError(msg)
    return MappingProxyType({str(key): value for key, value in raw_config.items()})


def _validate_checkpoint_mapping_matches_manifest(
    checkpoint_mapping: ClassMapping,
    records: tuple[ManifestRecord, ...],
) -> None:
    manifest_mapping = build_class_mapping(records)
    if dict(checkpoint_mapping.class_to_index) != dict(manifest_mapping.class_to_index):
        msg = "Checkpoint class mapping does not match manifest classes."
        raise EvaluationError(msg)


def _checkpoint_positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"Checkpoint training_config field '{key}' must be a positive integer."
        raise EvaluationError(msg)
    return value


def _checkpoint_non_negative_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = (
            f"Checkpoint training_config field '{key}' must be a "
            "non-negative integer."
        )
        raise EvaluationError(msg)
    return value


def _resolve_evaluation_device(device_name: str) -> torch.device:
    if device_name in {"auto", "cpu", "cuda"}:
        return resolve_device(cast(Any, device_name))
    msg = f"Unsupported device: {device_name}"
    raise EvaluationError(msg)


def _batch_tensors(batch: object, device: torch.device) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, list | tuple) or len(batch) != 2:
        msg = "Evaluation batches must contain images and targets."
        raise EvaluationError(msg)
    images, targets = batch
    if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
        msg = "Evaluation batches must contain tensor images and tensor targets."
        raise EvaluationError(msg)
    return images.to(device), targets.to(device)


def _tensor_ints(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.tolist()]


def _prediction_records(
    outputs: EvaluationOutputs,
    class_names: tuple[str, ...],
) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    confidences = outputs.probabilities.max(dim=1).values.tolist()
    for relative_path, target, prediction, confidence in zip(
        outputs.relative_paths,
        outputs.targets,
        outputs.predictions,
        confidences,
        strict=True,
    ):
        records.append(
            PredictionRecord(
                relative_path=relative_path,
                true_class=class_names[target],
                predicted_class=class_names[prediction],
                confidence=float(confidence),
                correct=target == prediction,
            )
        )
    return tuple(records)


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if isinstance(value, Path):
        return value
    msg = f"Expected path argument for {name}."
    raise TypeError(msg)


def _namespace_string(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if isinstance(value, str):
        return value
    msg = f"Expected string argument for {name}."
    raise TypeError(msg)


def _namespace_bool(args: argparse.Namespace, name: str) -> bool:
    value = getattr(args, name)
    if isinstance(value, bool):
        return value
    msg = f"Expected boolean argument for {name}."
    raise TypeError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
