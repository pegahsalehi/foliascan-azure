"""Evaluation report writers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from foliascan.evaluation.metrics import (
    ClassificationMetrics,
    ConfusionPair,
)

PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "relative_path",
    "true_class",
    "predicted_class",
    "confidence",
    "correct",
)
PER_CLASS_COLUMNS: tuple[str, ...] = (
    "class_name",
    "precision",
    "recall",
    "f1",
    "support",
)
CONFUSION_PAIR_COLUMNS: tuple[str, ...] = ("true_class", "predicted_class", "count")


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One evaluated prediction row."""

    relative_path: str
    true_class: str
    predicted_class: str
    confidence: float
    correct: bool


def write_predictions_csv(
    records: tuple[PredictionRecord, ...],
    output_path: Path,
) -> None:
    """Write all evaluated predictions to CSV."""

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTIONS_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(_prediction_row(record))


def write_per_class_metrics_csv(
    metrics: ClassificationMetrics,
    output_path: Path,
) -> None:
    """Write per-class precision, recall, F1, and support to CSV."""

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PER_CLASS_COLUMNS)
        writer.writeheader()
        for metric in metrics.per_class:
            writer.writerow(
                {
                    "class_name": metric.class_name,
                    "precision": _float_text(metric.precision),
                    "recall": _float_text(metric.recall),
                    "f1": _float_text(metric.f1),
                    "support": str(metric.support),
                }
            )


def write_misclassified_csv(
    records: tuple[PredictionRecord, ...],
    output_path: Path,
) -> None:
    """Write confidently sorted misclassified predictions to CSV."""

    misclassified = tuple(
        sorted(
            (record for record in records if not record.correct),
            key=lambda record: (-record.confidence, record.relative_path),
        )
    )
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTIONS_COLUMNS[:-1])
        writer.writeheader()
        for record in misclassified:
            writer.writerow(
                {
                    "relative_path": record.relative_path,
                    "true_class": record.true_class,
                    "predicted_class": record.predicted_class,
                    "confidence": _float_text(record.confidence),
                }
            )


def write_confusion_pairs_csv(
    pairs: tuple[ConfusionPair, ...],
    output_path: Path,
) -> None:
    """Write true-class to predicted-class mistake counts."""

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CONFUSION_PAIR_COLUMNS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "true_class": pair.true_class,
                    "predicted_class": pair.predicted_class,
                    "count": str(pair.count),
                }
            )


def write_metrics_json(
    *,
    metrics: ClassificationMetrics,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    class_to_index: dict[str, int],
    device: str,
    output_path: Path,
    base_dir: Path,
) -> None:
    """Write an aggregate metrics JSON report."""

    report = {
        "checkpoint_path": _portable_path(checkpoint_path, base_dir),
        "checkpoint_epoch": checkpoint_epoch,
        "total_samples": metrics.total_samples,
        "accuracy": metrics.accuracy,
        "macro": asdict(metrics.macro),
        "weighted": asdict(metrics.weighted),
        "per_class": [asdict(metric) for metric in metrics.per_class],
        "class_to_index": class_to_index,
        "device": device,
    }
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def _prediction_row(record: PredictionRecord) -> dict[str, str]:
    return {
        "relative_path": record.relative_path,
        "true_class": record.true_class,
        "predicted_class": record.predicted_class,
        "confidence": _float_text(record.confidence),
        "correct": str(record.correct).lower(),
    }


def _float_text(value: float) -> str:
    return f"{value:.12g}"


def _portable_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
