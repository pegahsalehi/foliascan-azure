"""Classification metrics for final FoliaScan evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


class EvaluationMetricsError(ValueError):
    """Raised when evaluation metrics cannot be calculated."""


@dataclass(frozen=True, slots=True)
class PerClassMetric:
    """Precision, recall, F1, and support for one class."""

    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Macro and weighted metric aggregates."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Complete classification metrics for an evaluation run."""

    total_samples: int
    accuracy: float
    per_class: tuple[PerClassMetric, ...]
    macro: AggregateMetrics
    weighted: AggregateMetrics
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class ConfusionPair:
    """One true-class to predicted-class mistake count."""

    true_class: str
    predicted_class: str
    count: int


def calculate_classification_metrics(
    *,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> ClassificationMetrics:
    """Calculate accuracy, per-class metrics, aggregates, and confusion matrix."""

    if len(targets) != len(predictions):
        msg = "Targets and predictions must have the same length."
        raise EvaluationMetricsError(msg)
    if not class_names:
        msg = "Cannot calculate metrics without class names."
        raise EvaluationMetricsError(msg)
    if not targets:
        msg = "Cannot calculate metrics for an empty evaluation set."
        raise EvaluationMetricsError(msg)

    class_count = len(class_names)
    _validate_label_indexes(targets, class_count, "target")
    _validate_label_indexes(predictions, class_count, "prediction")

    confusion_matrix = _confusion_matrix(targets, predictions, class_count)
    per_class = tuple(
        _per_class_metric(class_names[index], confusion_matrix, index)
        for index in range(class_count)
    )
    correct_count = sum(confusion_matrix[index][index] for index in range(class_count))
    total_samples = len(targets)

    return ClassificationMetrics(
        total_samples=total_samples,
        accuracy=correct_count / total_samples,
        per_class=per_class,
        macro=_macro_metrics(per_class),
        weighted=_weighted_metrics(per_class, total_samples),
        confusion_matrix=tuple(tuple(row) for row in confusion_matrix),
    )


def confusion_pairs(
    *,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> tuple[ConfusionPair, ...]:
    """Return mistake pairs sorted by frequency, then class names."""

    if len(targets) != len(predictions):
        msg = "Targets and predictions must have the same length."
        raise EvaluationMetricsError(msg)

    counts: Counter[tuple[int, int]] = Counter()
    for target, prediction in zip(targets, predictions, strict=True):
        if target != prediction:
            counts[(target, prediction)] += 1

    return tuple(
        ConfusionPair(
            true_class=class_names[target],
            predicted_class=class_names[prediction],
            count=count,
        )
        for (target, prediction), count in sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                class_names[item[0][0]].casefold(),
                class_names[item[0][1]].casefold(),
            ),
        )
    )


def _validate_label_indexes(
    values: Sequence[int],
    class_count: int,
    label_kind: str,
) -> None:
    for value in values:
        if value < 0 or value >= class_count:
            msg = f"Evaluation {label_kind} index is out of range: {value}"
            raise EvaluationMetricsError(msg)


def _confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_count: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target][prediction] += 1
    return matrix


def _per_class_metric(
    class_name: str,
    confusion_matrix: list[list[int]],
    class_index: int,
) -> PerClassMetric:
    true_positive = confusion_matrix[class_index][class_index]
    false_positive = sum(row[class_index] for row in confusion_matrix) - true_positive
    false_negative = sum(confusion_matrix[class_index]) - true_positive
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return PerClassMetric(
        class_name=class_name,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        support=sum(confusion_matrix[class_index]),
    )


def _macro_metrics(per_class: Sequence[PerClassMetric]) -> AggregateMetrics:
    class_count = len(per_class)
    return AggregateMetrics(
        precision=sum(metric.precision for metric in per_class) / class_count,
        recall=sum(metric.recall for metric in per_class) / class_count,
        f1=sum(metric.f1 for metric in per_class) / class_count,
    )


def _weighted_metrics(
    per_class: Sequence[PerClassMetric],
    total_samples: int,
) -> AggregateMetrics:
    return AggregateMetrics(
        precision=sum(metric.precision * metric.support for metric in per_class)
        / total_samples,
        recall=sum(metric.recall * metric.support for metric in per_class)
        / total_samples,
        f1=sum(metric.f1 * metric.support for metric in per_class) / total_samples,
    )


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2 * precision * recall, precision + recall)


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)

