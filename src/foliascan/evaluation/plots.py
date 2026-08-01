"""Confusion-matrix report outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib


def write_confusion_matrix_csv(
    *,
    matrix: tuple[tuple[int, ...], ...],
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write a raw-count confusion matrix CSV with deterministic class order."""

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_class"] + list(class_names))
        for class_name, row in zip(class_names, matrix, strict=True):
            writer.writerow([class_name, *row])


def write_confusion_matrix_png(
    *,
    matrix: tuple[tuple[int, ...], ...],
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write a readable raw-count confusion matrix plot."""

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    tick_indexes = range(len(class_names))
    axis.set_xticks(list(tick_indexes))
    axis.set_yticks(list(tick_indexes))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("FoliaScan Test Confusion Matrix")

    threshold = max((max(row) for row in matrix), default=0) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
