import csv
from pathlib import Path

from foliascan.evaluation.plots import (
    write_confusion_matrix_csv,
    write_confusion_matrix_png,
)


def test_write_confusion_matrix_csv_preserves_class_order(tmp_path: Path) -> None:
    output_path = tmp_path / "confusion_matrix.csv"

    write_confusion_matrix_csv(
        matrix=((2, 1), (0, 3)),
        class_names=("class_a", "class_b"),
        output_path=output_path,
    )

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows == [
        ["true_class", "class_a", "class_b"],
        ["class_a", "2", "1"],
        ["class_b", "0", "3"],
    ]


def test_write_confusion_matrix_png_creates_non_empty_image(tmp_path: Path) -> None:
    output_path = tmp_path / "confusion_matrix.png"

    write_confusion_matrix_png(
        matrix=((2, 1), (0, 3)),
        class_names=("class_a", "class_b"),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
