import csv
import json
from pathlib import Path

from foliascan.evaluation.metrics import (
    calculate_classification_metrics,
    confusion_pairs,
)
from foliascan.evaluation.reports import (
    PredictionRecord,
    write_confusion_pairs_csv,
    write_metrics_json,
    write_misclassified_csv,
    write_per_class_metrics_csv,
    write_predictions_csv,
)


def test_report_writers_create_expected_csv_and_json_outputs(tmp_path: Path) -> None:
    records = (
        PredictionRecord(
            relative_path="class_a/a.jpg",
            true_class="class_a",
            predicted_class="class_a",
            confidence=0.9,
            correct=True,
        ),
        PredictionRecord(
            relative_path="class_b/b.jpg",
            true_class="class_b",
            predicted_class="class_a",
            confidence=0.8,
            correct=False,
        ),
        PredictionRecord(
            relative_path="class_a/c.jpg",
            true_class="class_a",
            predicted_class="class_b",
            confidence=0.95,
            correct=False,
        ),
    )
    metrics = calculate_classification_metrics(
        targets=(0, 1, 0),
        predictions=(0, 0, 1),
        class_names=("class_a", "class_b"),
    )
    pairs = confusion_pairs(
        targets=(0, 1, 0),
        predictions=(0, 0, 1),
        class_names=("class_a", "class_b"),
    )

    write_predictions_csv(records, tmp_path / "predictions.csv")
    write_per_class_metrics_csv(metrics, tmp_path / "per_class_metrics.csv")
    write_misclassified_csv(records, tmp_path / "misclassified.csv")
    write_confusion_pairs_csv(pairs, tmp_path / "confusion_pairs.csv")
    write_metrics_json(
        metrics=metrics,
        checkpoint_path=tmp_path / "training" / "best_model.pt",
        checkpoint_epoch=7,
        class_to_index={"class_a": 0, "class_b": 1},
        device="cpu",
        output_path=tmp_path / "metrics.json",
        base_dir=tmp_path,
    )

    assert _csv_rows(tmp_path / "predictions.csv")[0] == {
        "relative_path": "class_a/a.jpg",
        "true_class": "class_a",
        "predicted_class": "class_a",
        "confidence": "0.9",
        "correct": "true",
    }
    assert _csv_rows(tmp_path / "per_class_metrics.csv")[0] == {
        "class_name": "class_a",
        "precision": "0.5",
        "recall": "0.5",
        "f1": "0.5",
        "support": "2",
    }
    misclassified_paths = [
        row["relative_path"] for row in _csv_rows(tmp_path / "misclassified.csv")
    ]
    assert misclassified_paths == [
        "class_a/c.jpg",
        "class_b/b.jpg",
    ]
    assert _csv_rows(tmp_path / "confusion_pairs.csv") == [
        {"true_class": "class_a", "predicted_class": "class_b", "count": "1"},
        {"true_class": "class_b", "predicted_class": "class_a", "count": "1"},
    ]

    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["checkpoint_path"] == "training/best_model.pt"
    assert report["checkpoint_epoch"] == 7
    assert report["accuracy"] == metrics.accuracy
    assert report["macro"] == {
        "precision": metrics.macro.precision,
        "recall": metrics.macro.recall,
        "f1": metrics.macro.f1,
    }
    assert report["class_to_index"] == {"class_a": 0, "class_b": 1}
    assert report["device"] == "cpu"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))
