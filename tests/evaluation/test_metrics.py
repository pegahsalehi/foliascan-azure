import pytest

from foliascan.evaluation.metrics import (
    EvaluationMetricsError,
    calculate_classification_metrics,
    confusion_pairs,
)


def test_calculate_classification_metrics_uses_explicit_class_order() -> None:
    metrics = calculate_classification_metrics(
        targets=(0, 0, 1, 1, 2),
        predictions=(0, 1, 1, 0, 1),
        class_names=("class_a", "class_b", "class_c"),
    )

    assert metrics.total_samples == 5
    assert metrics.accuracy == pytest.approx(0.4)
    assert metrics.confusion_matrix == (
        (1, 1, 0),
        (1, 1, 0),
        (0, 1, 0),
    )

    assert metrics.per_class[0].class_name == "class_a"
    assert metrics.per_class[0].precision == pytest.approx(0.5)
    assert metrics.per_class[0].recall == pytest.approx(0.5)
    assert metrics.per_class[0].f1 == pytest.approx(0.5)
    assert metrics.per_class[0].support == 2

    assert metrics.per_class[1].class_name == "class_b"
    assert metrics.per_class[1].precision == pytest.approx(1 / 3)
    assert metrics.per_class[1].recall == pytest.approx(0.5)
    assert metrics.per_class[1].f1 == pytest.approx(0.4)
    assert metrics.per_class[1].support == 2

    assert metrics.per_class[2].class_name == "class_c"
    assert metrics.per_class[2].precision == 0.0
    assert metrics.per_class[2].recall == 0.0
    assert metrics.per_class[2].f1 == 0.0
    assert metrics.per_class[2].support == 1

    assert metrics.macro.precision == pytest.approx((0.5 + (1 / 3)) / 3)
    assert metrics.macro.recall == pytest.approx((0.5 + 0.5) / 3)
    assert metrics.macro.f1 == pytest.approx((0.5 + 0.4) / 3)
    assert metrics.weighted.precision == pytest.approx(((0.5 * 2) + ((1 / 3) * 2)) / 5)
    assert metrics.weighted.recall == pytest.approx(((0.5 * 2) + (0.5 * 2)) / 5)
    assert metrics.weighted.f1 == pytest.approx(((0.5 * 2) + (0.4 * 2)) / 5)


def test_confusion_pairs_are_sorted_by_count_then_class_names() -> None:
    pairs = confusion_pairs(
        targets=(0, 0, 1, 2),
        predictions=(1, 1, 0, 1),
        class_names=("class_a", "class_b", "class_c"),
    )

    assert [(pair.true_class, pair.predicted_class, pair.count) for pair in pairs] == [
        ("class_a", "class_b", 2),
        ("class_b", "class_a", 1),
        ("class_c", "class_b", 1),
    ]


def test_metrics_reject_mismatched_prediction_lengths() -> None:
    with pytest.raises(EvaluationMetricsError, match="same length"):
        calculate_classification_metrics(
            targets=(0,),
            predictions=(0, 1),
            class_names=("class_a", "class_b"),
        )


def test_metrics_reject_out_of_range_class_indexes() -> None:
    with pytest.raises(EvaluationMetricsError, match="out of range"):
        calculate_classification_metrics(
            targets=(0, 2),
            predictions=(0, 1),
            class_names=("class_a", "class_b"),
        )
