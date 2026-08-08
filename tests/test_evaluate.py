"""Confusion matrix counting, metrics and the gold/prediction column join."""

import pytest

from classifier.evaluate import (
    MATRIX_COLUMNS,
    UNDEFINED,
    Counts,
    build_matrix_rows,
    confusion_counts,
    matrix_row,
    sensitivity,
    specificity,
)


def test_counts_each_quadrant():
    counts = confusion_counts(
        [
            ("Abnormal", "abnormal"),  # TP
            ("Abnormal", "normal"),  # FN
            ("Normal", "normal"),  # TN
            ("Normal", "abnormal"),  # FP
        ]
    )
    assert (counts.true_positive, counts.false_negative) == (1, 1)
    assert (counts.true_negative, counts.false_positive) == (1, 1)
    assert counts.total == 4


def test_matching_is_case_insensitive():
    """Gold writes 'Abnormal', the models write 'abnormal'."""
    counts = confusion_counts([("ABNORMAL", "AbNoRmAl"), ("  normal  ", "Normal")])
    assert counts.true_positive == 1
    assert counts.true_negative == 1


def test_blank_predictions_are_skipped():
    """An unfinished run must score only the cases it completed."""
    counts = confusion_counts([("Abnormal", "abnormal"), ("Normal", ""), ("Abnormal", "")])
    assert counts.total == 1
    assert counts.true_positive == 1


def test_unrecognised_gold_is_skipped():
    counts = confusion_counts([("", "abnormal"), ("maybe", "normal"), ("Normal", "normal")])
    assert counts.total == 1
    assert counts.true_negative == 1


def test_sensitivity_and_specificity():
    counts = Counts(true_positive=3, false_negative=1, true_negative=4, false_positive=1)
    assert sensitivity(counts) == pytest.approx(0.75)
    assert specificity(counts) == pytest.approx(0.8)


def test_sensitivity_is_undefined_without_gold_positives():
    """0/0 is undefined, not zero - writing 0 would imply the model missed positives."""
    counts = Counts(true_positive=0, false_negative=0, true_negative=48, false_positive=2)
    assert sensitivity(counts) == UNDEFINED
    assert specificity(counts) == pytest.approx(0.96)


def test_specificity_is_undefined_without_gold_negatives():
    counts = Counts(true_positive=5, false_negative=1, true_negative=0, false_positive=0)
    assert specificity(counts) == UNDEFINED
    assert sensitivity(counts) == pytest.approx(5 / 6)


def test_matrix_row_uses_the_example_column_order():
    row = matrix_row("bronchitis", Counts(true_positive=17, false_negative=1, true_negative=25, false_positive=7))
    assert list(row) == list(MATRIX_COLUMNS)
    assert row["Check"] == 50
    assert row["Positive Ground Truth"] == 18
    assert row["Negative Ground Truth"] == 32
    assert row["Ground Truth Check"] == 50


def test_renamed_column_joins_gold_to_prediction():
    """Gold says Fe_Alveolar; the prediction files say Alveolar_interstitial_pattern."""
    gold = {"1": {"Fe_Alveolar": "Abnormal"}, "2": {"Fe_Alveolar": "Normal"}}
    predictions = {
        "1": {"Alveolar_interstitial_pattern": "abnormal"},
        "2": {"Alveolar_interstitial_pattern": "abnormal"},
    }
    rows = build_matrix_rows(
        gold, predictions, ["Fe_Alveolar"], {"Fe_Alveolar": "Alveolar_interstitial_pattern"}
    )
    assert rows[0]["condition"] == "Fe_Alveolar"
    assert rows[0]["True Positive"] == 1
    assert rows[0]["False Positive"] == 1


def test_cases_missing_from_predictions_are_ignored():
    gold = {"1": {"bronchitis": "Abnormal"}, "2": {"bronchitis": "Normal"}}
    predictions = {"1": {"bronchitis": "abnormal"}}
    rows = build_matrix_rows(gold, predictions, ["bronchitis"], {})
    assert rows[0]["Check"] == 1
    assert rows[0]["True Positive"] == 1


def test_one_row_per_condition():
    conditions = ["bronchitis", "cardiomegaly", "pneumonia"]
    gold = {"1": {c: "Normal" for c in conditions}}
    predictions = {"1": {c: "normal" for c in conditions}}
    rows = build_matrix_rows(gold, predictions, conditions, {})
    assert [r["condition"] for r in rows] == conditions
