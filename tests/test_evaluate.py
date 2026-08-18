"""Confusion matrix counting, metrics and the gold/prediction column join."""

from inspect import signature

import pytest

from classifier.evaluate import (
    MATRIX_COLUMNS,
    UNDEFINED,
    Counts,
    build_matrix_rows,
    confusion_counts,
    matrix_row,
    sensitivity,
    sensitivity_interval,
    specificity,
    specificity_interval,
    wilson_interval,
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


# --- Wilson score intervals ----------------------------------------------------
#
# The point of these on a 50-case study: eleven of the twenty feline conditions report a
# sensitivity of exactly 1.000, most of them resting on one to five positive cases. The
# interval is what tells the reader which of those is worth believing.


def test_default_alpha_is_five_percent():
    """MATRIX_COLUMNS spells "95%" as a literal. If the default moves, they start lying."""
    assert signature(wilson_interval).parameters["alpha"].default == 0.05


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 1), (1, 1), (1, 2), (5, 5), (17, 18), (25, 32), (0, 50), (50, 50), (49, 50)],
)
def test_interval_brackets_its_own_point_estimate(successes, trials):
    low, high = wilson_interval(successes, trials)
    assert low <= successes / trials <= high


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 1), (1, 1), (3, 7), (17, 18), (0, 50), (50, 50)],
)
def test_bounds_stay_inside_zero_and_one(successes, trials):
    low, high = wilson_interval(successes, trials)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_a_perfect_score_does_not_claim_certainty():
    """The left_sided_cardiomegaly case: one positive, called correctly, sensitivity
    1.000. Wald would report 1.000 +- 0. Wilson keeps the lower bound honest."""
    low, high = wilson_interval(1, 1)
    assert high == 1.0
    assert low < 1.0
    assert low == pytest.approx(0.2065493144, abs=1e-9)


def test_zero_successes_gives_a_lower_bound_of_exactly_zero():
    low, high = wilson_interval(0, 50)
    assert low == 0.0
    assert high > 0.0


def test_the_interval_narrows_as_evidence_accumulates():
    """Same perfect proportion, ten times the cases."""
    narrow_low, _ = wilson_interval(50, 50)
    wide_low, _ = wilson_interval(5, 5)
    assert narrow_low > wide_low


def test_no_trials_has_no_interval():
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (17, 18, (0.7424269922, 0.9901248090)),
        (25, 32, (0.6124500635, 0.8897616375)),
        (1, 1, (0.2065493144, 1.0)),
        (50, 50, (0.9286524009, 1.0)),
        (0, 50, (0.0, 0.0713475991)),
    ],
)
def test_pinned_values(successes, trials, expected):
    """Computed once and pinned, so a refactor of the formula cannot drift quietly."""
    assert wilson_interval(successes, trials) == pytest.approx(expected, abs=1e-9)


def test_intervals_read_their_denominators_off_the_counts():
    counts = Counts(true_positive=17, false_negative=1, true_negative=25, false_positive=7)
    assert sensitivity_interval(counts) == pytest.approx((0.7424269922, 0.9901248090), abs=1e-9)
    assert specificity_interval(counts) == pytest.approx((0.6124500635, 0.8897616375), abs=1e-9)


def test_a_condition_with_no_positives_has_no_sensitivity_interval():
    """Seven of the twenty feline conditions look like this."""
    counts = Counts(true_negative=50)
    assert sensitivity_interval(counts) is None
    assert specificity_interval(counts) is not None


def test_undefined_bounds_are_written_as_na_beside_an_undefined_estimate():
    """The bounds must never show numbers next to a sensitivity of N/A."""
    row = matrix_row("esophagitis", Counts(true_negative=50))
    assert row["Sensitivity"] == UNDEFINED
    assert row["Sensitivity 95% CI low"] == UNDEFINED
    assert row["Sensitivity 95% CI high"] == UNDEFINED
    assert row["Specificity"] == 1.0
    assert row["Specificity 95% CI low"] == pytest.approx(0.9286524009, abs=1e-9)


def test_matrix_row_emits_exactly_the_declared_columns_in_order():
    """Guards the constant and the row builder against drifting apart."""
    row = matrix_row("bronchitis", Counts(true_positive=17, false_negative=1, true_negative=25, false_positive=7))
    assert tuple(row) == MATRIX_COLUMNS
