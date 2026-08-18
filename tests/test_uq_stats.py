"""Discrimination statistics. Pure functions over arrays - no files, no API."""

import random

import pytest

from uncertainty.stats import (
    bin_indices,
    brier_score,
    calibration_points,
    clustered_bootstrap_ci,
    expected_calibration_error,
    roc_auc,
)


def test_a_perfect_separator_scores_one():
    assert roc_auc([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) == 1.0


def test_a_perfectly_wrong_separator_scores_zero():
    assert roc_auc([0.1, 0.1, 0.9, 0.9], [1, 1, 0, 0]) == 0.0


def test_constant_confidence_scores_one_half():
    assert roc_auc([0.7] * 4, [1, 0, 1, 0]) == 0.5


def test_no_incorrect_answers_returns_none_rather_than_raising():
    """sklearn raises here; a tier the model aced must not crash the whole report."""
    assert roc_auc([0.9, 0.8, 0.7], [1, 1, 1]) is None


def test_no_correct_answers_returns_none():
    assert roc_auc([0.9, 0.8, 0.7], [0, 0, 0]) is None


def _clustered_data(n_cases=20, per_case=19, seed=0):
    """Both correctness and confidence vary at the case level.

    This is the structure of the real data: the 19 findings of one case are read off one
    shared report, so an easy case tends to be right *and* confident across all 19 rows.
    Modelling only correctness at the case level would leave the confidences independent
    within a case, and clustering would then buy no extra variance at all.
    """
    rng = random.Random(seed)
    confidence, correct, case_ids = [], [], []
    for index in range(n_cases):
        case_correct = 1 if index % 2 == 0 else 0
        centre = (0.6 if case_correct else 0.4) + rng.gauss(0, 0.20)
        for _ in range(per_case):
            confidence.append(min(1.0, max(0.0, rng.gauss(centre, 0.05))))
            correct.append(case_correct)
            case_ids.append(f"case{index}")
    return confidence, correct, case_ids


def test_clustered_bootstrap_is_wider_than_row_level_resampling():
    """The whole reason for clustering: 380 rows are only 20 independent observations.

    Passing a unique id per row makes the same function resample rows, so this compares
    the two strategies on identical data. Row-level resampling reports an interval about
    four times too narrow.
    """
    confidence, correct, case_ids = _clustered_data()
    row_ids = [str(i) for i in range(len(confidence))]

    low_c, high_c = clustered_bootstrap_ci(confidence, correct, case_ids, iterations=400, seed=1)
    low_r, high_r = clustered_bootstrap_ci(confidence, correct, row_ids, iterations=400, seed=1)

    assert (high_c - low_c) > (high_r - low_r)


def test_bootstrap_interval_brackets_the_point_estimate():
    confidence, correct, case_ids = _clustered_data()
    point = roc_auc(confidence, correct)

    low, high = clustered_bootstrap_ci(confidence, correct, case_ids, iterations=400, seed=1)

    assert low <= point <= high


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    confidence, correct, case_ids = _clustered_data()
    first = clustered_bootstrap_ci(confidence, correct, case_ids, iterations=100, seed=7)
    second = clustered_bootstrap_ci(confidence, correct, case_ids, iterations=100, seed=7)
    assert first == second


def test_a_degenerate_group_yields_nan_rather_than_raising():
    """Every bootstrap draw has one class, so there is no interval to report."""
    low, high = clustered_bootstrap_ci([0.9, 0.8], [1, 1], ["a", "b"], iterations=10)
    assert low != low and high != high  # NaN is the only value not equal to itself


# --- calibration --------------------------------------------------------------------


def test_three_distinct_values_with_ten_bins_does_not_raise():
    """Regression test. pd.qcut(x, 10) raises 'Bin edges must be unique' on SC data,
    which at N=5 has exactly three distinct values. That is a certainty, not a risk."""
    confidence = [0.6] * 10 + [0.8] * 10 + [1.0] * 10
    correct = [0] * 5 + [1] * 5 + [0] * 3 + [1] * 7 + [0] * 1 + [1] * 9

    error = expected_calibration_error(confidence, correct, max_bins=10)

    assert 0.0 <= error <= 1.0


def test_raw_qcut_really_does_raise_on_that_input():
    """Pins the reason the adaptive binning exists, so nobody 'simplifies' it back."""
    import pandas as pd

    with pytest.raises(ValueError, match="Bin edges must be unique"):
        pd.qcut([0.6] * 10 + [0.8] * 10 + [1.0] * 10, 10)


def test_few_distinct_values_bin_by_value():
    _, n_bins = bin_indices([0.6, 0.8, 1.0, 0.6, 0.8], max_bins=10)
    assert n_bins == 3


def test_many_distinct_values_use_quantile_bins():
    _, n_bins = bin_indices([i / 100 for i in range(100)], max_bins=10)
    assert n_bins == 10


def test_identical_values_collapse_to_one_bin():
    indices, n_bins = bin_indices([0.7] * 20, max_bins=10)
    assert n_bins == 1
    assert set(indices) == {0}


def test_bin_indices_are_ordered_by_confidence():
    indices, _ = bin_indices([1.0, 0.6, 0.8], max_bins=10)
    assert list(indices) == [2, 0, 1]


def test_a_perfectly_calibrated_set_has_near_zero_ece():
    """Half the 0.5-confidence rows are right; all the 1.0-confidence rows are right."""
    confidence = [0.5] * 100 + [1.0] * 100
    correct = [1] * 50 + [0] * 50 + [1] * 100

    assert expected_calibration_error(confidence, correct) < 0.01


def test_a_fully_overconfident_set_has_ece_of_one():
    assert expected_calibration_error([1.0] * 50, [0] * 50) == pytest.approx(1.0)


def test_a_confident_perfect_predictor_has_zero_brier():
    assert brier_score([1.0, 1.0, 0.0], [1, 1, 0]) == 0.0


def test_brier_fully_penalises_a_confident_error():
    assert brier_score([1.0], [0]) == 1.0


def test_calibration_points_return_one_entry_per_bin():
    confidence = [0.6] * 10 + [0.8] * 10 + [1.0] * 10
    correct = [0] * 5 + [1] * 5 + [0] * 3 + [1] * 7 + [1] * 10

    points = calibration_points(confidence, correct, max_bins=10)

    assert len(points) == 3
    assert points[0] == (pytest.approx(0.6), pytest.approx(0.5), 10)
    assert points[1] == (pytest.approx(0.8), pytest.approx(0.7), 10)
    assert points[2] == (pytest.approx(1.0), pytest.approx(1.0), 10)
