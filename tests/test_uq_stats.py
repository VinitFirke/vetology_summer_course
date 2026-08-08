"""Discrimination statistics. Pure functions over arrays - no files, no API."""

import random

import pytest

from uncertainty.stats import clustered_bootstrap_ci, roc_auc


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
