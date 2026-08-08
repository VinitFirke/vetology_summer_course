"""The two proxy primitives. Pure functions, no API, no files."""

import itertools

import pytest

from uncertainty.proxies import confidence_elicitation, sample_consistency


def test_unanimous_replicates_give_full_confidence():
    assert sample_consistency(["abnormal"] * 5) == ("abnormal", 1.0)


def test_three_two_split_gives_point_six():
    answer, confidence = sample_consistency(
        ["abnormal", "abnormal", "abnormal", "normal", "normal"]
    )
    assert answer == "abnormal"
    assert confidence == pytest.approx(0.6)


def test_four_one_split_gives_point_eight():
    answer, confidence = sample_consistency(
        ["normal", "normal", "normal", "normal", "abnormal"]
    )
    assert answer == "normal"
    assert confidence == pytest.approx(0.8)


def test_the_minority_label_never_wins():
    answer, _ = sample_consistency(
        ["normal", "abnormal", "abnormal", "abnormal", "abnormal"]
    )
    assert answer == "abnormal"


def test_five_replicates_can_never_tie():
    """Exhaustive over all 32 binary combinations - the reason N must be odd."""
    for combination in itertools.product(["normal", "abnormal"], repeat=5):
        _, confidence = sample_consistency(list(combination))
        assert confidence > 0.5


def test_n_equals_five_yields_only_three_distinct_values():
    """Documented limitation: this is why the SC ROC curve has three points."""
    observed = {
        sample_consistency(list(c))[1]
        for c in itertools.product(["normal", "abnormal"], repeat=5)
    }
    assert observed == {0.6, 0.8, 1.0}


def test_confidence_elicitation_scales_to_the_unit_interval():
    assert confidence_elicitation(90) == pytest.approx(0.90)
    assert confidence_elicitation(0) == 0.0
    assert confidence_elicitation(100) == 1.0
