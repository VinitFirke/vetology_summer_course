"""The two proxy primitives. Pure functions, no API, no files."""

import itertools

import pytest

from classifier.schemas import FindingName
from uncertainty.proxies import (
    build_rows,
    confidence_elicitation,
    matches,
    sample_consistency,
)


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


# --- row assembly -------------------------------------------------------------------

ALL_FINDINGS = [f.value for f in FindingName]


def _replicate(**overrides) -> dict:
    """One replicate: every finding normal, except those overridden."""
    labels = {name: "normal" for name in ALL_FINDINGS}
    labels.update(overrides)
    return labels


def _gold(**overrides) -> dict:
    labels = {name: "Normal" for name in ALL_FINDINGS}
    labels.update(overrides)
    return labels


def test_matches_is_case_insensitive():
    """Gold writes 'Abnormal', the models write 'abnormal'."""
    assert matches("Abnormal", "abnormal") is True
    assert matches("Normal", "normal") is True
    assert matches("Abnormal", "normal") is False


def test_builds_one_sc_row_per_finding():
    samples = {"A": [_replicate() for _ in range(5)]}
    rows = build_rows(samples, ce_scores={}, gold={"A": _gold()}, provider="kimi", tier="low")

    sc_rows = [r for r in rows if r.proxy == "SC"]
    assert len(sc_rows) == 19


def test_diseased_lungs_never_appears():
    """Derived in csv_io, never judged, so it has no CE score and no token probability."""
    samples = {"A": [_replicate() for _ in range(5)]}
    gold = {"A": {**_gold(), "diseased_lungs": "Normal"}}
    rows = build_rows(samples, ce_scores={}, gold=gold, provider="kimi", tier="low")

    assert all(row.finding != "diseased_lungs" for row in rows)


def test_ce_rows_appear_only_where_a_score_exists():
    samples = {"A": [_replicate() for _ in range(5)]}
    ce_scores = {"A": {"cardiomegaly": 90}}
    rows = build_rows(samples, ce_scores, gold={"A": _gold()}, provider="kimi", tier="low")

    ce_rows = [r for r in rows if r.proxy == "CE"]
    assert len(ce_rows) == 1
    assert ce_rows[0].finding == "cardiomegaly"
    assert ce_rows[0].confidence == pytest.approx(0.90)


def test_ce_answer_comes_from_replicate_one_not_the_majority():
    """CE rates the SinglePass answer; SC reports the majority. They can disagree."""
    replicates = [
        _replicate(cardiomegaly="abnormal"),  # replicate 1
        _replicate(),
        _replicate(),
        _replicate(),
        _replicate(),
    ]
    rows = build_rows(
        {"A": replicates},
        ce_scores={"A": {"cardiomegaly": 70}},
        gold={"A": _gold()},
        provider="kimi",
        tier="low",
    )
    by_proxy = {r.proxy: r for r in rows if r.finding == "cardiomegaly"}

    assert by_proxy["CE"].answer == "abnormal"
    assert by_proxy["SC"].answer == "normal"
    assert by_proxy["SC"].confidence == pytest.approx(0.8)


def test_correct_flag_reflects_the_gold_standard():
    samples = {"A": [_replicate(pneumonia="abnormal") for _ in range(5)]}
    gold = {"A": _gold(pneumonia="Abnormal")}
    rows = build_rows(samples, ce_scores={}, gold=gold, provider="kimi", tier="low")

    pneumonia = next(r for r in rows if r.finding == "pneumonia")
    esophagitis = next(r for r in rows if r.finding == "esophagitis")
    assert pneumonia.correct == 1
    assert esophagitis.correct == 1  # both normal, also a match


def test_a_wrong_label_scores_zero():
    samples = {"A": [_replicate() for _ in range(5)]}
    gold = {"A": _gold(pneumonia="Abnormal")}
    rows = build_rows(samples, ce_scores={}, gold=gold, provider="kimi", tier="low")

    pneumonia = next(r for r in rows if r.finding == "pneumonia")
    assert pneumonia.correct == 0


def test_cases_below_the_replicate_floor_are_dropped():
    samples = {"A": [_replicate() for _ in range(2)], "B": [_replicate() for _ in range(5)]}
    gold = {"A": _gold(), "B": _gold()}
    rows = build_rows(samples, ce_scores={}, gold=gold, provider="kimi", tier="low")

    assert {row.case_id for row in rows} == {"B"}


def test_surviving_replicate_count_is_recorded():
    samples = {"A": [_replicate() for _ in range(4)]}
    rows = build_rows(samples, ce_scores={}, gold={"A": _gold()}, provider="kimi", tier="low")

    assert all(row.n_replicates == 4 for row in rows)


def test_cases_absent_from_the_gold_standard_are_skipped():
    samples = {"UNKNOWN": [_replicate() for _ in range(5)]}
    rows = build_rows(samples, ce_scores={}, gold={}, provider="kimi", tier="low")

    assert rows == []


def test_findings_with_a_blank_gold_label_are_skipped():
    """An unscored cell is not the same as a normal one."""
    samples = {"A": [_replicate() for _ in range(5)]}
    gold = {"A": {**_gold(), "pneumonia": ""}}
    rows = build_rows(samples, ce_scores={}, gold=gold, provider="kimi", tier="low")

    assert all(row.finding != "pneumonia" for row in rows)
    assert len([r for r in rows if r.proxy == "SC"]) == 18


def test_provider_and_tier_are_stamped_on_every_row():
    samples = {"A": [_replicate() for _ in range(5)]}
    rows = build_rows(samples, ce_scores={}, gold={"A": _gold()}, provider="mistral", tier="high")

    assert all(r.provider == "mistral" and r.tier == "high" for r in rows)
