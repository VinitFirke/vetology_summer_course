"""The effort mapping table and the cost estimator that gates every paid run."""

import pytest

from uncertainty.config import (
    REPLICATES,
    TIERS,
    UQ_PROVIDERS,
    effort_for,
    estimate_cost,
    samples_path,
)


def test_every_provider_maps_every_tier():
    for provider in UQ_PROVIDERS:
        for tier in TIERS:
            assert effort_for(provider, tier), f"{provider}/{tier} has no effort string"


def test_replicate_count_is_odd():
    """An even N can split evenly on a binary label and have no majority."""
    assert REPLICATES % 2 == 1


def test_mistral_medium_and_high_are_the_same_configuration():
    """Deliberate: the gap between those two columns is the sampling-noise floor."""
    assert effort_for("mistral", "medium") == effort_for("mistral", "high")


def test_openai_tiers_are_distinct():
    values = {effort_for("openai", tier) for tier in TIERS}
    assert len(values) == 3


def test_unknown_tier_raises():
    with pytest.raises(KeyError):
        effort_for("openai", "extreme")


def test_sample_paths_are_distinct_per_provider_and_tier():
    a = samples_path("kimi", "low")
    b = samples_path("kimi", "high")
    c = samples_path("openai", "low")
    assert len({a, b, c}) == 3


def test_cost_estimate_for_a_known_input():
    """2 cases, medium tier only, 5 replicates, on gpt-5.6-luna at $0.20/$1.20.

    replicate calls = 2 x 5 x 1 = 10, CE calls = 2 x 1 = 2
    input  = 10 x 1064 + 2 x 1400             = 13,440
    output = 10 x 1384 x 0.55 x 1.0 + 2 x 800 =  9,212
    """
    estimate = estimate_cost("openai", ("medium",), n_cases=2, replicates=5)

    assert estimate.calls == 12
    assert estimate.input_tokens == 13_440
    assert estimate.output_tokens == 9_212
    assert estimate.dollars == pytest.approx(0.0137, abs=0.0005)


def test_high_tier_costs_more_than_low_tier():
    low = estimate_cost("kimi", ("low",), n_cases=50, replicates=5)
    high = estimate_cost("kimi", ("high",), n_cases=50, replicates=5)
    assert high.dollars > low.dollars
    assert high.input_tokens == low.input_tokens  # only output scales with effort


def test_full_run_lands_inside_the_budget():
    """Guards against a price or baseline edit quietly blowing the budget."""
    budgets = {"openai": 10.0, "mistral": 10.0, "kimi": 20.0}
    for provider, budget in budgets.items():
        estimate = estimate_cost(provider, TIERS, n_cases=50, replicates=REPLICATES)
        assert estimate.dollars < budget, f"{provider}: ${estimate.dollars} exceeds ${budget}"
