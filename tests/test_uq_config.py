"""The effort mapping table and the cost estimator that gates every paid run."""

import pytest

from uncertainty.config import (
    MEASURED_CE_TOKENS,
    MEASURED_TOKENS,
    REPLICATES,
    TIERS,
    UQ_PROVIDERS,
    effort_for,
    estimate_cost,
    samples_path,
    tiers_for,
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

    Measured openai/medium: replicate (1009, 474), CE (985, 426).

    replicate calls = 2 x 5 = 10, CE calls = 2
    input  = 10 x 1009 + 2 x 985 = 12,060
    output = 10 x  474 + 2 x 426 =  5,592
    """
    estimate = estimate_cost("openai", ("medium",), n_cases=2, replicates=5)

    assert estimate.calls == 12
    assert estimate.input_tokens == 12_060
    assert estimate.output_tokens == 5_592
    assert estimate.dollars == pytest.approx(0.0091, abs=0.0005)


def test_high_tier_costs_more_than_low_tier():
    low = estimate_cost("kimi", ("low",), n_cases=50, replicates=5)
    high = estimate_cost("kimi", ("high",), n_cases=50, replicates=5)
    assert high.dollars > low.dollars


def test_every_provider_and_tier_has_measured_tokens():
    """A missing cell would raise KeyError mid-estimate, after the guard was trusted."""
    for provider in UQ_PROVIDERS:
        for tier in TIERS:
            assert MEASURED_TOKENS[provider][tier][1] > 0
            assert MEASURED_CE_TOKENS[provider][tier][1] > 0


def test_mistral_medium_and_high_share_one_measurement():
    """They are the same configuration, so pooling their samples is the honest read."""
    assert MEASURED_TOKENS["mistral"]["medium"] == MEASURED_TOKENS["mistral"]["high"]


def test_render_reports_the_replicate_count_it_was_given():
    """The guard's job is to state what will happen; it must not report the default."""
    estimate = estimate_cost("openai", ("low",), n_cases=2, replicates=1)

    text = estimate.render("openai", 2, ("low",), replicates=1)

    assert "2 cases x 1 replicates x 1 tiers" in text
    assert "x 5 replicates" not in text


def test_render_falls_back_to_the_default_replicate_count():
    estimate = estimate_cost("openai", ("low",), n_cases=2)
    assert f"x {REPLICATES} replicates" in estimate.render("openai", 2, ("low",))


def test_full_run_lands_inside_the_budget():
    """Guards against a price or measurement edit quietly blowing the budget.

    This test earned its place: it is what caught mistral at $18.50 against $10 when the
    estimated token counts were replaced with measured ones.
    """
    budgets = {"openai": 10.0, "mistral": 10.0, "kimi": 20.0}
    for provider, budget in budgets.items():
        estimate = estimate_cost(
            provider, tiers_for(provider), n_cases=50, replicates=REPLICATES
        )
        assert estimate.dollars < budget, f"{provider}: ${estimate.dollars} exceeds ${budget}"


def test_mistral_skips_its_duplicate_medium_tier():
    """medium and high are the same effort=high config; running both costs $8.61 twice."""
    assert tiers_for("mistral") == ("low", "high")
    assert tiers_for("openai") == TIERS
    assert tiers_for("kimi") == TIERS


def test_tiers_for_narrows_to_a_requested_subset():
    assert tiers_for("kimi", ("low", "medium")) == ("low", "medium")
    assert tiers_for("mistral", ("low", "medium")) == ("low",)


def test_requesting_only_a_tier_a_provider_skips_yields_nothing():
    """--tier medium must not silently run mistral's high instead."""
    assert tiers_for("mistral", ("medium",)) == ()


def test_tiers_for_preserves_the_canonical_order():
    assert tiers_for("kimi", ("high", "low")) == ("low", "high")
