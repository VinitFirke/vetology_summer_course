"""Configuration for the uncertainty runs: the effort axis, paths, and money.

Every number that decides what a run costs lives here, in one visible table, so a
budget check is a matter of reading one file rather than tracing a call graph.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from classifier.config import PROJECT_ROOT, Provider

Tier = Literal["low", "medium", "high"]

TIERS: tuple[Tier, ...] = ("low", "medium", "high")
UQ_PROVIDERS: tuple[Provider, ...] = ("openai", "mistral", "kimi")

# Odd on purpose: a binary label cannot split evenly across an odd number of replicates,
# so majority vote needs no tie-break rule anywhere in the code.
REPLICATES = 5

# Distinct from classifier.config.DEFAULT_MODEL_IDS. gpt-5.5 at $30/1M output puts this
# design at ~$28 against a $10 budget; luna runs the identical design for ~$1.40.
UQ_MODEL_IDS: dict[Provider, str] = {
    "openai": "gpt-5.6-luna",
    "mistral": "mistral-medium-latest",
    "kimi": "kimi-k3",
}

# Canonical tier -> the string each provider actually accepts. Verified by probe.py.
# Mistral's medium and high are deliberately identical: that pair is a negative control,
# and the gap between those two result columns is run-to-run sampling noise.
EFFORT_LEVELS: dict[Provider, dict[Tier, str]] = {
    "openai": {"low": "low", "medium": "medium", "high": "high"},
    "mistral": {"low": "none", "medium": "high", "high": "high"},
    "kimi": {"low": "low", "medium": "high", "high": "max"},
}

# Model id -> ($ per 1M input tokens, $ per 1M output tokens). Checked 2026-08-07.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (0.20, 1.20),
    "mistral-medium-latest": (1.50, 7.50),
    "kimi-k3": (3.00, 15.00),
}

# Measured per-case tokens at medium effort with the FULL schema, from logs/token_usage.md.
BASELINE_TOKENS: dict[Provider, tuple[int, int]] = {
    "openai": (1064, 1384),
    "mistral": (1050, 1050),
    "kimi": (1200, 1000),
}

# Dropping evidence and reasoning removes roughly 45% of output tokens.
SLIM_OUTPUT_FACTOR = 0.55

# One CE call: the case text plus 19 labels in, 19 integers plus thinking out.
CE_TOKENS: tuple[int, int] = (1400, 800)

# ESTIMATES. Task 11 replaces these with values measured by a 2-case smoke run per tier.
EFFORT_OUTPUT_MULTIPLIER: dict[Tier, float] = {"low": 0.5, "medium": 1.0, "high": 2.5}

UQ_DIR = PROJECT_ROOT / "dataset_LLM_uncertainty"
FIGURES_DIR = UQ_DIR / "figures"
PROXY_CSV = UQ_DIR / "uncertainty_proxies.csv"
RESULTS_XLSX = UQ_DIR / "uncertainty_results.xlsx"
CE_PROMPT_FILE = PROJECT_ROOT / "prompts" / "ce_prompt.json"

# Rows below this many surviving replicates are dropped, matching the paper's exclusion
# of questions where the model produced an error response.
MIN_REPLICATES = 3


def effort_for(provider: Provider, tier: Tier) -> str:
    """The provider-specific effort string for a canonical tier."""
    return EFFORT_LEVELS[provider][tier]


def samples_path(provider: Provider, tier: Tier) -> Path:
    return UQ_DIR / f"samples_{provider}_{tier}.jsonl"


def ce_path(provider: Provider, tier: Tier) -> Path:
    return UQ_DIR / f"ce_{provider}_{tier}.jsonl"


def failures_path(provider: Provider, tier: Tier) -> Path:
    return UQ_DIR / f"failures_{provider}_{tier}.jsonl"


class CostEstimate(BaseModel):
    """What a planned run will cost, before any of it is spent."""

    calls: int
    input_tokens: int
    output_tokens: int
    dollars: float

    def render(
        self,
        provider: Provider,
        n_cases: int,
        tiers: tuple[Tier, ...],
        replicates: int = REPLICATES,
    ) -> str:
        """One provider's block of the cost guard.

        `replicates` is passed in rather than read from the module constant: the guard's
        whole job is to state accurately what is about to happen, so it must reflect the
        --replicates the caller actually chose.
        """
        return (
            f"{provider} / {UQ_MODEL_IDS[provider]}\n"
            f"  {n_cases} cases x {replicates} replicates x {len(tiers)} tiers "
            f"= {self.calls} calls (incl. CE)\n"
            f"  est. {self.input_tokens / 1e6:.2f}M input, "
            f"{self.output_tokens / 1e6:.2f}M output  ->  ~${self.dollars:.2f}"
        )


def estimate_cost(
    provider: Provider,
    tiers: tuple[Tier, ...],
    n_cases: int,
    replicates: int = REPLICATES,
) -> CostEstimate:
    """Estimate a run's cost. Input is flat per call; only output scales with effort."""
    in_per_case, out_per_case = BASELINE_TOKENS[provider]
    ce_in, ce_out = CE_TOKENS

    replicate_calls = n_cases * replicates * len(tiers)
    ce_calls = n_cases * len(tiers)
    input_tokens = replicate_calls * in_per_case + ce_calls * ce_in

    output_tokens = 0.0
    for tier in tiers:
        multiplier = EFFORT_OUTPUT_MULTIPLIER[tier]
        output_tokens += n_cases * replicates * out_per_case * SLIM_OUTPUT_FACTOR * multiplier
        output_tokens += n_cases * ce_out * multiplier

    price_in, price_out = PRICES[UQ_MODEL_IDS[provider]]
    dollars = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out

    return CostEstimate(
        calls=replicate_calls + ce_calls,
        input_tokens=input_tokens,
        output_tokens=round(output_tokens),
        dollars=round(dollars, 4),
    )
