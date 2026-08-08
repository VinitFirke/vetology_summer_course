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

# Which tiers are actually run per provider.
#
# Mistral's medium and high map to the same effort string, so medium is a duplicate
# condition rather than a distinct one. It was worth keeping as a sampling-noise control
# while it was free; the 2026-08-08 smoke run measured it at $8.61, which would put
# Mistral at $18.50 against a $10 budget. Dropping the duplicate brings it to $9.89 and
# costs nothing but the control.
PROVIDER_TIERS: dict[Provider, tuple[Tier, ...]] = {
    "openai": TIERS,
    "mistral": ("low", "high"),
    "kimi": TIERS,
}


def tiers_for(provider: Provider, requested: tuple[Tier, ...] | None = None) -> tuple[Tier, ...]:
    """The tiers actually run for a provider, optionally narrowed by a CLI --tier.

    Order follows TIERS so results tables read low, medium, high regardless of provider.
    """
    available = PROVIDER_TIERS[provider]
    if requested is None:
        return available
    return tuple(tier for tier in available if tier in requested)

# Model id -> ($ per 1M input tokens, $ per 1M output tokens). Checked 2026-08-07.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (0.20, 1.20),
    "mistral-medium-latest": (1.50, 7.50),
    "kimi-k3": (3.00, 15.00),
}

# MEASURED 2026-08-08 by a 2-case smoke run per provider per tier, slim CaseLabels
# schema. Mean (input, output) tokens for one replicate call.
#
# Replaces the earlier BASELINE_TOKENS x SLIM_OUTPUT_FACTOR x EFFORT_OUTPUT_MULTIPLIER
# arrangement. A single global multiplier could not represent what the measurements
# show: the low tier costs 0.88x medium on openai but 0.06x on mistral, because
# mistral's low tier is `none`, which disables reasoning outright. Per-provider,
# per-tier numbers are both simpler and correct.
MEASURED_TOKENS: dict[Provider, dict[Tier, tuple[int, int]]] = {
    "openai": {"low": (1009, 416), "medium": (1009, 474), "high": (1009, 904)},
    # mistral's medium and high are the same configuration (effort=high), so their four
    # samples are pooled: [2033, 9622, 1870, 1654], mean 3795. That 9622 is one runaway
    # reasoning trace. The mean rather than the median is used deliberately - a budget
    # guard should over-estimate, and outliers are what you actually get billed for.
    "mistral": {"low": (1018, 360), "medium": (1018, 3795), "high": (1018, 3795)},
    "kimi": {"low": (1173, 308), "medium": (1173, 1018), "high": (1172, 1330)},
}

# MEASURED the same way: one CE call per case per tier. CE output scales with effort too
# (259 tokens at openai/low, 3171 at mistral/high), which the old flat estimate missed.
MEASURED_CE_TOKENS: dict[Provider, dict[Tier, tuple[int, int]]] = {
    "openai": {"low": (985, 320), "medium": (985, 426), "high": (985, 527)},
    "mistral": {"low": (1020, 396), "medium": (1020, 2761), "high": (1020, 2761)},
    "kimi": {"low": (1159, 282), "medium": (1159, 1306), "high": (1158, 1137)},
}

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
    """Estimate a run's cost from the measured per-provider, per-tier token counts."""
    input_tokens = 0
    output_tokens = 0

    for tier in tiers:
        replicate_in, replicate_out = MEASURED_TOKENS[provider][tier]
        ce_in, ce_out = MEASURED_CE_TOKENS[provider][tier]

        input_tokens += n_cases * replicates * replicate_in + n_cases * ce_in
        output_tokens += n_cases * replicates * replicate_out + n_cases * ce_out

    price_in, price_out = PRICES[UQ_MODEL_IDS[provider]]
    dollars = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out

    return CostEstimate(
        calls=n_cases * replicates * len(tiers) + n_cases * len(tiers),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dollars=round(dollars, 4),
    )
