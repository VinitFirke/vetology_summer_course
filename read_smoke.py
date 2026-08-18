"""Read the Task 11 smoke run back and report measured tokens per tier.

Answers the two questions the smoke run exists to answer:
  1. What do calls actually cost per tier, so the estimates stop being guesses?
  2. Does the reasoning-effort knob do anything at all on each provider?

    python read_smoke.py

Kept after the smoke run because it is also how you compare estimate against actual once
the full run is in: the same table, computed from whatever is on disk.
"""

import json

from uncertainty.config import EFFORT_OUTPUT_MULTIPLIER, TIERS, UQ_PROVIDERS, effort_for, samples_path


def averages(provider: str, tier: str) -> tuple[int, int, int] | None:
    """(mean input tokens, mean output tokens, n) for one provider at one tier."""
    path = samples_path(provider, tier)
    if not path.exists():
        return None

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None

    n = len(records)
    return (
        round(sum(r["usage"].get("input_tokens", 0) for r in records) / n),
        round(sum(r["usage"].get("output_tokens", 0) for r in records) / n),
        n,
    )


def main() -> None:
    print("Measured tokens per case (slim CaseLabels schema)\n")
    print(f"{'provider':<9}{'tier':<8}{'effort':<9}{'input':>8}{'output':>9}{'n':>4}")
    print("-" * 47)

    measured: dict[str, dict[str, tuple[int, int, int]]] = {}
    for provider in UQ_PROVIDERS:
        measured[provider] = {}
        for tier in TIERS:
            result = averages(provider, tier)
            effort = effort_for(provider, tier)
            if result is None:
                print(f"{provider:<9}{tier:<8}{effort:<9}{'(missing)':>21}")
                continue
            measured[provider][tier] = result
            print(f"{provider:<9}{tier:<8}{effort:<9}{result[0]:>8}{result[1]:>9}{result[2]:>4}")

    print("\n\nDoes the effort knob do anything? (output tokens relative to medium)\n")
    print(f"{'provider':<9}{'low':>8}{'medium':>8}{'high':>8}   verdict")
    print("-" * 52)

    ratios: dict[str, dict[str, float]] = {}
    for provider in UQ_PROVIDERS:
        tiers = measured.get(provider, {})
        if "medium" not in tiers:
            print(f"{provider:<9}  incomplete - rerun the medium smoke")
            continue

        base = tiers["medium"][1] or 1
        row = {t: tiers[t][1] / base for t in TIERS if t in tiers}
        ratios[provider] = row

        cells = "".join(f"{row.get(t, float('nan')):>8.2f}" for t in TIERS)
        if "low" in row and "high" in row:
            spread = abs(row["high"] - row["low"])
            if provider == "mistral":
                verdict = "medium==high is the intended control"
            elif spread < 0.10:
                verdict = "*** IGNORED - low and high match ***"
            else:
                verdict = "responds to effort"
        else:
            verdict = "incomplete"
        print(f"{provider:<9}{cells}   {verdict}")

    if not ratios:
        return

    print("\n\nPaste into uncertainty/config.py\n")
    print("# MEASURED from a 2-case smoke run per tier. Slim schema, so no further discount.")
    print("SLIM_OUTPUT_FACTOR = 1.0")
    print("\nBASELINE_TOKENS: dict[Provider, tuple[int, int]] = {")
    for provider in UQ_PROVIDERS:
        if "medium" in measured.get(provider, {}):
            i, o, _ = measured[provider]["medium"]
            print(f'    "{provider}": ({i}, {o}),')
    print("}")

    print("\nEFFORT_OUTPUT_MULTIPLIER: dict[Tier, float] = {")
    for tier in TIERS:
        values = [r[tier] for r in ratios.values() if tier in r]
        mean = sum(values) / len(values) if values else EFFORT_OUTPUT_MULTIPLIER[tier]
        print(f'    "{tier}": {mean:.2f},')
    print("}")

    print(
        "\nThen: pytest tests/test_uq_config.py -v"
        "\n      (test_cost_estimate_for_a_known_input asserts the OLD numbers -"
        "\n       update its docstring arithmetic and expected values to match)"
        "\n      python uq_main.py --provider all --dry-run"
    )


if __name__ == "__main__":
    main()
