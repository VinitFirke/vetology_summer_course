"""Collect uncertainty samples. This is the program that spends money.

    python uq_main.py --provider kimi --dry-run     show the bill, make no calls
    python uq_main.py --provider kimi --yes         run it
    python uq_main.py --provider all --yes          all three, in sequence

Resumable: re-running the same command charges only for what is missing.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from classifier import config as classifier_config
from classifier.config import Settings, load_settings
from classifier.csv_io import read_cases
from classifier.prompt import load_prompt
from classifier.schemas import RadiologyCase
from uncertainty import config as uq_config
from uncertainty.llm import build_tier_model
from uncertainty.sample import (
    RunPlan,
    append_record,
    elicit_confidence,
    labels_from_record,
    plan_run,
    read_samples,
    sample_replicate,
)


def log_tier_usage(provider: str, tier: str, records: list[dict]) -> None:
    """Append this tier's real token spend to logs/token_usage.md.

    Both pipelines report spend in one place, so the running total is one file rather
    than two. Format mirrors main.py's append_token_log table.
    """
    if not records:
        return

    inputs = sum(r.get("usage", {}).get("input_tokens", 0) for r in records)
    outputs = sum(r.get("usage", {}).get("output_tokens", 0) for r in records)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    classifier_config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not classifier_config.LOG_FILE.exists()
    with classifier_config.LOG_FILE.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write("# Token usage log\n")
        handle.write(f"\n## UQ run {stamp}\n\n")
        handle.write("| provider | model | tier | effort | calls | input | output |\n")
        handle.write("|---|---|---|---|---|---|---|\n")
        handle.write(
            f"| {provider} | `{uq_config.UQ_MODEL_IDS[provider]}` | {tier} | "
            f"`{uq_config.effort_for(provider, tier)}` | {len(records)} | "
            f"{inputs:,} | {outputs:,} |\n"
        )


def run_one_tier(
    provider: str,
    tier: str,
    settings: Settings,
    cases: list[RadiologyCase],
    plan: RunPlan,
    workers: int,
) -> None:
    """Fill the shortfall for one provider at one tier."""
    model = build_tier_model(provider, tier, settings)
    prompt = load_prompt(classifier_config.PROMPT_FILE)
    ce_prompt = load_prompt(uq_config.CE_PROMPT_FILE)

    by_id = {case.case_id: case for case in cases}
    samples_file = uq_config.samples_path(provider, tier)
    existing = read_samples(samples_file)

    jobs: list[tuple[str, int]] = []
    for (job_tier, case_id), missing in plan.work.items():
        if job_tier != tier:
            continue
        done = len(existing.get(case_id, []))
        jobs.extend((case_id, done + offset + 1) for offset in range(missing))

    effort = uq_config.effort_for(provider, tier)
    print(f"\n=== {provider} / {tier} (effort={effort}) - {len(jobs)} replicate calls ===")

    def do_replicate(job: tuple[str, int]) -> dict:
        case_id, replicate = job
        return sample_replicate(model, prompt, by_id[case_id], provider, tier, replicate)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(do_replicate, job): job for job in jobs}
        for future in as_completed(futures):
            case_id, replicate = futures[future]
            try:
                append_record(samples_file, future.result())
            except Exception as error:  # noqa: BLE001 - one bad case must not end the run
                append_record(
                    uq_config.failures_path(provider, tier),
                    {"case_id": case_id, "replicate": replicate, "error": str(error)},
                )
                print(f"  {case_id} r{replicate} FAILED: {error}")
                continue
            completed += 1
            if completed % 25 == 0:
                print(f"  {completed}/{len(jobs)} replicates")

    # CE runs after the replicates, because it rates replicate 1's labels.
    refreshed = read_samples(samples_file)
    ce_file = uq_config.ce_path(provider, tier)
    pending = sorted(cid for (t, cid) in plan.ce_needed if t == tier and cid in refreshed)
    print(f"  {len(pending)} CE calls")

    def do_ce(case_id: str) -> dict:
        labels = labels_from_record(refreshed[case_id][0])
        return elicit_confidence(model, ce_prompt, by_id[case_id], labels, provider, tier)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(do_ce, cid): cid for cid in pending}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                append_record(ce_file, future.result())
            except Exception as error:  # noqa: BLE001
                append_record(
                    uq_config.failures_path(provider, tier),
                    {"case_id": case_id, "stage": "ce", "error": str(error)},
                )
                print(f"  {case_id} CE FAILED: {error}")

    # Report real spend from what actually landed on disk, not from the estimate.
    final = read_samples(samples_file)
    log_tier_usage(provider, tier, [r for records in final.values() for r in records])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=[*uq_config.UQ_PROVIDERS, "all"], default="all")
    parser.add_argument("--tier", choices=[*uq_config.TIERS, "all"], default="all")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--replicates", type=int, default=uq_config.REPLICATES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="print the bill and exit")
    parser.add_argument("--yes", action="store_true", help="required to make any paid call")
    args = parser.parse_args()

    if args.replicates % 2 == 0:
        raise SystemExit(
            f"--replicates must be odd (got {args.replicates}); an even count can tie "
            "on a binary label and leave no majority."
        )

    settings = load_settings()
    cases = read_cases(classifier_config.INPUT_CSV)
    if args.limit:
        cases = cases[: args.limit]

    providers = list(uq_config.UQ_PROVIDERS) if args.provider == "all" else [args.provider]
    tiers = uq_config.TIERS if args.tier == "all" else (args.tier,)

    plans = {
        provider: plan_run(provider, tiers, cases, args.replicates, uq_config.UQ_DIR)
        for provider in providers
    }

    print(f"{len(cases)} cases, {args.replicates} replicates, tiers: {', '.join(tiers)}\n")
    total = 0.0
    for provider, plan in plans.items():
        print(plan.estimate.render(provider, len(cases), tiers, args.replicates))
        total += plan.estimate.dollars
    print(f"\n  TOTAL  ~${total:.2f}")

    if args.dry_run:
        print("\n(dry run - no calls made)")
        return
    if not args.yes:
        print("\nRe-run with --yes to proceed.")
        return

    for provider, plan in plans.items():
        if plan.is_empty:
            print(f"\n{provider}: nothing left to do")
            continue
        for tier in tiers:
            run_one_tier(provider, tier, settings, cases, plan, args.workers)

    print(f"\nSamples written to {uq_config.UQ_DIR}")


if __name__ == "__main__":
    main()
