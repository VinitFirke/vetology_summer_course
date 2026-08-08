"""Classify feline thoracic radiology reports with one or more LLM providers.

Examples:
    python main.py --smoke                  2 cases on every provider, to check wiring
    python main.py --provider openai        full 50-case run
    python main.py --provider all           all three providers
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from classifier import config
from classifier.classify import CaseResult, Usage, classify_case, labels_from
from classifier.csv_io import build_label_row, read_cases, read_dataframe, write_labeled_csv
from classifier.llm import build_model
from classifier.prompt import Prompt, load_prompt
from classifier.schemas import COL_CASE_ID, LABEL_COLUMNS, Label, RadiologyCase


def load_existing(provider: config.Provider) -> tuple[dict[str, dict[str, Label]], list[dict]]:
    """Read back a previous run's labels and reasoning, for --resume.

    Returns ({case_id: labels}, [reasoning dicts]). Only cases whose labels were
    actually written count as done.
    """
    csv_path = config.output_csv_path(provider)
    labels: dict[str, dict[str, Label]] = {}
    if csv_path.exists():
        df = read_dataframe(csv_path)
        for _, row in df.iterrows():
            case_id = str(row[COL_CASE_ID]).strip()
            values = {col: str(row[col]).strip() for col in LABEL_COLUMNS}
            if all(values.values()):
                labels[case_id] = values

    reasoning_path = config.reasoning_json_path(provider)
    reasoning: list[dict] = []
    if reasoning_path.exists():
        stored = json.loads(reasoning_path.read_text(encoding="utf-8"))
        reasoning = [c for c in stored.get("cases", []) if c.get("case_id") in labels]

    return labels, reasoning


def run_provider(
    provider: config.Provider,
    settings: config.Settings,
    prompt: Prompt,
    cases: list[RadiologyCase],
    resume: bool = False,
) -> tuple[Usage, list[str]]:
    """Classify every case with one provider, writing results as they arrive.

    The CSV is rewritten after each case, so stopping the run early leaves a
    partially filled file rather than nothing.
    """
    model = build_model(provider, settings)
    model_id = settings.model_for(provider)

    labels_by_case: dict[str, dict[str, Label]] = {}
    carried_reasoning: list[dict] = []
    if resume:
        labels_by_case, carried_reasoning = load_existing(provider)
        already_done = len(labels_by_case)
        cases = [c for c in cases if c.case_id not in labels_by_case]
        print(f"\n=== {provider} ({model_id}) - resuming, {already_done} already done ===")
    else:
        print(f"\n=== {provider} ({model_id}) - {len(cases)} cases ===")

    if not cases:
        print("  nothing left to do")
        return Usage(), []

    results: list[CaseResult] = []
    failed: list[str] = []
    total = Usage()

    for index, case in enumerate(cases, start=1):
        try:
            result = classify_case(model, prompt, case)
        except RuntimeError as error:
            print(f"  [{index}/{len(cases)}] {case.case_id} FAILED: {error}")
            failed.append(case.case_id)
            continue

        labels_by_case[case.case_id] = build_label_row(labels_from(result.classification))
        results.append(result)
        total = total + result.usage

        abnormal = sum(1 for v in labels_by_case[case.case_id].values() if v == "abnormal")
        print(
            f"  [{index}/{len(cases)}] {case.case_id}: {abnormal}/20 abnormal "
            f"({result.usage.total_tokens} tokens)"
        )

        write_labeled_csv(config.INPUT_CSV, config.output_csv_path(provider), labels_by_case)

    reasoning_path = config.reasoning_json_path(provider)
    reasoning_path.write_text(
        json.dumps(
            {
                "provider": provider,
                "model": model_id,
                "prompt_version": prompt.version,
                "cases": carried_reasoning + [r.classification.model_dump() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"  -> {config.output_csv_path(provider).name}")
    print(f"  -> {reasoning_path.name}")
    return total, failed


def append_token_log(
    rows: list[tuple[config.Provider, str, int, Usage, list[str]]],
) -> None:
    """Append one section per run to logs/token_usage.md."""
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not config.LOG_FILE.exists()

    lines: list[str] = []
    if is_new:
        lines.append("# Token usage log\n")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n## Run {stamp}\n")
    lines.append("| provider | model | cases | input | output | total | failed |")
    lines.append("|---|---|---|---|---|---|---|")
    for provider, model_id, n_cases, usage, failed in rows:
        lines.append(
            f"| {provider} | `{model_id}` | {n_cases} | {usage.input_tokens:,} | "
            f"{usage.output_tokens:,} | {usage.total_tokens:,} | "
            f"{len(failed) or '-'} |"
        )

    with config.LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nToken usage appended to {config.LOG_FILE.relative_to(config.PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=[*config.PROVIDERS, "all"],
        default="all",
        help="which provider to run (default: all)",
    )
    parser.add_argument("--limit", type=int, help="only classify the first N cases")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run 2 cases on every provider, to check wiring before a full run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases already labelled in the output file (retries failures only)",
    )
    args = parser.parse_args()

    settings = config.load_settings()
    prompt = load_prompt(config.PROMPT_FILE)
    cases = read_cases(config.INPUT_CSV)

    limit = 2 if args.smoke else args.limit
    if limit:
        cases = cases[:limit]

    providers: list[config.Provider] = (
        list(config.PROVIDERS) if args.smoke or args.provider == "all" else [args.provider]
    )

    print(f"Loaded {len(cases)} cases from {Path(config.INPUT_CSV).name}")
    print(f"Prompt version {prompt.version}")

    rows = []
    for provider in providers:
        usage, failed = run_provider(provider, settings, prompt, cases, resume=args.resume)
        rows.append((provider, settings.model_for(provider), len(cases), usage, failed))

    append_token_log(rows)


if __name__ == "__main__":
    main()
