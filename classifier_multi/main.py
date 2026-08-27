"""Command line entry point: one command to classify, one to score.

They are separate subcommands rather than one run because that is what makes the
boundary observable. `classify` never imports evaluate - the import sits inside the
evaluate handler - so a classification run cannot read the answers even by accident.
tests/test_gold_boundary.py checks that this holds.
"""

import argparse
import json

from classifier_multi import config
from classifier_multi.categories import CATEGORY_NAMES, get_category
from classifier_multi.classify import classify_case
from classifier_multi.csv_io import build_label_row, read_cases, write_labeled_csv
from classifier_multi.llm import build_model
from classifier_multi.prompt import load_prompt
from classifier_multi.schemas import labels_from


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classifier_multi")
    subcommands = parser.add_subparsers(dest="command", required=True)

    classify_parser = subcommands.add_parser(
        "classify", help="label one category's cases with one provider"
    )
    classify_parser.add_argument("--category", required=True, choices=CATEGORY_NAMES)
    classify_parser.add_argument("--provider", required=True, choices=config.PROVIDERS)
    classify_parser.add_argument(
        "--variant", default="zeroshot", choices=config.VARIANTS
    )

    evaluate_parser = subcommands.add_parser(
        "evaluate", help="score predictions against the gold standard"
    )
    evaluate_parser.add_argument("--category", required=True, choices=CATEGORY_NAMES)
    evaluate_parser.add_argument(
        "--variant", default="zeroshot", choices=config.VARIANTS
    )

    return parser


def run_classify(args: argparse.Namespace) -> int:
    category = get_category(args.category)
    settings = config.load_settings()
    model = build_model(args.provider, settings)
    prompt = load_prompt(config.prompt_path(category, args.variant))
    cases = read_cases(config.input_csv_path(category), category)

    labels_by_case: dict[str, dict[str, str]] = {}
    reasoning: list[dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        result = classify_case(model, prompt, category, case)
        labels = labels_from(result.classification)
        labels_by_case[case.case_id] = build_label_row(category, labels)
        reasoning.append(result.classification.model_dump(mode="json"))

    predictions = config.predictions_path(category, args.provider, args.variant)
    write_labeled_csv(
        config.input_csv_path(category), predictions, category, labels_by_case
    )

    reasoning_file = config.reasoning_path(category, args.provider, args.variant)
    reasoning_file.parent.mkdir(parents=True, exist_ok=True)
    reasoning_file.write_text(json.dumps(reasoning, indent=2), encoding="utf-8")

    print(f"wrote {predictions}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    # Imported here, not at module scope: this is the only code path allowed to know
    # the gold standard exists.
    from classifier_multi import evaluate

    category = get_category(args.category)
    print(
        f"scoring {category.name} ({args.variant}) "
        f"against {evaluate.gold_csv_path(category)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "classify":
        return run_classify(args)
    return run_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
