"""Command line for the gold-standard workflow.

    python -m gold_standard.main worklist --sheet canine_thorax
    python -m gold_standard.main show     --sheet canine_thorax --count 25
    python -m gold_standard.main apply    --sheet canine_thorax
    python -m gold_standard.main verify

`show` is the one a reader actually lives in: it prints the next unjudged cases from
the worklist, findings and conclusions only. The AI-report columns are not printed at
all, which is the cheapest possible guarantee that they cannot influence a judgement.
"""

import argparse
from pathlib import Path

from gold_standard.csv_io import apply_labels, column_index, read_sheet
from gold_standard.evidence import expand_all, load_evidence, scored_count
from gold_standard.screen import eligible_case_ids
from gold_standard.sheets import (
    COL_CASE_ID,
    COL_CONCLUSIONS,
    COL_FINDINGS,
    GOLD_DIR,
    SHEETS,
    Sheet,
    backup_csv_path,
    get_sheet,
    gold_csv_path,
)
from gold_standard.verify import check_sheet, coverage_report, labelled_rows
from gold_standard.worklist import build_worklist, load_worklist, save_worklist


def _sheets(name: str | None) -> list[Sheet]:
    return [get_sheet(name)] if name else list(SHEETS.values())


def cmd_screen(args: argparse.Namespace) -> None:
    for sheet in _sheets(args.sheet):
        print(f"{sheet.name}: {len(eligible_case_ids(sheet))} eligible cases")


def cmd_worklist(args: argparse.Namespace) -> None:
    for sheet in _sheets(args.sheet):
        items = build_worklist(sheet, seed=args.seed)
        save_worklist(sheet, items)
        coverage = sum(1 for i in items if i.reason.startswith("coverage:"))
        print(
            f"{sheet.name}: {len(items)} queued "
            f"({coverage} coverage candidates, {len(items) - coverage} random) "
            f"-> {sheet.name} worklist saved"
        )


def cmd_show(args: argparse.Namespace) -> None:
    """Print the next unjudged cases from the worklist."""
    sheet = get_sheet(args.sheet)
    records = load_evidence(sheet)
    items = load_worklist(sheet)

    header, rows = read_sheet(gold_csv_path(sheet))
    case_col = column_index(header, COL_CASE_ID)
    findings_col = column_index(header, COL_FINDINGS)
    conclusions_col = column_index(header, COL_CONCLUSIONS)
    by_case = {row[case_col]: row for row in rows}

    shown = 0
    for item in items:
        if shown >= args.count:
            break
        if item.case_id in records:
            continue
        row = by_case[item.case_id]
        print("=" * 100)
        print(f"CASE {item.case_id}   [{item.reason}]   sheet={sheet.name}")
        print("-" * 100)
        print("FINDINGS:")
        print(row[findings_col].strip())
        print("-" * 100)
        print("CONCLUSIONS:")
        print(row[conclusions_col].strip())
        print()
        shown += 1

    print(f"[{shown} case(s) shown; {scored_count(records)} already scored]")


def cmd_find(args: argparse.Namespace) -> None:
    """Print unjudged eligible cases whose report matches a regex.

    The worklist's coverage block is built once from keywords.py, so when a column
    turns out still to be short after those candidates are read, this is how the next
    ones are found without rebuilding the worklist and disturbing the random tail.
    """
    import re

    from gold_standard.screen import eligible_case_ids, report_text

    sheet = get_sheet(args.sheet)
    records = load_evidence(sheet)
    eligible = set(eligible_case_ids(sheet))
    pattern = re.compile(args.pattern, re.I)

    header, rows = read_sheet(gold_csv_path(sheet))
    case_col = column_index(header, COL_CASE_ID)
    findings_col = column_index(header, COL_FINDINGS)
    conclusions_col = column_index(header, COL_CONCLUSIONS)

    shown = 0
    for row in rows:
        if shown >= args.count:
            break
        case_id = row[case_col]
        if case_id in records or case_id not in eligible:
            continue
        if not pattern.search(report_text(header, row)):
            continue
        print("=" * 100)
        print(f"CASE {case_id}   [found: {args.pattern}]   sheet={sheet.name}")
        print("-" * 100)
        print("FINDINGS:")
        print(row[findings_col].strip())
        print("-" * 100)
        print("CONCLUSIONS:")
        print(row[conclusions_col].strip())
        print()
        shown += 1

    print(f"[{shown} case(s) shown]")


def cmd_record(args: argparse.Namespace) -> None:
    """Merge a batch of judgements into the evidence store.

    The batch file is {case_id: {"abnormal": {finding: {evidence, reasoning}}}}, or
    {case_id: {"skipped": true, "note": "..."}} for a case the worklist offered but
    the report does not support scoring. Merging rather than replacing means a batch
    can be re-recorded after a correction without disturbing the rest.
    """
    import json

    from gold_standard.evidence import CaseRecord, save_evidence

    sheet = get_sheet(args.sheet)
    records = load_evidence(sheet)
    batch = json.loads(Path(args.file).read_text(encoding="utf-8"))

    for case_id, payload in batch.items():
        records[case_id] = CaseRecord(case_id=case_id, **payload)

    save_evidence(sheet, records)
    skipped = sum(1 for r in records.values() if r.skipped)
    print(
        f"{sheet.name}: recorded {len(batch)}, "
        f"store now holds {scored_count(records)} scored + {skipped} skipped"
    )


def cmd_apply(args: argparse.Namespace) -> None:
    """Project the evidence store into the CSV and prove nothing else moved."""
    for sheet in _sheets(args.sheet):
        records = load_evidence(sheet)
        labels = expand_all(sheet, records)
        if not labels:
            print(f"{sheet.name}: no judgements recorded yet")
            continue
        written = apply_labels(gold_csv_path(sheet), sheet, labels)
        from gold_standard.csv_io import verify_untouched

        problems = verify_untouched(gold_csv_path(sheet), backup_csv_path(sheet), sheet)
        status = "clean" if not problems else f"{len(problems)} PROBLEMS"
        print(f"{sheet.name}: {written} rows written, non-label cells {status}")
        for problem in problems[:10]:
            print(f"   {problem}")


def cmd_verify(args: argparse.Namespace) -> None:
    failed = False
    for sheet in _sheets(args.sheet):
        results = check_sheet(sheet, minimum=args.minimum)
        print(f"\n{sheet.name}: {len(labelled_rows(sheet))} cases scored")
        for check, problems in results.items():
            if problems:
                failed = True
                print(f"  {check}: {len(problems)} problem(s)")
                for problem in problems[:10]:
                    print(f"     {problem}")
            else:
                print(f"  {check}: ok")

    report_path = GOLD_DIR / "coverage_report.md"
    report_path.write_text(coverage_report(_sheets(args.sheet), args.minimum), encoding="utf-8")
    print(f"\nCoverage report written to {report_path}")
    if failed:
        raise SystemExit(1)


def main() -> None:
    # --sheet is declared on a shared parent so it can be typed *after* the
    # subcommand, which is the order anyone actually types it in.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sheet", choices=sorted(SHEETS), help="default: all sheets")

    parser = argparse.ArgumentParser(prog="gold_standard", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("screen", parents=[common], help="count eligible cases").set_defaults(
        func=cmd_screen
    )

    worklist = sub.add_parser("worklist", parents=[common], help="build the reading order")
    worklist.add_argument("--seed", type=int, default=20260818)
    worklist.set_defaults(func=cmd_worklist)

    show = sub.add_parser("show", parents=[common], help="print the next unjudged cases")
    show.add_argument("--count", type=int, default=25)
    show.set_defaults(func=cmd_show)

    find = sub.add_parser("find", parents=[common], help="find unjudged cases by regex")
    find.add_argument("--pattern", required=True)
    find.add_argument("--count", type=int, default=5)
    find.set_defaults(func=cmd_find)

    record = sub.add_parser(
        "record", parents=[common], help="merge a batch JSON into the evidence store"
    )
    record.add_argument("--file", required=True)
    record.set_defaults(func=cmd_record)

    sub.add_parser(
        "apply", parents=[common], help="write the evidence store into the CSVs"
    ).set_defaults(func=cmd_apply)

    verify = sub.add_parser("verify", parents=[common], help="run every check")
    verify.add_argument("--minimum", type=int, default=3)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
