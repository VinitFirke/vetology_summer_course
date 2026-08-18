"""Checks that must pass before the gold standard is called finished.

Each function returns a list of human-readable problems rather than raising, so one
run reports everything wrong at once instead of stopping at the first fault. An empty
list means clean.
"""

from gold_standard.csv_io import column_index, read_sheet
from gold_standard.sheets import (
    ABNORMAL,
    COL_CASE_ID,
    NORMAL,
    VALID_LABELS,
    Sheet,
    backup_csv_path,
    gold_csv_path,
)


def labelled_rows(sheet: Sheet) -> dict[str, dict[str, str]]:
    """Every row with at least one filled label cell, as {case_id: {column: value}}."""
    header, rows = read_sheet(gold_csv_path(sheet))
    case_col = column_index(header, COL_CASE_ID)
    indices = {c: column_index(header, c) for c in sheet.label_columns}

    labelled: dict[str, dict[str, str]] = {}
    for row in rows:
        values = {c: row[i] for c, i in indices.items()}
        if any(v.strip() for v in values.values()):
            labelled[row[case_col]] = values
    return labelled


def scored_case_ids(sheet: Sheet) -> list[str]:
    return list(labelled_rows(sheet))


def completeness_problems(sheet: Sheet) -> list[str]:
    """A scored row must have every label cell filled with a valid value."""
    problems: list[str] = []
    for case_id, values in labelled_rows(sheet).items():
        for column, value in values.items():
            if value not in VALID_LABELS:
                problems.append(
                    f"case {case_id}: {column} is {value!r}, expected normal or abnormal"
                )
    return problems


def rollup_problems(sheet: Sheet, rows: dict[str, dict[str, str]]) -> list[str]:
    """A derived column must agree with the columns beneath it."""
    problems: list[str] = []
    for case_id, values in rows.items():
        for summary, inputs in sheet.derived.items():
            positives = [i for i in inputs if values.get(i) == ABNORMAL]
            if positives and values.get(summary) != ABNORMAL:
                problems.append(
                    f"case {case_id}: {summary} is {values.get(summary)} "
                    f"but {positives[0]} is abnormal"
                )
            elif not positives and values.get(summary) == ABNORMAL:
                problems.append(
                    f"case {case_id}: {summary} is abnormal but no lung finding is"
                )
    return problems


def coverage_counts(sheet: Sheet) -> dict[str, int]:
    """How many scored cases are abnormal in each label column."""
    rows = labelled_rows(sheet)
    return {
        column: sum(1 for values in rows.values() if values.get(column) == ABNORMAL)
        for column in sheet.label_columns
    }


def coverage_problems(sheet: Sheet, minimum: int = 3) -> list[str]:
    return [
        f"{column} has {count} abnormal cases, below the minimum of {minimum}"
        for column, count in coverage_counts(sheet).items()
        if count < minimum
    ]


def untouched_problems(sheet: Sheet) -> list[str]:
    from gold_standard.csv_io import verify_untouched

    return verify_untouched(gold_csv_path(sheet), backup_csv_path(sheet), sheet)


def check_sheet(sheet: Sheet, minimum: int = 3) -> dict[str, list[str]]:
    """Run every check. Values are problem lists; all empty means the sheet passes."""
    rows = labelled_rows(sheet)
    return {
        "count": (
            []
            if len(rows) == sheet.target_cases
            else [f"{len(rows)} cases scored, target is {sheet.target_cases}"]
        ),
        "completeness": completeness_problems(sheet),
        "rollup": rollup_problems(sheet, rows),
        "coverage": coverage_problems(sheet, minimum),
        "untouched": untouched_problems(sheet),
    }


def coverage_report(sheets: list[Sheet], minimum: int = 3) -> str:
    """A markdown table of abnormal counts per column, per sheet."""
    lines = ["# Coverage report", ""]
    for sheet in sheets:
        rows = labelled_rows(sheet)
        counts = coverage_counts(sheet)
        lines += [
            f"## {sheet.name}",
            "",
            f"Cases scored: {len(rows)} (target {sheet.target_cases})",
            "",
            "| Column | Abnormal | Normal | Meets minimum |",
            "|---|---:|---:|---|",
        ]
        for column in sheet.label_columns:
            abnormal = counts[column]
            lines.append(
                f"| `{column}` | {abnormal} | {len(rows) - abnormal} | "
                f"{'yes' if abnormal >= minimum else 'NO'} |"
            )
        lines.append("")
    return "\n".join(lines)
