"""Reading and rewriting the scoring sheets without disturbing them.

These files are edited in place, so the write path is the riskiest code here. Three
things matter and all three are load-bearing:

  * cp1252, not UTF-8. One sheet holds 742 non-breaking spaces and another a lone
    u-umlaut; opened as UTF-8 they raise, and opened as latin-1 they would be written
    back subtly differently.
  * CRLF row terminators, while quoted report fields contain bare LFs. csv handles
    both correctly only when the file is opened with newline="".
  * pandas is not used. A DataFrame round-trip renormalises quoting across every
    untouched row, which would make "did anything else change?" unanswerable.

With those three, csv.reader -> csv.writer is byte-identical on all three sheets, so
an in-place edit can be proven to have touched nothing but the cells it meant to.
"""

import csv
from pathlib import Path

from gold_standard.sheets import COL_CASE_ID, VALID_LABELS, Sheet

ENCODING = "cp1252"
LINE_TERMINATOR = "\r\n"

# Report fields run to tens of thousands of characters; the default 128 KB limit is
# ample, but the sheets are third-party data and a future one may not be.
csv.field_size_limit(10_000_000)


def read_sheet(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, data rows) exactly as stored."""
    with open(path, newline="", encoding=ENCODING) as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0], rows[1:]


def write_sheet(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write header and rows back in the sheets' own dialect."""
    with open(path, "w", newline="", encoding=ENCODING) as handle:
        writer = csv.writer(handle, lineterminator=LINE_TERMINATOR)
        writer.writerow(header)
        writer.writerows(rows)


def column_index(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError as exc:
        raise ValueError(f"Column {name!r} is not in this sheet") from exc


def apply_labels(path: Path, sheet: Sheet, labels: dict[str, dict[str, str]]) -> int:
    """Write labels for the given cases into the sheet in place.

    `labels` maps CaseID to {column: label}. Returns the number of rows written.
    Raises rather than writing anything if a label is outside the vocabulary or a
    CaseID is not in the sheet, so a typo can never land as a silent bad cell.
    """
    header, rows = read_sheet(path)
    case_col = column_index(header, COL_CASE_ID)
    indices = {c: column_index(header, c) for c in sheet.label_columns}

    for case_id, row_labels in labels.items():
        for column, value in row_labels.items():
            if column not in indices:
                raise ValueError(f"{column!r} is not a label column of {sheet.name}")
            if value not in VALID_LABELS:
                raise ValueError(
                    f"{value!r} is not a valid label for {case_id}/{column}; "
                    f"expected one of {sorted(VALID_LABELS)}"
                )

    by_case: dict[str, list[list[str]]] = {}
    for row in rows:
        by_case.setdefault(row[case_col], []).append(row)

    missing = [c for c in labels if c not in by_case]
    if missing:
        raise ValueError(f"CaseIDs not present in {sheet.name}: {missing}")

    written = 0
    for case_id, row_labels in labels.items():
        for row in by_case[case_id]:  # a duplicate CaseID gets the same labels
            for column, value in row_labels.items():
                row[indices[column]] = value
            written += 1

    write_sheet(path, header, rows)
    return written


def verify_untouched(path: Path, backup_path: Path, sheet: Sheet) -> list[str]:
    """Compare every non-label cell against the backup. Empty list means clean."""
    header, rows = read_sheet(path)
    backup_header, backup_rows = read_sheet(backup_path)

    problems: list[str] = []
    if header != backup_header:
        problems.append("header differs from backup")
        return problems
    if len(rows) != len(backup_rows):
        problems.append(f"row count {len(rows)} != backup {len(backup_rows)}")
        return problems

    label_indices = {column_index(header, c) for c in sheet.label_columns}
    for line, (row, backup_row) in enumerate(zip(rows, backup_rows), start=2):
        for index, (value, backup_value) in enumerate(zip(row, backup_row)):
            if index in label_indices:
                continue
            if value != backup_value:
                problems.append(f"line {line}: column {header[index]!r} changed")
    return problems
