"""Reading cases from the CSV and writing labelled copies of it.

The input CSV is only ever read. Each provider gets its own output copy, so a
failed run can never damage the source data.
"""

from pathlib import Path

import pandas as pd

from classifier.schemas import (
    ABNORMAL,
    COL_CASE_ID,
    COL_CONCLUSIONS,
    COL_FINDINGS,
    DISEASED_LUNGS,
    DISEASED_LUNGS_INPUTS,
    LABEL_COLUMNS,
    NORMAL,
    FindingName,
    Label,
    RadiologyCase,
)


def read_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read the CSV as plain text.

    dtype=str and keep_default_na=False stop pandas turning empty cells into NaN
    and numeric-looking IDs into floats, so the text columns round-trip unchanged.
    """
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def verify_columns(df: pd.DataFrame) -> None:
    """Fail loudly if the CSV headers and the FindingName enum have drifted apart.

    Without this, renaming a column in the spreadsheet would silently produce a
    file where that finding is never labelled.
    """
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected label columns: {missing}")

    for col in (COL_CASE_ID, COL_FINDINGS, COL_CONCLUSIONS):
        if col not in df.columns:
            raise ValueError(f"CSV is missing required text column: {col!r}")

    enum_names = {f.value for f in FindingName}
    csv_label_names = set(LABEL_COLUMNS) - {DISEASED_LUNGS}
    if enum_names != csv_label_names:
        raise ValueError(
            "FindingName enum does not match the CSV label columns.\n"
            f"  only in enum: {sorted(enum_names - csv_label_names)}\n"
            f"  only in CSV:  {sorted(csv_label_names - enum_names)}"
        )


def read_cases(csv_path: Path) -> list[RadiologyCase]:
    """Load the 50 cases, reading only the original radiologist's text."""
    df = read_dataframe(csv_path)
    verify_columns(df)
    return [
        RadiologyCase(
            case_id=row[COL_CASE_ID],
            findings_text=row[COL_FINDINGS],
            conclusions_text=row[COL_CONCLUSIONS],
        )
        for _, row in df.iterrows()
    ]


def derive_diseased_lungs(labels: dict[str, Label]) -> Label:
    """diseased_lungs is abnormal when any lung parenchyma or lower-airway finding is.

    Derived here rather than asked of the model, so it can never contradict the
    findings it is built from.
    """
    for finding in DISEASED_LUNGS_INPUTS:
        if labels.get(finding.value) == ABNORMAL:
            return ABNORMAL
    return NORMAL


def build_label_row(labels: dict[str, Label]) -> dict[str, Label]:
    """Fill all 20 label columns: the 19 judged, plus the derived one.

    Any finding the model failed to return falls back to 'normal', matching the
    rule that silence means normal.
    """
    row: dict[str, Label] = {
        finding.value: labels.get(finding.value, NORMAL) for finding in FindingName
    }
    row[DISEASED_LUNGS] = derive_diseased_lungs(row)
    return row


def write_labeled_csv(
    input_csv: Path,
    output_csv: Path,
    labels_by_case: dict[str, dict[str, Label]],
) -> pd.DataFrame:
    """Write a copy of the input CSV with the 20 label columns filled in.

    Cases missing from labels_by_case keep their empty cells, so a partial run
    produces a partial file rather than a wrong one.
    """
    df = read_dataframe(input_csv)
    verify_columns(df)

    for column in LABEL_COLUMNS:
        df[column] = [
            labels_by_case.get(case_id, {}).get(column, "")
            for case_id in df[COL_CASE_ID]
        ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8", lineterminator="\n")
    return df
