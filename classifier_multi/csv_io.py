"""Reading cases from a category's CSV and writing labelled copies of it.

The input CSV is only ever read. Each category/provider pair gets its own output
copy, so a failed run can never damage the source data.
"""

from pathlib import Path

import pandas as pd

from classifier_multi.categories import Category
from classifier_multi.schemas import (
    ABNORMAL,
    COL_CASE_ID,
    COL_CONCLUSIONS,
    COL_FINDINGS,
    NORMAL,
    Label,
    RadiologyCase,
)


def read_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read the CSV as plain text.

    dtype=str and keep_default_na=False stop pandas turning empty cells into NaN and
    numeric-looking IDs into floats, so the text columns round-trip unchanged.
    """
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def verify_columns(df: pd.DataFrame, category: Category) -> None:
    """Fail loudly if the CSV headers and the category definition have drifted apart.

    Without this, renaming a column in the spreadsheet would silently produce a file
    where that finding is never labelled.
    """
    missing = [c for c in category.label_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{category.name}: CSV is missing expected label columns: {missing}"
        )

    for col in (COL_CASE_ID, COL_FINDINGS, COL_CONCLUSIONS):
        if col not in df.columns:
            raise ValueError(f"{category.name}: CSV is missing required column: {col!r}")

    unknown_derived = [
        f"{derived} <- {source}"
        for derived, sources in category.derived.items()
        for source in sources
        if source not in category.label_columns
    ]
    if unknown_derived:
        raise ValueError(
            f"{category.name}: derived column draws on columns that do not exist: "
            f"{unknown_derived}"
        )


def read_cases(csv_path: Path, category: Category) -> list[RadiologyCase]:
    """Load the cases, reading only the original radiologist's text."""
    df = read_dataframe(csv_path)
    verify_columns(df, category)
    return [
        RadiologyCase(
            case_id=str(row[COL_CASE_ID]).strip(),
            findings_text=row[COL_FINDINGS],
            conclusions_text=row[COL_CONCLUSIONS],
        )
        for _, row in df.iterrows()
    ]


def derive_column(sources: tuple[str, ...], labels: dict[str, Label]) -> Label:
    """A summary column is abnormal when any finding rolling into it is abnormal.

    Derived here rather than asked of the model, so it can never contradict the
    findings it is built from.
    """
    return ABNORMAL if any(labels.get(s) == ABNORMAL for s in sources) else NORMAL


def build_label_row(category: Category, labels: dict[str, Label]) -> dict[str, Label]:
    """Fill every label column: the ones judged, plus any derived from them.

    Any finding the model failed to return falls back to 'normal', matching the rule
    that silence in the report means normal.
    """
    row: dict[str, Label] = {
        name: labels.get(name, NORMAL) for name in category.asked_findings
    }
    for derived, sources in category.derived.items():
        row[derived] = derive_column(sources, row)
    return row


def write_labeled_csv(
    input_csv: Path,
    output_csv: Path,
    category: Category,
    labels_by_case: dict[str, dict[str, Label]],
) -> pd.DataFrame:
    """Write a copy of the input CSV with the label columns filled in.

    Cases missing from labels_by_case keep their empty cells, so a partial run
    produces a partial file rather than a wrong one.
    """
    df = read_dataframe(input_csv)
    verify_columns(df, category)

    for column in category.label_columns:
        df[column] = [
            labels_by_case.get(str(case_id).strip(), {}).get(column, "")
            for case_id in df[COL_CASE_ID]
        ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8", lineterminator="\n")
    return df
