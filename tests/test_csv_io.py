"""Reading cases and writing labelled copies, without touching the input file."""

import pandas as pd
import pytest

from classifier.csv_io import read_cases, read_dataframe, verify_columns, write_labeled_csv
from classifier.schemas import (
    ABNORMAL,
    COL_CASE_ID,
    COL_CONCLUSIONS,
    COL_FINDINGS,
    DISEASED_LUNGS,
    LABEL_COLUMNS,
    NORMAL,
    FindingName,
)

TEXT_COLUMNS = [
    COL_CASE_ID,
    "Link to AI report",
    "Link to Rad report",
    COL_FINDINGS,
    COL_CONCLUSIONS,
    "Recommendations (original radiologist report)",
    "Original Radiologist",
    "Findings (AI report)",
    "Conclusions (AI report)",
    "Recommendations (AI report)",
]


@pytest.fixture
def sample_csv(tmp_path):
    """A two-row stand-in with the same shape as the real dataset."""
    rows = []
    for case_id in ("1001", "1002"):
        row = {col: f"{col}-{case_id}" for col in TEXT_COLUMNS}
        row[COL_CASE_ID] = case_id
        row.update({col: "" for col in LABEL_COLUMNS})
        rows.append(row)

    path = tmp_path / "sample(in).csv"
    pd.DataFrame(rows, columns=TEXT_COLUMNS + list(LABEL_COLUMNS)).to_csv(path, index=False)
    return path


def test_read_cases_reads_only_radiologist_text(sample_csv):
    cases = read_cases(sample_csv)
    assert [c.case_id for c in cases] == ["1001", "1002"]
    assert cases[0].findings_text == f"{COL_FINDINGS}-1001"
    assert cases[0].conclusions_text == f"{COL_CONCLUSIONS}-1001"


def test_write_leaves_the_input_file_untouched(sample_csv, tmp_path):
    before = sample_csv.read_bytes()
    write_labeled_csv(
        sample_csv,
        tmp_path / "out.csv",
        {"1001": {col: NORMAL for col in LABEL_COLUMNS}},
    )
    assert sample_csv.read_bytes() == before


def test_labels_land_in_the_right_columns(sample_csv, tmp_path):
    labels = {col: NORMAL for col in LABEL_COLUMNS}
    labels[FindingName.cardiomegaly.value] = ABNORMAL

    out = tmp_path / "out.csv"
    write_labeled_csv(sample_csv, out, {"1001": labels, "1002": labels})
    written = read_dataframe(out)

    assert list(written.columns) == TEXT_COLUMNS + list(LABEL_COLUMNS)
    assert written.loc[0, FindingName.cardiomegaly.value] == ABNORMAL
    assert written.loc[0, DISEASED_LUNGS] == NORMAL
    assert written.loc[0, COL_FINDINGS] == f"{COL_FINDINGS}-1001"


def test_unlabelled_cases_stay_empty(sample_csv, tmp_path):
    """A partial run must produce a partial file, never a wrongly-filled one."""
    out = tmp_path / "out.csv"
    write_labeled_csv(sample_csv, out, {"1001": {col: NORMAL for col in LABEL_COLUMNS}})
    written = read_dataframe(out)

    assert written.loc[0, FindingName.bronchitis.value] == NORMAL
    assert written.loc[1, FindingName.bronchitis.value] == ""


def test_verify_columns_rejects_a_renamed_column(sample_csv):
    df = read_dataframe(sample_csv).rename(columns={FindingName.bronchitis.value: "bronchitis_v2"})
    with pytest.raises(ValueError, match="missing expected label columns"):
        verify_columns(df)
