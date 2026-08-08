"""The enum, the CSV headers and the derived column must agree.

This is the drift guard: renaming a column in the spreadsheet without updating
the enum would otherwise produce a file where that finding is silently never
labelled.
"""

from classifier import config
from classifier.csv_io import read_dataframe, verify_columns
from classifier.schemas import (
    DISEASED_LUNGS,
    DISEASED_LUNGS_INPUTS,
    LABEL_COLUMNS,
    FindingName,
)


def test_enum_has_19_findings():
    assert len(FindingName) == 19


def test_diseased_lungs_is_not_asked_of_the_model():
    assert DISEASED_LUNGS not in {f.value for f in FindingName}


def test_label_columns_are_enum_plus_derived():
    assert set(LABEL_COLUMNS) == {f.value for f in FindingName} | {DISEASED_LUNGS}
    assert len(LABEL_COLUMNS) == 20


def test_diseased_lungs_inputs_are_real_findings():
    assert len(DISEASED_LUNGS_INPUTS) == 10
    for finding in DISEASED_LUNGS_INPUTS:
        assert finding in FindingName


def test_real_csv_matches_the_enum():
    """Runs against the actual dataset, so a rename there fails the suite."""
    df = read_dataframe(config.INPUT_CSV)
    verify_columns(df)
    assert list(df.columns[10:30]) == list(LABEL_COLUMNS)
