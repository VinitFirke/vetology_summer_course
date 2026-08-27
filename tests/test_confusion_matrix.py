"""Confusion matrix counts, and the errors that must not pass silently."""

import pandas as pd
import pytest

from classifier_multi.confusion_matrix import confusion_matrix

CASE_ID = "CaseID"


def frame(rows: dict[str, list[str]]) -> pd.DataFrame:
    """A labels frame shaped the way read_labels returns one: CaseID as the index."""
    return pd.DataFrame(rows).set_index(CASE_ID)


def test_counts_and_rates_on_a_hand_worked_example():
    """Four cases, one condition, one of each cell.

    gold abnormal + predicted abnormal -> TP     Sensitivity = 1/2 = 50.00%
    gold abnormal + predicted normal   -> FN     Specificity = 1/2 = 50.00%
    gold normal   + predicted normal   -> TN
    gold normal   + predicted abnormal -> FP

    Case '5' is scored only in the gold standard, so the join must drop it.
    """
    gold = frame(
        {
            CASE_ID: ["1", "2", "3", "4", "5"],
            "ascites": ["abnormal", "abnormal", "normal", "normal", "abnormal"],
        }
    )
    predictions = frame(
        {
            CASE_ID: ["1", "2", "3", "4"],
            "ascites": ["abnormal", "normal", "normal", "abnormal"],
        }
    )

    matrix = confusion_matrix(gold, predictions)

    assert len(matrix) == 1
    row = matrix.iloc[0]
    assert row["condition"] == "ascites"
    assert row["True Positive"] == 1
    assert row["False Negative"] == 1
    assert row["True Negative"] == 1
    assert row["False Positive"] == 1
    assert row["Sensitivity"] == "50.00%"
    assert row["Specificity"] == "50.00%"


def test_invalid_label_raises():
    """A cell that is neither 'normal' nor 'abnormal' must stop the run."""
    gold = frame({CASE_ID: ["1", "2"], "ascites": ["abnormal", "normal"]})
    predictions = frame({CASE_ID: ["1", "2"], "ascites": ["abnormal", "enlarged"]})

    with pytest.raises(ValueError, match="invalid label 'enlarged'"):
        confusion_matrix(gold, predictions)


def test_no_matching_case_ids_raises():
    """Nothing in common means there is nothing to score, not an empty matrix."""
    gold = frame({CASE_ID: ["1", "2"], "ascites": ["abnormal", "normal"]})
    predictions = frame({CASE_ID: ["97", "98"], "ascites": ["abnormal", "normal"]})

    with pytest.raises(ValueError, match="no case ID appears in both"):
        confusion_matrix(gold, predictions)
