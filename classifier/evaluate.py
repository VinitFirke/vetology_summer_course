"""Scoring predictions against the manually scored gold standard.

Everything here is a pure function over label strings, so the whole module tests
without touching a file or an API.

Definitions, per condition:
    True Positive   gold Abnormal, prediction Abnormal
    False Negative  gold Abnormal, prediction Normal
    True Negative   gold Normal,   prediction Normal
    False Positive  gold Normal,   prediction Abnormal

    Sensitivity = TP / (TP + FN)      Specificity = TN / (TN + FP)
"""

from pydantic import BaseModel

# Written when a metric has no denominator - undefined, which is not the same as zero.
UNDEFINED = "N/A"

# Column order is taken verbatim from the example workbook's "Confusion Matrix" sheet.
MATRIX_COLUMNS: tuple[str, ...] = (
    "condition",
    "True Positive",
    "False Negative",
    "True Negative",
    "False Positive",
    "Sensitivity",
    "Specificity",
    "Check",
    "Positive Ground Truth",
    "Negative Ground Truth",
    "Ground Truth Check",
)


class Counts(BaseModel):
    """The four cells of one condition's confusion matrix."""

    true_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    false_positive: int = 0

    @property
    def positive_ground_truth(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negative_ground_truth(self) -> int:
        return self.true_negative + self.false_positive

    @property
    def total(self) -> int:
        return self.positive_ground_truth + self.negative_ground_truth


def is_abnormal(value: str) -> bool:
    """Gold writes 'Abnormal', the models write 'abnormal'."""
    return value.strip().casefold() == "abnormal"


def is_normal(value: str) -> bool:
    return value.strip().casefold() == "normal"


def confusion_counts(pairs: list[tuple[str, str]]) -> Counts:
    """Count one condition across cases, from (gold_label, predicted_label) pairs.

    Pairs where either side is blank or unrecognised are skipped, so an unfinished
    run scores only the cases it actually completed.
    """
    counts = Counts()
    for gold, predicted in pairs:
        if not (is_abnormal(gold) or is_normal(gold)):
            continue
        if is_abnormal(predicted):
            if is_abnormal(gold):
                counts.true_positive += 1
            else:
                counts.false_positive += 1
        elif is_normal(predicted):
            if is_abnormal(gold):
                counts.false_negative += 1
            else:
                counts.true_negative += 1
    return counts


def sensitivity(counts: Counts) -> float | str:
    """TP / (TP + FN), or N/A when the gold standard holds no positives to find."""
    if counts.positive_ground_truth == 0:
        return UNDEFINED
    return counts.true_positive / counts.positive_ground_truth


def specificity(counts: Counts) -> float | str:
    """TN / (TN + FP), or N/A when the gold standard holds no negatives."""
    if counts.negative_ground_truth == 0:
        return UNDEFINED
    return counts.true_negative / counts.negative_ground_truth


def matrix_row(condition: str, counts: Counts) -> dict[str, object]:
    """One row of the output sheet, in the example workbook's column order."""
    return {
        "condition": condition,
        "True Positive": counts.true_positive,
        "False Negative": counts.false_negative,
        "True Negative": counts.true_negative,
        "False Positive": counts.false_positive,
        "Sensitivity": sensitivity(counts),
        "Specificity": specificity(counts),
        "Check": counts.total,
        "Positive Ground Truth": counts.positive_ground_truth,
        "Negative Ground Truth": counts.negative_ground_truth,
        "Ground Truth Check": counts.positive_ground_truth + counts.negative_ground_truth,
    }


def build_matrix_rows(
    gold_by_case: dict[str, dict[str, str]],
    pred_by_case: dict[str, dict[str, str]],
    conditions: list[str],
    gold_to_prediction: dict[str, str],
) -> list[dict[str, object]]:
    """Build one row per condition, joining gold and predictions on CaseID.

    `conditions` are the gold column names; gold_to_prediction maps any that were
    renamed in the prediction files.
    """
    rows = []
    for condition in conditions:
        prediction_column = gold_to_prediction.get(condition, condition)
        pairs = [
            (gold_labels.get(condition, ""), pred_by_case[case_id].get(prediction_column, ""))
            for case_id, gold_labels in gold_by_case.items()
            if case_id in pred_by_case
        ]
        rows.append(matrix_row(condition, confusion_counts(pairs)))
    return rows
