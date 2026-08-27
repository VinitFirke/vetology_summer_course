"""Scoring predictions against the manually scored gold standard.

Everything here is a pure function over label strings, so the whole module tests
without touching a file or an API.

Definitions, per condition:
    True Positive   gold Abnormal, prediction Abnormal
    False Negative  gold Abnormal, prediction Normal
    True Negative   gold Normal,   prediction Normal
    False Positive  gold Normal,   prediction Abnormal

    Sensitivity = TP / (TP + FN)      Specificity = TN / (TN + FP)

One thing differs from the single-category scorer: the gold standard does not always
spell the positive class the way the models do. Canine VHS is scored normal/enlarged
by the manual scorers while every model returns normal/abnormal, so gold values are
put through the alias map defined in this module. The map lives here rather than on
Category because it describes the gold standard, and nothing outside this module is
allowed to know the gold standard exists.
"""

from pathlib import Path

from pydantic import BaseModel

from classifier_multi.categories import Category
from classifier_multi.config import DATA_DIR, PROJECT_ROOT, Variant

# This module is the only place in the package that knows the gold standard exists.
# Every other module is checked by tests/test_gold_boundary.py for even a mention of
# it, so a classification run cannot reach the answers by accident.
GOLD_DIR = PROJECT_ROOT / "dataset_gold_standard"

# Off limits everywhere, including here: the untouched source spreadsheets.
FORBIDDEN_DIR = "_originals"

GOLD_CSV_NAMES: dict[str, str] = {
    "feline_thorax": "feline_thorax_gold_standard.csv",
    "canine_thorax": "canine_thorax_gold_standard.csv",
    "canine_abdomen": "canine_abdomen_gold_standard.csv",
}

# Vocabulary the manual scorers used that the models are never asked to produce.
# Canine VHS is scored normal/enlarged there; every model returns normal/abnormal.
GOLD_VALUE_ALIASES: dict[str, dict[str, dict[str, str]]] = {
    "canine_thorax": {"vhs": {"enlarged": "abnormal"}},
}


class OriginalsAccessError(RuntimeError):
    """Raised on any attempt to reach dataset_gold_standard/_originals."""


def reject_originals(path: Path) -> None:
    """Refuse a path that reaches into the untouched source spreadsheets.

    Checked rather than merely never written, so a future edit that constructs such a
    path fails immediately instead of quietly reading files that are not ours to use.
    """
    if FORBIDDEN_DIR in Path(path).parts:
        raise OriginalsAccessError(
            f"{FORBIDDEN_DIR} is off limits and must never be read: {path}"
        )


def gold_csv_path(category: Category) -> Path:
    path = GOLD_DIR / GOLD_CSV_NAMES[category.name]
    reject_originals(path)
    return path


def gold_value_aliases(category: Category) -> dict[str, dict[str, str]]:
    """Per-condition value maps for this category, empty when none are needed."""
    return GOLD_VALUE_ALIASES.get(category.name, {})


def confusion_matrix_path(category: Category, variant: Variant) -> Path:
    """Written beside the predictions it scores, so the gold folder stays read-only."""
    return DATA_DIR / variant / f"confusion_matrix_{category.name}.xlsx"

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


def normalise(value: str, aliases: dict[str, str] | None = None) -> str:
    """Fold a raw cell to 'abnormal', 'normal', or '' for anything unrecognised.

    `aliases` maps this condition's extra vocabulary onto the two labels - for the
    canine VHS column, {'enlarged': 'abnormal'}. Matching is case-insensitive, so a
    scorer writing "Enlarged" counts the same as "enlarged".
    """
    text = value.strip().casefold()
    if aliases:
        text = aliases.get(text, text)
    return text if text in ("normal", "abnormal") else ""


def confusion_counts(
    pairs: list[tuple[str, str]], aliases: dict[str, str] | None = None
) -> Counts:
    """Count one condition across cases, from (gold_label, predicted_label) pairs.

    Pairs where either side is blank or unrecognised are skipped, so an unfinished run
    scores only the cases it actually completed.
    """
    counts = Counts()
    for raw_gold, raw_predicted in pairs:
        gold = normalise(raw_gold, aliases)
        predicted = normalise(raw_predicted, aliases)
        if not gold or not predicted:
            continue
        if predicted == "abnormal":
            if gold == "abnormal":
                counts.true_positive += 1
            else:
                counts.false_positive += 1
        else:
            if gold == "abnormal":
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
    value_aliases: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Build one row per condition, joining gold and predictions on CaseID."""
    value_aliases = value_aliases or {}
    rows = []
    for condition in conditions:
        pairs = [
            (gold_labels.get(condition, ""), pred_by_case[case_id].get(condition, ""))
            for case_id, gold_labels in gold_by_case.items()
            if case_id in pred_by_case
        ]
        rows.append(
            matrix_row(condition, confusion_counts(pairs, value_aliases.get(condition)))
        )
    return rows
