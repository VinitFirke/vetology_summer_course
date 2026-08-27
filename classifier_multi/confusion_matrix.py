"""Confusion matrices for the few-shot predictions, scored against the gold standard.

One matrix per (category, model). Each row is one condition:

    True Positive   gold abnormal, prediction abnormal
    False Negative  gold abnormal, prediction normal
    True Negative   gold normal,   prediction normal
    False Positive  gold normal,   prediction abnormal

    Sensitivity = TP / (TP + FN)      Specificity = TN / (TN + FP)

Only case IDs present on BOTH sides are scored. The gold standard holds all 200 cases
while a few-shot run covers the first 100, so the unscored tail of a predictions file
is dropped by the join rather than counted as anything.

Strictness is the point of this module, and it is where it differs from evaluate.py:
once a case is in the join, every one of its label cells must read 'normal' or
'abnormal'. Anything else - a blank, a typo, a stray 'enlarged' - raises ValueError
instead of being skipped, so a half-parsed run can never quietly produce a matrix that
looks complete.

Columns and the two-decimal percentage format are taken verbatim from
'Example Confusion Matrix Output 2(Confusion Matrix).csv'.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "dataset_LLM_classification" / "fewshot"
GOLD_DIR = PROJECT_ROOT / "dataset_gold_standard"

CATEGORIES: tuple[str, ...] = ("canine_abdomen", "canine_thorax", "feline_thorax")
MODELS: tuple[str, ...] = ("gemma", "kimi", "nemotron", "qwen")

CASE_ID = "CaseID"
POSITIVE = "abnormal"
NEGATIVE = "normal"
LABELS = (NEGATIVE, POSITIVE)

# Written when a metric has no denominator: undefined, which is not the same as zero.
UNDEFINED = "N/A"

# Report text and bookkeeping. Everything else in a row is a condition.
METADATA_COLUMNS = frozenset(
    {
        CASE_ID,
        "Link to AI report",
        "Link to Rad report",
        "Findings (original radiologist report)",
        "Conclusions (original radiologist report)",
        "Recommendations (original radiologist report)",
        "Original Radiologist",
        "Findings (AI report)",
        "Conclusions (AI report)",
        "Recommendations (AI report)",
        "",
    }
)

# Two of the gold CSVs end in a trailing comma, which pandas reads as a nameless
# column called "Unnamed: 30". It carries no labels and is not a condition.
UNNAMED_PREFIX = "Unnamed:"

MATRIX_COLUMNS: tuple[str, ...] = (
    "condition",
    "True Positive",
    "False Negative",
    "True Negative",
    "False Positive",
    "Sensitivity",
    "Specificity",
)


def gold_path(category: str) -> Path:
    return GOLD_DIR / f"{category}_gold_standard.csv"


def predictions_path(category: str, model: str) -> Path:
    return PREDICTIONS_DIR / f"{category}_classified_{model}.csv"


def output_path(category: str, model: str) -> Path:
    return PREDICTIONS_DIR / f"confusion_matrix_output_{category}_{model}.csv"


def read_labels(path: Path) -> pd.DataFrame:
    """Read a scored CSV down to its condition columns, indexed by CaseID.

    Values are lowercased and stripped here so that casing never decides an outcome.
    Rows whose condition cells are all blank are dropped: that is an unscored case,
    not a bad one, and requirement 3 says only cases scored on both sides count.
    """
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if CASE_ID not in frame.columns:
        raise ValueError(f"{path.name} has no {CASE_ID} column")

    conditions = [
        c
        for c in frame.columns
        if c not in METADATA_COLUMNS and not c.startswith(UNNAMED_PREFIX)
    ]
    if not conditions:
        raise ValueError(f"{path.name} has no condition columns")

    labels = frame[conditions].apply(lambda col: col.str.strip().str.lower())
    labels.insert(0, CASE_ID, frame[CASE_ID].str.strip())

    duplicates = labels[CASE_ID][labels[CASE_ID].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{path.name} repeats case IDs: {sorted(set(duplicates))}")

    scored = labels[conditions].notna().any(axis=1) & labels[conditions].ne("").any(
        axis=1
    )
    return labels[scored].set_index(CASE_ID)


def matched_case_ids(gold: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    """Case IDs scored on both sides, in gold-standard order."""
    matched = [case_id for case_id in gold.index if case_id in predictions.index]
    if not matched:
        raise ValueError(
            "no case ID appears in both the gold standard and the predictions"
        )
    return matched


def check_labels(frame: pd.DataFrame, source: str) -> None:
    """Every cell must be 'normal' or 'abnormal'. Anything else is a hard error."""
    for condition in frame.columns:
        column = frame[condition]
        bad = column[~column.isin(LABELS)]
        if not bad.empty:
            case_id = bad.index[0]
            raise ValueError(
                f"{source}: invalid label {bad.iloc[0]!r} for condition "
                f"{condition!r} on case {case_id!r}; expected one of {LABELS}"
            )


def as_percentage(numerator: int, denominator: int) -> str:
    """Two-decimal percentage, or N/A when the class is absent from the gold standard."""
    if denominator == 0:
        return UNDEFINED
    return f"{numerator / denominator * 100:.2f}%"


def confusion_matrix(gold: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per condition, counted over the case IDs the two frames share.

    Conditions are taken from the gold standard; a condition the predictions do not
    carry is a mismatch worth failing on, not a row of zeroes.
    """
    case_ids = matched_case_ids(gold, predictions)

    missing = [c for c in gold.columns if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions are missing conditions: {missing}")

    gold_labels = gold.loc[case_ids, list(gold.columns)]
    predicted_labels = predictions.loc[case_ids, list(gold.columns)]
    check_labels(gold_labels, "gold standard")
    check_labels(predicted_labels, "predictions")

    rows = []
    for condition in gold.columns:
        actual = gold_labels[condition]
        predicted = predicted_labels[condition]
        true_positive = int(((actual == POSITIVE) & (predicted == POSITIVE)).sum())
        false_negative = int(((actual == POSITIVE) & (predicted == NEGATIVE)).sum())
        true_negative = int(((actual == NEGATIVE) & (predicted == NEGATIVE)).sum())
        false_positive = int(((actual == NEGATIVE) & (predicted == POSITIVE)).sum())
        rows.append(
            {
                "condition": condition,
                "True Positive": true_positive,
                "False Negative": false_negative,
                "True Negative": true_negative,
                "False Positive": false_positive,
                "Sensitivity": as_percentage(
                    true_positive, true_positive + false_negative
                ),
                "Specificity": as_percentage(
                    true_negative, true_negative + false_positive
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(MATRIX_COLUMNS))


def build_matrix(category: str, model: str) -> tuple[pd.DataFrame, int]:
    """Score one model on one category. Returns the matrix and the cases counted."""
    gold = read_labels(gold_path(category))
    predictions = read_labels(predictions_path(category, model))
    matrix = confusion_matrix(gold, predictions)
    return matrix, len(matched_case_ids(gold, predictions))


if __name__ == "__main__":
    for category in CATEGORIES:
        for model in MODELS:
            matrix, case_count = build_matrix(category, model)
            destination = output_path(category, model)
            matrix.to_csv(destination, index=False)
            print(f"\n=== {category} / {model} - {case_count} matched cases ===")
            print(matrix.to_string(index=False))
            print(f"wrote {destination}")
