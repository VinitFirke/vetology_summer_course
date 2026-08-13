"""Scoring predictions against the manually scored gold standard.

Everything here is a pure function over label strings, so the whole module tests
without touching a file or an API.

Definitions, per condition:
    True Positive   gold Abnormal, prediction Abnormal
    False Negative  gold Abnormal, prediction Normal
    True Negative   gold Normal,   prediction Normal
    False Positive  gold Normal,   prediction Abnormal

    Sensitivity = TP / (TP + FN)      Specificity = TN / (TN + FP)

Both are reported with a 95% Wilson score interval. Three things about those intervals
are worth knowing before quoting them:

1. They are per condition, and that is what makes them valid. Within one condition each
   of the 50 cases contributes exactly one observation, so the binomial independence
   assumption holds. It would not hold for a single pooled interval across all 20
   conditions, because the same 50 reports are read for every one of them;
   clustered_bootstrap_ci in uncertainty/stats.py records measuring that error at about
   four times too narrow. Pooling was considered and declined, not overlooked.

2. They describe agreement with the gold standard, not agreement with truth. The gold
   standard is one manual scoring. Published comparisons report an inter-radiologist
   agreement rate alongside their metrics precisely because rater disagreement is a
   separate source of error that no binomial interval on these counts can capture.

3. A wide interval is the finding, not a defect. At 50 cases with one to five positives
   for most conditions, a sensitivity of 1.000 spanning 0.21 to 1.00 is the honest
   description of what one case can tell you.
"""

import math
from statistics import NormalDist

from pydantic import BaseModel

# Written when a metric has no denominator - undefined, which is not the same as zero.
UNDEFINED = "N/A"

# Column order is taken verbatim from the example workbook's "Confusion Matrix" sheet,
# with each interval placed next to the estimate it qualifies. The "95%" in these headers
# is a literal, so it is only true while wilson_interval's default alpha stays 0.05;
# a test holds that default in place.
MATRIX_COLUMNS: tuple[str, ...] = (
    "condition",
    "True Positive",
    "False Negative",
    "True Negative",
    "False Positive",
    "Sensitivity",
    "Sensitivity 95% CI low",
    "Sensitivity 95% CI high",
    "Specificity",
    "Specificity 95% CI low",
    "Specificity 95% CI high",
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


def wilson_interval(
    successes: int, trials: int, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion. None when there are no trials.

    Wilson rather than the textbook Wald interval, because almost every condition here
    sits on a boundary with a tiny denominator, which is exactly where Wald breaks. At
    successes == trials Wald gives p +- z*sqrt(p(1-p)/n) = 1.000 +- 0: a zero-width
    interval asserting certainty from what may be a single case. That is the failure
    these intervals exist to prevent. Wald also strays below 0 and above 1 near the
    boundaries; Wilson cannot.

    The z quantile comes from the standard library rather than scipy, so this module
    keeps its one dependency and stays a set of pure functions. NormalDist agrees with
    scipy.stats.norm.ppf to thirteen decimal places.

    The max/min clamps are float-safety only. Algebraically Wilson already returns
    exactly 1.0 when successes == trials and exactly 0.0 when successes == 0.
    """
    if trials <= 0:
        return None

    z = NormalDist().inv_cdf(1 - alpha / 2)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def sensitivity_interval(counts: Counts) -> tuple[float, float] | None:
    """Interval around TP / (TP + FN), or None when the gold standard holds no positives."""
    return wilson_interval(counts.true_positive, counts.positive_ground_truth)


def specificity_interval(counts: Counts) -> tuple[float, float] | None:
    """Interval around TN / (TN + FP), or None when the gold standard holds no negatives."""
    return wilson_interval(counts.true_negative, counts.negative_ground_truth)


def _bounds(interval: tuple[float, float] | None) -> tuple[object, object]:
    """Split an interval into two cells, writing N/A into both when it is undefined.

    Keeps the bounds consistent with the point estimate: a condition whose sensitivity
    reads N/A must not show numeric bounds beside it.
    """
    if interval is None:
        return UNDEFINED, UNDEFINED
    return interval


def matrix_row(condition: str, counts: Counts) -> dict[str, object]:
    """One row of the output sheet, in the example workbook's column order."""
    sensitivity_low, sensitivity_high = _bounds(sensitivity_interval(counts))
    specificity_low, specificity_high = _bounds(specificity_interval(counts))
    return {
        "condition": condition,
        "True Positive": counts.true_positive,
        "False Negative": counts.false_negative,
        "True Negative": counts.true_negative,
        "False Positive": counts.false_positive,
        "Sensitivity": sensitivity(counts),
        "Sensitivity 95% CI low": sensitivity_low,
        "Sensitivity 95% CI high": sensitivity_high,
        "Specificity": specificity(counts),
        "Specificity 95% CI low": specificity_low,
        "Specificity 95% CI high": specificity_high,
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
