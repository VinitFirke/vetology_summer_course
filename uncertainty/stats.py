"""Discrimination and calibration statistics.

Pure functions over plain sequences. Plotting and file writing live in
uq_analyze_main.py, so everything here is testable with a list literal.
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

# The paper's threshold for clinically useful discriminative ability.
USEFUL_AUC = 0.7


def roc_auc(confidence: Sequence[float], correct: Sequence[int]) -> float | None:
    """Area under the ROC curve, or None when it is undefined.

    sklearn raises if `correct` holds only one class. That happens for real - a tier
    where the model got everything right has no incorrect answers to separate - and it
    must be reported as N/A rather than take down the whole report.
    """
    if len(set(correct)) < 2:
        return None
    return float(roc_auc_score(list(correct), list(confidence)))


def clustered_bootstrap_ci(
    confidence: Sequence[float],
    correct: Sequence[int],
    case_ids: Sequence[str],
    iterations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the ROC AUC, resampling whole cases rather than rows.

    The 19 findings of one case are read off one shared report, so both correctness and
    confidence are correlated within a case. Resampling rows would pretend there are 950
    independent observations where there are 50 clusters; measured on synthetic data with
    that structure, it reports an interval about four times too narrow.

    The authors' notebook takes the 5th and 95th percentiles and prints the result as a
    "95% Confidence Interval"; that is a 90% interval. alpha=0.05 gives a real one.

    Returns (nan, nan) when no bootstrap draw had both classes present.
    """
    by_case: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for value, outcome, case_id in zip(confidence, correct, case_ids):
        by_case[case_id].append((value, outcome))

    cases = list(by_case)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []

    for _ in range(iterations):
        drawn = rng.integers(0, len(cases), size=len(cases))
        sampled_confidence: list[float] = []
        sampled_correct: list[int] = []
        for index in drawn:
            for value, outcome in by_case[cases[index]]:
                sampled_confidence.append(value)
                sampled_correct.append(outcome)

        auc = roc_auc(sampled_confidence, sampled_correct)
        if auc is not None:
            aucs.append(auc)

    if not aucs:
        return (float("nan"), float("nan"))

    return (
        float(np.percentile(aucs, 100 * alpha / 2)),
        float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    )
