"""Discrimination and calibration statistics.

Pure functions over plain sequences. Plotting and file writing live in
uq_analyze_main.py, so everything here is testable with a list literal.
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
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


def bin_indices(confidence: Sequence[float], max_bins: int = 10) -> tuple[np.ndarray, int]:
    """Assign each observation to a bin, adapting to how many distinct values exist.

    The authors bin with pd.qcut(confidence, 10). Our SC confidence at N=5 takes exactly
    three values, and qcut cannot produce ten unique edges from three values - it raises
    ValueError: Bin edges must be unique. So: bin by distinct value when there are few,
    and fall back to quantile bins only when there are enough to support them.

    Bin indices are ordered by confidence, so bin 0 is always the least confident.
    """
    values = np.asarray(confidence, dtype=float)
    distinct = np.unique(values)

    if len(distinct) <= max_bins:
        return np.searchsorted(distinct, values), len(distinct)

    ranks = np.asarray(pd.qcut(values, max_bins, labels=False, duplicates="drop"), dtype=int)
    return ranks, int(ranks.max()) + 1


def expected_calibration_error(
    confidence: Sequence[float],
    correct: Sequence[int],
    max_bins: int = 10,
) -> float:
    """Weighted mean gap between stated confidence and observed accuracy.

        ECE = sum_b (n_b / N) * |mean_confidence_b - accuracy_b|
    """
    values = np.asarray(confidence, dtype=float)
    outcomes = np.asarray(correct, dtype=float)
    indices, n_bins = bin_indices(values, max_bins)

    total = len(values)
    error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        if not mask.any():
            continue
        weight = mask.sum() / total
        error += weight * abs(values[mask].mean() - outcomes[mask].mean())

    return float(error)


def brier_score(confidence: Sequence[float], correct: Sequence[int]) -> float:
    """Mean squared distance between stated confidence and the outcome.

    No binning, so unlike ECE this is well defined however few distinct values there are.
    """
    values = np.asarray(confidence, dtype=float)
    outcomes = np.asarray(correct, dtype=float)
    return float(np.mean((values - outcomes) ** 2))


def calibration_points(
    confidence: Sequence[float],
    correct: Sequence[int],
    max_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """(mean confidence, observed accuracy, count) per bin - the calibration plot's data."""
    values = np.asarray(confidence, dtype=float)
    outcomes = np.asarray(correct, dtype=float)
    indices, n_bins = bin_indices(values, max_bins)

    points: list[tuple[float, float, int]] = []
    for bin_index in range(n_bins):
        mask = indices == bin_index
        if not mask.any():
            continue
        points.append((float(values[mask].mean()), float(outcomes[mask].mean()), int(mask.sum())))

    return points
