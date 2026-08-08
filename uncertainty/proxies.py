"""Turning raw replicate labels into confidence numbers.

Everything here is a pure function over label strings, so the whole module tests
without touching a file or an API - the same property that makes
classifier/evaluate.py cheap to work on.
"""

from collections import Counter

from classifier.schemas import Label


def sample_consistency(labels: list[Label]) -> tuple[Label, float]:
    """The majority label and the fraction of replicates that agreed with it.

    This is the whole SC proxy. The paper needed a GPT-4 annotator and two
    sentence-transformer models here, but only because their answers were free text and
    "do these two strings mean the same thing?" was a hard question. With a closed
    label set it is `==`.
    """
    counts = Counter(labels)
    answer, agreed = counts.most_common(1)[0]
    return answer, agreed / len(labels)


def confidence_elicitation(score: int) -> float:
    """The model's self-rated 0-100 certainty, on the unit interval.

    The paper's scale runs 0 = definitely uncertain to 100 = definitely certain, so
    higher already means more confident and no inversion is applied.
    """
    return score / 100.0
