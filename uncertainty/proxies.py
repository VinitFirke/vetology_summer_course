"""Turning raw replicate labels into confidence numbers.

Everything here is a pure function over label strings, so the whole module tests
without touching a file or an API - the same property that makes
classifier/evaluate.py cheap to work on.
"""

from collections import Counter

from pydantic import BaseModel

from classifier.evaluate import is_abnormal, is_normal
from classifier.schemas import FindingName, Label
from uncertainty.config import MIN_REPLICATES


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


class ProxyRow(BaseModel):
    """One (case, finding, proxy) observation - the unit of the whole analysis."""

    provider: str
    tier: str
    case_id: str
    finding: str
    proxy: str
    confidence: float
    answer: str
    gold: str
    correct: int
    n_replicates: int


def matches(gold: str, predicted: str) -> bool:
    """Whether a prediction agrees with the gold label.

    Delegates to the same two helpers the confusion matrix uses, so the UQ study and
    evaluate_main.py can never disagree about what counts as a match.
    """
    if is_abnormal(gold):
        return is_abnormal(predicted)
    return is_normal(predicted)


def build_rows(
    samples: dict[str, list[dict[str, str]]],
    ce_scores: dict[str, dict[str, int]],
    gold: dict[str, dict[str, str]],
    provider: str,
    tier: str,
    min_replicates: int = MIN_REPLICATES,
) -> list[ProxyRow]:
    """Join replicates, CE scores and the gold standard into the long-format table.

    Iterating FindingName is what excludes `diseased_lungs`: it is derived in
    csv_io.derive_diseased_lungs() and is not a member of the enum, so the exclusion is
    structural rather than a filter someone can forget to apply.
    """
    rows: list[ProxyRow] = []

    for case_id, replicates in sorted(samples.items()):
        gold_labels = gold.get(case_id)
        if gold_labels is None or len(replicates) < min_replicates:
            continue

        for finding in FindingName:
            name = finding.value
            labels = [r[name] for r in replicates if name in r]
            if len(labels) < min_replicates:
                continue

            gold_label = gold_labels.get(name, "")
            if not (is_abnormal(gold_label) or is_normal(gold_label)):
                continue

            common = {
                "provider": provider,
                "tier": tier,
                "case_id": case_id,
                "finding": name,
                "gold": gold_label,
                "n_replicates": len(labels),
            }

            sc_answer, sc_confidence = sample_consistency(labels)
            rows.append(
                ProxyRow(
                    **common,
                    proxy="SC",
                    confidence=sc_confidence,
                    answer=sc_answer,
                    correct=int(matches(gold_label, sc_answer)),
                )
            )

            score = ce_scores.get(case_id, {}).get(name)
            if score is not None:
                # CE rates the SinglePass answer, which is replicate 1 - not the majority.
                ce_answer = labels[0]
                rows.append(
                    ProxyRow(
                        **common,
                        proxy="CE",
                        confidence=confidence_elicitation(score),
                        answer=ce_answer,
                        correct=int(matches(gold_label, ce_answer)),
                    )
                )

    return rows
