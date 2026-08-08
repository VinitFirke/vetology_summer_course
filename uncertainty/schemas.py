"""Slim data shapes used only by the uncertainty runs.

FindingName and Label are imported rather than redeclared, so the enum stays the single
source of truth and the two pipelines cannot drift apart in what they ask about.

Why a separate CaseLabels rather than making evidence/reasoning Optional on FindingLabel:
optional fields still appear in the JSON Schema sent to the model, which tells it "you may
omit this" rather than "do not produce this". A smaller class is a genuinely smaller
contract, and therefore genuinely fewer output tokens.
"""

from pydantic import BaseModel, Field

from classifier.schemas import FindingName, Label


class FindingVote(BaseModel):
    """One finding judged on one case - the label and nothing else."""

    finding: FindingName
    label: Label


class CaseLabels(BaseModel):
    """One replicate's complete read of one case."""

    case_id: str
    findings: list[FindingVote]


class FindingConfidence(BaseModel):
    """The model's self-rated certainty in one of its own labels."""

    finding: FindingName
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Certainty in the proposed label for this finding. "
            "0 means definitely uncertain, 100 means definitely certain."
        ),
    )


class CaseConfidence(BaseModel):
    """Step 2 of two-step confidence elicitation: one score per finding."""

    case_id: str
    scores: list[FindingConfidence]
