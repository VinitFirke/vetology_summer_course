"""The slim UQ schemas, and classify_case honouring the schema it is handed."""

import pytest
from pydantic import ValidationError

from classifier import config
from classifier.classify import classify_case
from classifier.prompt import load_prompt
from classifier.schemas import CaseClassification, FindingName, RadiologyCase
from uncertainty.schemas import CaseConfidence, CaseLabels, FindingConfidence, FindingVote

CASE = RadiologyCase(
    case_id="12345",
    findings_text="A minimal to mild diffuse bronchial pattern is present.",
    conclusions_text="1. Minimal-mild diffuse bronchial pulmonary pattern.",
)


class FakeRaw:
    usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


class RecordingModel:
    """Remembers which schema it was asked to bind, so the test can assert on it."""

    def __init__(self, response):
        self.response = response
        self.requested_schema = None

    def with_structured_output(self, schema, include_raw=False):
        self.requested_schema = schema
        return self

    def invoke(self, _messages):
        return self.response


@pytest.fixture
def prompt():
    return load_prompt(config.PROMPT_FILE)


def test_case_labels_accepts_a_valid_vote():
    labels = CaseLabels(
        case_id="1",
        findings=[FindingVote(finding=FindingName.cardiomegaly, label="abnormal")],
    )
    assert labels.findings[0].label == "abnormal"


def test_finding_vote_rejects_a_label_outside_the_two():
    with pytest.raises(ValidationError):
        FindingVote(finding=FindingName.cardiomegaly, label="Abnormal ")


def test_slim_schema_carries_no_evidence_or_reasoning():
    """These two fields are ~45% of output tokens and no proxy reads them."""
    assert "evidence" not in FindingVote.model_fields
    assert "reasoning" not in FindingVote.model_fields


def test_confidence_rejects_a_score_above_one_hundred():
    with pytest.raises(ValidationError):
        FindingConfidence(finding=FindingName.pneumonia, score=150)


def test_confidence_accepts_both_boundaries():
    assert FindingConfidence(finding=FindingName.pneumonia, score=0).score == 0
    assert FindingConfidence(finding=FindingName.pneumonia, score=100).score == 100


def test_case_confidence_holds_many_scores():
    payload = CaseConfidence(
        case_id="1",
        scores=[FindingConfidence(finding=FindingName.pneumonia, score=80)],
    )
    assert payload.scores[0].score == 80


def test_classify_case_binds_the_schema_it_is_given(prompt):
    parsed = CaseLabels(case_id="x", findings=[])
    model = RecordingModel({"parsed": parsed, "raw": FakeRaw(), "parsing_error": None})

    classify_case(model, prompt, CASE, schema=CaseLabels)

    assert model.requested_schema is CaseLabels


def test_classify_case_defaults_to_the_full_schema(prompt):
    """No current caller passes schema; they must keep getting CaseClassification."""
    parsed = CaseClassification(case_id="x", findings=[])
    model = RecordingModel({"parsed": parsed, "raw": FakeRaw(), "parsing_error": None})

    classify_case(model, prompt, CASE)

    assert model.requested_schema is CaseClassification
