"""classify_case with a fake model - proves the wiring without spending a cent.

This is only possible because main() builds the chat model and passes it in as an
argument (REFERENCE.md REF3c), rather than classify_case constructing one itself.
"""

import pytest

from classifier.classify import classify_case, labels_from
from classifier.prompt import load_prompt
from classifier import config
from classifier.schemas import (
    ABNORMAL,
    NORMAL,
    CaseClassification,
    FindingLabel,
    FindingName,
    RadiologyCase,
)

CASE = RadiologyCase(
    case_id="12345",
    findings_text="A minimal to mild diffuse bronchial pattern is present.",
    conclusions_text="1. Minimal-mild diffuse bronchial pulmonary pattern.",
)


class FakeRaw:
    """Stands in for the AIMessage, carrying only the token counts we read."""

    usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


class FakeStructured:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeModel:
    def __init__(self, responses):
        self.structured = FakeStructured(responses)

    def with_structured_output(self, _schema, include_raw=False):
        return self.structured


def make_classification(**overrides) -> CaseClassification:
    labels = {finding: NORMAL for finding in FindingName}
    labels.update(overrides)
    return CaseClassification(
        case_id="ignored-by-design",
        findings=[
            FindingLabel(finding=finding, label=label, evidence="", reasoning="none stated")
            for finding, label in labels.items()
        ],
    )


@pytest.fixture
def prompt():
    return load_prompt(config.PROMPT_FILE)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Skip the real retry backoff - we test the retry logic, not the clock."""
    monkeypatch.setattr("classifier.classify.time.sleep", lambda _seconds: None)


def test_returns_labels_and_usage(prompt):
    model = FakeModel([{"parsed": make_classification(), "raw": FakeRaw(), "parsing_error": None}])
    result = classify_case(model, prompt, CASE)

    assert len(result.classification.findings) == 19
    assert result.usage.total_tokens == 150
    assert labels_from(result.classification)[FindingName.bronchitis.value] == NORMAL


def test_case_id_is_overwritten_with_the_real_one(prompt):
    """Providers do not reliably echo the case id back."""
    model = FakeModel([{"parsed": make_classification(), "raw": FakeRaw(), "parsing_error": None}])
    result = classify_case(model, prompt, CASE)
    assert result.classification.case_id == "12345"


def test_abnormal_labels_survive_the_round_trip(prompt):
    classification = make_classification(**{FindingName.cardiomegaly: ABNORMAL})
    model = FakeModel([{"parsed": classification, "raw": FakeRaw(), "parsing_error": None}])

    labels = labels_from(classify_case(model, prompt, CASE).classification)
    assert labels[FindingName.cardiomegaly.value] == ABNORMAL
    assert labels[FindingName.pneumonia.value] == NORMAL


def test_retries_then_succeeds(prompt):
    model = FakeModel(
        [
            ConnectionError("provider hiccup"),
            {"parsed": make_classification(), "raw": FakeRaw(), "parsing_error": None},
        ]
    )
    result = classify_case(model, prompt, CASE, max_attempts=3)
    assert model.structured.calls == 2
    assert result.usage.total_tokens == 150


def test_raises_after_exhausting_attempts(prompt):
    model = FakeModel([ConnectionError("down")] * 3)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        classify_case(model, prompt, CASE, max_attempts=3)
    assert model.structured.calls == 3


def test_unparseable_output_is_retried_then_raises(prompt):
    model = FakeModel([{"parsed": None, "raw": None, "parsing_error": "bad json"}] * 2)
    with pytest.raises(RuntimeError):
        classify_case(model, prompt, CASE, max_attempts=2)


def test_backoff_grows_and_is_capped():
    """Rate limits need seconds, not milliseconds - but never more than a minute."""
    from classifier.classify import backoff_seconds

    assert [backoff_seconds(i) for i in range(4)] == [5, 10, 20, 40]
    assert backoff_seconds(10) == 60
