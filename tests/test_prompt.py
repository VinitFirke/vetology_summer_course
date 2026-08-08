"""The prompt must carry the report text and all 19 finding names, and no rules."""

from classifier import config
from classifier.prompt import finding_list_text, load_prompt, render_messages
from classifier.schemas import DISEASED_LUNGS, FindingName, RadiologyCase

CASE = RadiologyCase(
    case_id="12345",
    findings_text="A minimal to mild diffuse bronchial pattern is present.",
    conclusions_text="1. Minimal-mild diffuse bronchial pulmonary pattern.",
)


def test_prompt_file_loads():
    prompt = load_prompt(config.PROMPT_FILE)
    assert prompt.version
    assert "veterinary radiologist" in prompt.system.lower()


def test_finding_list_has_every_finding():
    text = finding_list_text()
    for finding in FindingName:
        assert finding.value in text
    assert len(text.strip().splitlines()) == 19


def test_derived_column_is_never_shown_to_the_model():
    assert DISEASED_LUNGS not in finding_list_text()


def test_rendered_message_carries_case_text():
    prompt = load_prompt(config.PROMPT_FILE)
    messages = render_messages(prompt, CASE)

    assert len(messages) == 2
    user_text = messages[1].content
    assert CASE.case_id in user_text
    assert CASE.findings_text in user_text
    assert CASE.conclusions_text in user_text
    for finding in FindingName:
        assert finding.value in user_text


def test_prompt_contains_no_clinical_decision_rules():
    """The whole point is to measure the model's own judgement, not rule-following."""
    prompt = load_prompt(config.PROMPT_FILE)
    lowered = (prompt.system + prompt.user_template).lower()
    for banned in ("differential", "bronchial pattern", "left-sided", "air bronchogram"):
        assert banned not in lowered, f"prompt leaks a clinical rule: {banned!r}"
