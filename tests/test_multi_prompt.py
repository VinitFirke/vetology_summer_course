"""Few-shot examples must become real conversation turns, and must match the schema."""

import json

import pytest

from classifier_multi import config
from classifier_multi.categories import CATEGORIES, get_category
from classifier_multi.prompt import (
    PromptExample,
    example_answer,
    load_prompt,
    render_messages,
)
from classifier_multi.schemas import RadiologyCase

CASE = RadiologyCase(
    case_id="TEST-1", findings_text="Findings text.", conclusions_text="Conclusions text."
)


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_both_prompt_variants_load(name):
    category = get_category(name)
    for variant in config.VARIANTS:
        prompt = load_prompt(config.prompt_path(category, variant))
        assert prompt.version
        assert "veterinary radiologist" in prompt.system.lower()


def test_zero_shot_prompt_is_two_messages():
    category = get_category("canine_abdomen")
    prompt = load_prompt(config.prompt_path(category, "zeroshot"))
    assert prompt.examples == []
    assert len(render_messages(prompt, category, CASE)) == 2


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_few_shot_adds_two_messages_per_example(name):
    category = get_category(name)
    prompt = load_prompt(config.prompt_path(category, "fewshot"))
    assert prompt.examples, f"{name} few-shot prompt has no examples"
    messages = render_messages(prompt, category, CASE)
    assert len(messages) == 2 * len(prompt.examples) + 2
    assert [m.type for m in messages[1:-1]] == ["human", "ai"] * len(prompt.examples)
    assert messages[-1].type == "human"


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_example_answers_cover_every_asked_finding_in_order(name):
    category = get_category(name)
    prompt = load_prompt(config.prompt_path(category, "fewshot"))
    for example in prompt.examples:
        answer = json.loads(example_answer(category, example))
        assert [f["finding"] for f in answer["findings"]] == list(category.asked_findings)
        assert all(f["label"] in ("normal", "abnormal") for f in answer["findings"])


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_example_answers_leave_evidence_and_reasoning_empty(name):
    """Inventing quotes in the examples would teach the model to invent quotes."""
    category = get_category(name)
    prompt = load_prompt(config.prompt_path(category, "fewshot"))
    for example in prompt.examples:
        answer = json.loads(example_answer(category, example))
        assert all(f["evidence"] == "" for f in answer["findings"])
        assert all(f["reasoning"] == "" for f in answer["findings"])


def test_unknown_finding_name_raises():
    category = get_category("canine_abdomen")
    bad = PromptExample(
        case_id="X", findings="f", conclusions="c", labels={"not_a_finding": "abnormal"}
    )
    with pytest.raises(ValueError, match="not_a_finding"):
        example_answer(category, bad)


def test_derived_column_is_rejected_as_a_label():
    """diseased_lungs is computed in code, so it must never appear in an example."""
    category = get_category("feline_thorax")
    bad = PromptExample(
        case_id="X", findings="f", conclusions="c", labels={"diseased_lungs": "abnormal"}
    )
    with pytest.raises(ValueError, match="diseased_lungs"):
        example_answer(category, bad)


def test_omitted_findings_default_to_normal():
    category = get_category("canine_abdomen")
    example = PromptExample(
        case_id="X", findings="f", conclusions="c", labels={"colitis": "abnormal"}
    )
    answer = json.loads(example_answer(category, example))
    by_name = {f["finding"]: f["label"] for f in answer["findings"]}
    assert by_name["colitis"] == "abnormal"
    assert by_name["gastritis"] == "normal"
