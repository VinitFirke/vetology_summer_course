"""Loading a category's prompt file and turning one case into chat messages.

Each study type has its own JSON file under classifier_multi/prompts/, so the persona
and the labelling conventions can be written for that species and body region without
a shared template trying to cover all three.

What is *not* in those files is the list of finding names: it comes from the category
definition, so the prompt and the answer schema can never disagree about what is
being asked.

A prompt file may also carry an `examples` list: worked cases that are replayed as
extra turns of the conversation before the real one, so the model sees the shape of a
correct answer before it is asked for one. Each example is written as report text plus
the labels that follow from it; the assistant half of the turn is generated here from
the category's finding list, so a hand-written example cannot drift out of step with
the schema the model is being asked to fill.
"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from classifier_multi.categories import Category
from classifier_multi.schemas import NORMAL, Label, RadiologyCase


class PromptExample(BaseModel):
    """One worked case, replayed to the model as a demonstration turn.

    `labels` need only name the findings worth stating - normally the abnormal ones.
    Anything left out is filled in as "normal", which is the same convention the system
    prompt states for findings the report does not mention, so a ten-finding example
    can be written as the one line that actually carries information.
    """

    case_id: str
    findings: str
    conclusions: str
    labels: dict[str, Label] = {}


class Prompt(BaseModel):
    """The editable parts of the prompt, as loaded from JSON."""

    version: str
    system: str
    user_template: str
    examples: list[PromptExample] = []


def load_prompt(prompt_path: Path) -> Prompt:
    text = prompt_path.read_text(encoding="utf-8")
    return Prompt.model_validate(json.loads(text))


def finding_list_text(category: Category) -> str:
    """The finding names, one per line, exactly as the schema spells them."""
    return "\n".join(f"- {name}" for name in category.asked_findings)


def render_user_text(
    prompt: Prompt, category: Category, case: RadiologyCase
) -> str:
    """Fill the user template with one case's report text."""
    return prompt.user_template.format(
        case_id=case.case_id,
        findings=case.findings_text,
        conclusions=case.conclusions_text,
        n_findings=len(category.asked_findings),
        finding_list=finding_list_text(category),
    )


def example_answer(category: Category, example: PromptExample) -> str:
    """The assistant half of an example turn: the answer the model should have given.

    Built from the category rather than hand-written, so every example covers every
    asked finding in the schema's own order and spelling. Evidence and reasoning are
    left empty: the examples exist to demonstrate the labelling, and inventing quotes
    for them would teach the model to invent quotes too.
    """
    unknown = sorted(set(example.labels) - set(category.asked_findings))
    if unknown:
        raise ValueError(
            f"Example {example.case_id!r} labels findings that {category.name} does not "
            f"ask about: {', '.join(unknown)}. "
            f"Valid names: {', '.join(category.asked_findings)}"
        )
    findings = [
        {
            "finding": name,
            "label": example.labels.get(name, NORMAL),
            "evidence": "",
            "reasoning": "",
        }
        for name in category.asked_findings
    ]
    return json.dumps(
        {"case_id": example.case_id, "findings": findings}, indent=2
    )


def render_messages(
    prompt: Prompt, category: Category, case: RadiologyCase
) -> list[BaseMessage]:
    """Build the messages sent to the model for one case.

    The system prompt, then a Human/AI pair per example, then the case being asked
    about. With no examples in the prompt file this is the same two messages it has
    always been.
    """
    messages: list[BaseMessage] = [SystemMessage(content=prompt.system)]
    for example in prompt.examples:
        example_case = RadiologyCase(
            case_id=example.case_id,
            findings_text=example.findings,
            conclusions_text=example.conclusions,
        )
        messages.append(
            HumanMessage(content=render_user_text(prompt, category, example_case))
        )
        messages.append(AIMessage(content=example_answer(category, example)))
    messages.append(
        HumanMessage(content=render_user_text(prompt, category, case))
    )
    return messages
