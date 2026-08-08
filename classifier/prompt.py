"""Loading the prompt file and turning one case into chat messages.

The prompt text lives in prompts/classification_prompt.json so it can be edited
without touching code. The finding names are NOT in that file - they come from
the FindingName enum, so the schema and the prompt can never disagree.
"""

import json
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from classifier.schemas import FindingName, RadiologyCase


class Prompt(BaseModel):
    """The editable parts of the prompt, as loaded from JSON."""

    version: str
    system: str
    user_template: str


def load_prompt(prompt_path: Path) -> Prompt:
    text = prompt_path.read_text(encoding="utf-8")
    return Prompt.model_validate(json.loads(text))


def finding_list_text() -> str:
    """The 19 finding names, one per line, exactly as the schema spells them."""
    return "\n".join(f"- {finding.value}" for finding in FindingName)


def render_messages(prompt: Prompt, case: RadiologyCase) -> list[BaseMessage]:
    """Build the two messages sent to the model for one case."""
    user_text = prompt.user_template.format(
        case_id=case.case_id,
        findings=case.findings_text,
        conclusions=case.conclusions_text,
        n_findings=len(FindingName),
        finding_list=finding_list_text(),
    )
    return [SystemMessage(content=prompt.system), HumanMessage(content=user_text)]
