"""Asking the model to judge one case.

with_structured_output(CaseClassification) makes the provider return JSON matching
the Pydantic schema, so a model cannot hand back "Abnormal ", "1", or a finding
name that is not one of the 19.
"""

import time

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from classifier.prompt import Prompt, render_messages
from classifier.schemas import CaseClassification, Label, RadiologyCase


class Usage(BaseModel):
    """Tokens spent on one call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class CaseResult(BaseModel):
    """What one case costs and what the model concluded."""

    classification: CaseClassification
    usage: Usage


def labels_from(classification: CaseClassification) -> dict[str, Label]:
    """Flatten the model's answer into {finding_name: label}."""
    return {item.finding.value: item.label for item in classification.findings}


def _usage_from_raw(raw_message: object) -> Usage:
    """Pull token counts off the raw AIMessage, if the provider reported any."""
    data = getattr(raw_message, "usage_metadata", None) or {}
    return Usage(
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
    )


def backoff_seconds(attempt: int) -> int:
    """Wait before retry number `attempt`, capped at a minute.

    Starts at 5s rather than 1s because the failures worth retrying are mostly
    rate limits, and a 1-2 second pause is far too short to clear one.
    """
    return min(60, 5 * 2**attempt)


def classify_case(
    model: BaseChatModel,
    prompt: Prompt,
    case: RadiologyCase,
    max_attempts: int = 10,
) -> CaseResult:
    """Judge all 19 findings for one case.

    Retries with exponential backoff on transient provider errors. Raises if every
    attempt fails, so the caller can record the failure rather than write a
    silently empty row.
    """
    structured = model.with_structured_output(CaseClassification, include_raw=True)
    messages = render_messages(prompt, case)

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = structured.invoke(messages)
            parsed = response.get("parsed")
            if parsed is None:
                raise ValueError(
                    f"Model returned unparseable output: {response.get('parsing_error')}"
                )
            # Providers do not always echo the case id back correctly.
            parsed.case_id = case.case_id
            return CaseResult(classification=parsed, usage=_usage_from_raw(response.get("raw")))
        except Exception as error:  # noqa: BLE001 - retry any provider-side failure
            last_error = error
            if attempt < max_attempts - 1:
                time.sleep(backoff_seconds(attempt))

    raise RuntimeError(
        f"Case {case.case_id} failed after {max_attempts} attempts: {last_error}"
    ) from last_error
