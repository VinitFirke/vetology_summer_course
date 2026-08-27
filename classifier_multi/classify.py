"""Asking the model to judge one case.

with_structured_output(schema) makes the provider return JSON matching the per-category
Pydantic schema, so a model cannot hand back "Abnormal ", "1", or a finding name that
is not on that category's list.
"""

import time

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from classifier_multi.categories import Category
from classifier_multi.prompt import Prompt, render_messages
from classifier_multi.schemas import (
    CaseClassification,
    RadiologyCase,
    classification_schema,
)


# How the answer schema is enforced. Every provider is asked the same way, because a
# comparison in which one model answered under a different contract would not be one.
#
# Not the library default. langchain's default routes ChatOpenAI through the OpenAI
# structured-output endpoint, which parses the reply with model_validate_json - and
# every Ollama Cloud model tested wraps its JSON in a ```json fence, which is not
# valid JSON, so every call failed. Measured on one canine_abdomen case:
#
#   method             gemma   qwen    kimi    nemotron
#   json_schema (dflt) fence   -       -       -
#   function_calling   ok      no tool call returned by the other three
#   json_mode          ok      ok      ok      ok
#
# json_mode asks for JSON and validates it here rather than sending the schema in the
# request, so the finding names and the answer shape are carried by the prompt: the
# finding list render_messages() injects, and - in the few-shot prompts - the worked
# example turns. A reply that invents a finding name still cannot get through, it just
# fails at parse time and is retried instead of being rejected by the provider.
STRUCTURED_METHOD = "json_mode"


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
    """What one case cost and what the model concluded."""

    classification: CaseClassification
    usage: Usage


def _usage_from_raw(raw_message: object) -> Usage:
    """Pull token counts off the raw AIMessage, if the provider reported any."""
    data = getattr(raw_message, "usage_metadata", None) or {}
    return Usage(
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
    )


# HTTP statuses that mean "this will never work", as opposed to "try again shortly":
# bad key, unpaid or expired subscription, key not entitled to this model. Retrying
# these is not just useless, it is expensive in wall-clock: ten attempts of capped
# backoff is over six minutes per case, so a dead provider would stall a 50-case run
# for five hours before reporting anything. Mistral returning 402 during this project
# is exactly the case that motivated the check.
FATAL_STATUS_CODES: tuple[int, ...] = (401, 402, 403, 404)


class ProviderUnavailable(RuntimeError):
    """The provider rejected the request in a way that retrying cannot fix."""


def fatal_status(error: Exception) -> int | None:
    """Return the HTTP status if this error is one there is no point retrying.

    Providers surface status codes inconsistently - httpx puts it on
    error.response.status_code, the OpenAI SDK on error.status_code - so both are
    checked, and anything unrecognised is treated as retryable.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if status in FATAL_STATUS_CODES else None


def backoff_seconds(attempt: int) -> int:
    """Wait before retry number `attempt`, capped at a minute.

    Starts at 5s rather than 1s because the failures worth retrying are mostly rate
    limits, and a 1-2 second pause is far too short to clear one.
    """
    return min(60, 5 * 2**attempt)


def invoke_structured(
    structured: object,
    messages: list,
    description: str,
    max_attempts: int = 10,
) -> tuple[BaseModel, object]:
    """Invoke a structured model, retrying transient failures. Returns (parsed, raw).

    Every paid call in this codebase goes through here, so there is one backoff policy
    rather than one per call site.
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = structured.invoke(messages)
            parsed = response.get("parsed")
            if parsed is None:
                raise ValueError(
                    f"Model returned unparseable output: {response.get('parsing_error')}"
                )
            return parsed, response.get("raw")
        except Exception as error:  # noqa: BLE001 - retry any provider-side failure
            status = fatal_status(error)
            if status is not None:
                raise ProviderUnavailable(
                    f"{description}: provider returned HTTP {status}, which retrying "
                    f"cannot fix - check the API key, subscription, and model access. "
                    f"{error}"
                ) from error
            last_error = error
            if attempt < max_attempts - 1:
                time.sleep(backoff_seconds(attempt))

    raise RuntimeError(
        f"{description} failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def classify_case(
    model: BaseChatModel,
    prompt: Prompt,
    category: Category,
    case: RadiologyCase,
    max_attempts: int = 10,
) -> CaseResult:
    """Judge every asked-about finding for one case.

    Retries with exponential backoff on transient provider errors. Raises if every
    attempt fails, so the caller can record the failure rather than write a silently
    empty row.
    """
    structured = model.with_structured_output(
        classification_schema(category), include_raw=True, method=STRUCTURED_METHOD
    )
    messages = render_messages(prompt, category, case)

    parsed, raw = invoke_structured(
        structured, messages, f"{category.name} case {case.case_id}", max_attempts
    )
    # Providers do not always echo the case id back correctly.
    parsed.case_id = case.case_id
    return CaseResult(classification=parsed, usage=_usage_from_raw(raw))
