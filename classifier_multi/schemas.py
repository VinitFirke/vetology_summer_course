"""Data shapes for the classification task, built per category.

The single-category classifier could hard-code a FindingName enum, because there was
only ever one set of findings. Here the finding names differ per study type, so the
enum is built at runtime from Category.asked_findings and the answer schema is built
around it with pydantic's create_model.

The point of going to that trouble rather than typing `finding: str` is unchanged
from the original: with an enum in the schema, a model physically cannot return a
finding name that is not on the list, so there is no validation step to forget.
Schemas are cached per category because with_structured_output is called once per
provider and rebuilding the classes each time would defeat pydantic's own caching.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

from classifier_multi.categories import Category, get_category

Label = Literal["normal", "abnormal"]

NORMAL: Label = "normal"
ABNORMAL: Label = "abnormal"

# Source columns we read. The AI-report columns are deliberately never read: the task
# is to reproduce the manual scoring of the *radiologist's* report.
COL_CASE_ID = "CaseID"
COL_FINDINGS = "Findings (original radiologist report)"
COL_CONCLUSIONS = "Conclusions (original radiologist report)"

EVIDENCE_DESCRIPTION = (
    "The sentence from the report that decided this label, quoted verbatim. "
    "Use an empty string if the report says nothing about this finding."
)
REASONING_DESCRIPTION = "One sentence explaining why this label follows from the evidence."


class RadiologyCase(BaseModel):
    """One row of source data, as read from the CSV."""

    case_id: str
    findings_text: str
    conclusions_text: str


class CaseClassification(BaseModel):
    """Base type for a model's complete read of one case.

    Declared so the rest of the codebase has a stable name to annotate against;
    the concrete per-category subclass is produced by classification_schema().
    """

    case_id: str
    findings: list


def finding_enum(category: Category) -> type[Enum]:
    """A str Enum of exactly the findings this category asks the model to judge."""
    return _finding_enum_cached(category.name)


def classification_schema(category: Category) -> type[CaseClassification]:
    """The structured-output schema for one category.

    Returns a CaseClassification subclass whose `findings` is a list of per-finding
    judgements, each restricted to this category's finding names.
    """
    return _classification_schema_cached(category.name)


# Cached on the category *name* rather than the Category itself: Category holds dict
# fields, so it is not hashable and cannot be an lru_cache key.
@lru_cache(maxsize=None)
def _finding_enum_cached(category_name: str) -> type[Enum]:
    category = get_category(category_name)
    return Enum(  # type: ignore[return-value]
        f"{category.name}_FindingName",
        {name: name for name in category.asked_findings},
        type=str,
    )


@lru_cache(maxsize=None)
def _classification_schema_cached(category_name: str) -> type[CaseClassification]:
    category = get_category(category_name)
    finding_label = create_model(
        f"{category.name}_FindingLabel",
        finding=(finding_enum(category), ...),
        label=(Label, ...),
        evidence=(str, Field(description=EVIDENCE_DESCRIPTION)),
        reasoning=(str, Field(description=REASONING_DESCRIPTION)),
        __doc__="One finding judged on one case, with the reasoning behind it.",
    )
    return create_model(
        f"{category.name}_CaseClassification",
        __base__=CaseClassification,
        findings=(list[finding_label], ...),  # type: ignore[valid-type]
        __doc__=f"The model's complete read of one {category.name} case.",
    )


def labels_from(classification: CaseClassification) -> dict[str, Label]:
    """Flatten the model's answer into {finding_name: label}."""
    return {item.finding.value: item.label for item in classification.findings}
