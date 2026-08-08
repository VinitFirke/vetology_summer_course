"""Data shapes for the classification task.

These Pydantic models are the single source of truth for *what* the model is asked
to label. Using an Enum for the finding names and a Literal for the label means an
invalid value cannot be constructed at all, rather than being caught later by a
validation step (REFERENCE.md REF4).
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FindingName(str, Enum):
    """The 19 findings the model judges.

    `diseased_lungs` is deliberately absent: it is derived in code from the
    findings below, never asked of the model. See DISEASED_LUNGS_INPUTS.
    """

    pulmonary_nodules = "pulmonary_nodules"
    esophagitis = "esophagitis"
    pneumonia = "pneumonia"
    bronchitis = "bronchitis"
    interstitial = "interstitial"
    hypo_plastic_trachea = "hypo_plastic_trachea"
    cardiomegaly = "cardiomegaly"
    right_sided_cardiomegaly = "right_sided_cardiomegaly"
    left_sided_cardiomegaly = "left_sided_cardiomegaly"
    pleural_effusion = "pleural_effusion"
    perihilar_infiltrate = "perihilar_infiltrate"
    focal_caudodorsal_lung = "focal_caudodorsal_lung"
    focal_perihilar = "focal_perihilar"
    bronchiectasis = "bronchiectasis"
    pulmonary_vessel_enlargement = "pulmonary_vessel_enlargement"
    thoracic_lymphadenopathy = "thoracic_lymphadenopathy"
    pulmonary_hypoinflation = "pulmonary_hypoinflation"
    pericardial_effusion = "pericardial_effusion"
    Alveolar_interstitial_pattern = "Alveolar_interstitial_pattern"


Label = Literal["normal", "abnormal"]

NORMAL: Label = "normal"
ABNORMAL: Label = "abnormal"

# Name of the derived column, and the findings that roll up into it.
DISEASED_LUNGS = "diseased_lungs"

DISEASED_LUNGS_INPUTS: tuple[FindingName, ...] = (
    FindingName.pulmonary_nodules,
    FindingName.pneumonia,
    FindingName.bronchitis,
    FindingName.interstitial,
    FindingName.Alveolar_interstitial_pattern,
    FindingName.perihilar_infiltrate,
    FindingName.focal_caudodorsal_lung,
    FindingName.focal_perihilar,
    FindingName.bronchiectasis,
    FindingName.pulmonary_hypoinflation,
)

# The 20 label columns in the exact order they appear in the CSV (columns 10-29).
LABEL_COLUMNS: tuple[str, ...] = (
    "pulmonary_nodules",
    "esophagitis",
    "pneumonia",
    "bronchitis",
    "interstitial",
    "diseased_lungs",
    "hypo_plastic_trachea",
    "cardiomegaly",
    "right_sided_cardiomegaly",
    "left_sided_cardiomegaly",
    "pleural_effusion",
    "perihilar_infiltrate",
    "focal_caudodorsal_lung",
    "focal_perihilar",
    "bronchiectasis",
    "pulmonary_vessel_enlargement",
    "thoracic_lymphadenopathy",
    "pulmonary_hypoinflation",
    "pericardial_effusion",
    "Alveolar_interstitial_pattern",
)

# Source columns we read. AI-report columns are deliberately never read.
COL_CASE_ID = "CaseID"
COL_FINDINGS = "Findings (original radiologist report)"
COL_CONCLUSIONS = "Conclusions (original radiologist report)"


class FindingLabel(BaseModel):
    """One finding judged on one case, with the reasoning behind it."""

    finding: FindingName
    label: Label
    evidence: str = Field(
        description=(
            "The sentence from the report that decided this label, quoted verbatim. "
            "Use an empty string if the report says nothing about this finding."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining why this label follows from the evidence."
    )


class CaseClassification(BaseModel):
    """The model's complete read of one case: all 19 findings judged."""

    case_id: str
    findings: list[FindingLabel]


class RadiologyCase(BaseModel):
    """One row of source data, as read from the CSV."""

    case_id: str
    findings_text: str
    conclusions_text: str
