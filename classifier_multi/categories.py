"""The three study categories, and everything that differs between them.

This module is the only place that knows a canine abdomen has ten findings and a
canine thorax has twenty columns. Every other module takes a Category and works
from it, so adding a fourth study type means adding one entry here rather than
editing branching logic spread across the codebase.

The label column tuples are copied verbatim from the CSV headers, in file order,
because csv_io writes them back by name and verify_columns fails loudly if the
spreadsheet and this file ever drift apart.

The order is the CSV's own header order. It sets the order findings are put to the
model and the order of the answer schema; it does not affect the written file, which
csv_io fills in by column name.
"""

from typing import Literal

from pydantic import BaseModel

CategoryName = Literal["feline_thorax", "canine_thorax", "canine_abdomen"]

CATEGORY_NAMES: tuple[CategoryName, ...] = (
    "feline_thorax",
    "canine_thorax",
    "canine_abdomen",
)


class Category(BaseModel):
    """One study type: its data file, its findings, and its prompt.

    `derived` maps a column the model is never asked about to the columns that roll
    up into it. A derived column is abnormal when any of its inputs is abnormal.
    Deriving in code rather than asking the model means the summary column can never
    contradict the findings underneath it.

    Nothing here knows the answers exist. The manually scored spreadsheets, their
    filenames and the vocabulary their scorers used belong to evaluate.py, which is
    the only module allowed to read them.
    """

    name: CategoryName
    input_csv: str
    label_columns: tuple[str, ...]
    derived: dict[str, tuple[str, ...]] = {}

    @property
    def asked_findings(self) -> tuple[str, ...]:
        """The findings put to the model: every label column that is not derived."""
        return tuple(c for c in self.label_columns if c not in self.derived)

    def prompt_filename(self, variant: str | None = None) -> str:
        """The prompt file for this category, optionally a named variant of it.

        The zero-shot prompt is the bare category name; a variant such as "fewshot"
        is a suffix, so the baseline file is never the one edited when a new prompt
        style is tried and the two stay comparable run to run.
        """
        stem = self.name if variant is None else f"{self.name}_{variant}"
        return f"{stem}.json"


# Lung parenchyma and lower-airway findings roll up into diseased_lungs. The feline
# list carries Alveolar_interstitial_pattern; the canine sheet has no such column.
_FELINE_DISEASED_LUNGS: tuple[str, ...] = (
    "pulmonary_nodules",
    "pneumonia",
    "bronchitis",
    "interstitial",
    "Alveolar_interstitial_pattern",
    "perihilar_infiltrate",
    "focal_caudodorsal_lung",
    "focal_perihilar",
    "bronchiectasis",
    "pulmonary_hypoinflation",
)

_CANINE_DISEASED_LUNGS: tuple[str, ...] = tuple(
    c for c in _FELINE_DISEASED_LUNGS if c != "Alveolar_interstitial_pattern"
)


FELINE_THORAX = Category(
    name="feline_thorax",
    input_csv="feline_thorax_TOBE_classified(in).csv",
    label_columns=(
        "pulmonary_nodules",
        "esophagitis",
        "pneumonia",
        "bronchitis",
        "interstitial",
        "diseased_lungs",
        "hypo_plastic_trachea",
        "cardiomegaly",
        "pleural_effusion",
        "perihilar_infiltrate",
        "focal_caudodorsal_lung",
        "right_sided_cardiomegaly",
        "focal_perihilar",
        "left_sided_cardiomegaly",
        "bronchiectasis",
        "pulmonary_vessel_enlargement",
        "thoracic_lymphadenopathy",
        "pulmonary_hypoinflation",
        "pericardial_effusion",
        "Alveolar_interstitial_pattern",
    ),
    derived={"diseased_lungs": _FELINE_DISEASED_LUNGS},
)

CANINE_THORAX = Category(
    name="canine_thorax",
    input_csv="canine_thorax_TOBE_classified(in).csv",
    label_columns=(
        "perihilar_infiltrate",
        "pneumonia",
        "bronchitis",
        "interstitial",
        "diseased_lungs",
        "hypo_plastic_trachea",
        "cardiomegaly",
        "pulmonary_nodules",
        "pleural_effusion",
        "focal_caudodorsal_lung",
        "focal_perihilar",
        "pulmonary_hypoinflation",
        "right_sided_cardiomegaly",
        "pericardial_effusion",
        "bronchiectasis",
        "pulmonary_vessel_enlargement",
        "left_sided_cardiomegaly",
        "thoracic_lymphadenopathy",
        "esophagitis",
        "vhs",
    ),
    derived={"diseased_lungs": _CANINE_DISEASED_LUNGS},
)

CANINE_ABDOMEN = Category(
    name="canine_abdomen",
    input_csv="canine_abdomen_TOBE_classified(in).csv",
    label_columns=(
        "gastritis",
        "ascites",
        "colitis",
        "liver_mass",
        "pancreatitis",
        "microhepatia",
        "small_intestinal_obstruction",
        "splenic_mass",
        "splenomegaly",
        "hepatomegaly",
    ),
    # No summary column on the abdominal sheet: every column is judged directly.
)

CATEGORIES: dict[CategoryName, Category] = {
    c.name: c for c in (FELINE_THORAX, CANINE_THORAX, CANINE_ABDOMEN)
}


def get_category(name: str) -> Category:
    if name not in CATEGORIES:
        raise ValueError(
            f"Unknown category {name!r}. Choose one of: {', '.join(CATEGORY_NAMES)}"
        )
    return CATEGORIES[name]  # type: ignore[index]
