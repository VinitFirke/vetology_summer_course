"""The three gold-standard sheets and everything that differs between them.

Deliberately separate from classifier_multi.categories: that module describes the
files the *models* are scored on, which are named for a 50-case run and carry no
region marker. This one describes the raw scoring sheets being labelled now. The
label column tuples and the diseased_lungs roll-up are copied to match
classifier_multi exactly, so gold and model derivations can never diverge.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Region = Literal["thorax", "abdomen"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "dataset_gold_standard"
BACKUP_DIR = GOLD_DIR / "_originals"

COL_CASE_ID = "CaseID"
COL_FINDINGS = "Findings (original radiologist report)"
COL_CONCLUSIONS = "Conclusions (original radiologist report)"

NORMAL = "normal"
ABNORMAL = "abnormal"
VALID_LABELS = frozenset({NORMAL, ABNORMAL})

# Parenchymal and lower-airway findings roll up into diseased_lungs. Copied from
# classifier_multi.categories; the feline list carries the extra alveolar column.
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


class Sheet(BaseModel):
    """One scoring sheet: its file, its region, its columns, its roll-ups."""

    name: str
    csv_name: str
    region: Region
    label_columns: tuple[str, ...]
    derived: dict[str, tuple[str, ...]] = {}
    target_cases: int = 300

    @property
    def judged_columns(self) -> tuple[str, ...]:
        """Columns a reader judges directly: every label column that is not derived."""
        return tuple(c for c in self.label_columns if c not in self.derived)


CANINE_THORAX = Sheet(
    name="canine_thorax",
    csv_name="canine_thorax_scoring_data_gold_standard(Sheet1).csv",
    region="thorax",
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

FELINE_THORAX = Sheet(
    name="feline_thorax",
    csv_name="feline_thorax_scoring_data_gold_standard(Sheet1).csv",
    region="thorax",
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

CANINE_ABDOMEN = Sheet(
    name="canine_abdomen",
    csv_name="canine_abdomen_scoring_data_gold_standard(Sheet1).csv",
    region="abdomen",
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
)

SHEETS: dict[str, Sheet] = {
    s.name: s for s in (CANINE_THORAX, FELINE_THORAX, CANINE_ABDOMEN)
}


def get_sheet(name: str) -> Sheet:
    if name not in SHEETS:
        raise ValueError(f"Unknown sheet {name!r}. Choose one of: {', '.join(SHEETS)}")
    return SHEETS[name]


def gold_csv_path(sheet: Sheet) -> Path:
    return GOLD_DIR / sheet.csv_name


def backup_csv_path(sheet: Sheet) -> Path:
    return BACKUP_DIR / sheet.csv_name


def evidence_path(sheet: Sheet) -> Path:
    return GOLD_DIR / f"evidence_{sheet.name}.json"


def worklist_path(sheet: Sheet) -> Path:
    return GOLD_DIR / f"worklist_{sheet.name}.json"
