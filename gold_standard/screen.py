"""Deciding which cases belong to which sheet.

Every sheet mixes study types: barely half of the canine thorax rows evaluate a
thorax at all, the rest being abdominal, spinal or appendicular studies that happen
to share the worklist. The brief asks for region-focused cases only, so a case is
eligible when its report actually *evaluates* the region rather than merely
mentioning it.

The distinction is why these patterns look for descriptive statements about the
region's structures - "the cardiac silhouette is", "pulmonary pattern" - and not for
the words "thorax" or "abdomen". An abdominal report that closes with "the included
thorax is normal" has not read the thorax in any way a labeller could score.

Screening is a filter, not an authority. The reader still drops a case that turns out
on the page not to evaluate the region, and takes the next one from the pool.
"""

import re

from gold_standard.csv_io import column_index, read_sheet
from gold_standard.sheets import (
    COL_CASE_ID,
    COL_CONCLUSIONS,
    COL_FINDINGS,
    Region,
    Sheet,
    gold_csv_path,
)

_THORAX = re.compile(
    r"pulmonary parenchyma\s*:"
    r"|cardiac silhouette\s*:"
    r"|pleural space\s*:"
    r"|pulmonary vasculature\s*:"
    r"|cardiac silhouette (?:is|size)"
    r"|pulmonary vasculature (?:is|are)"
    r"|pulmonary pattern"
    r"|lung lobe"
    r"|bronchial pattern"
    r"|interstitial pattern"
    r"|alveolar pattern"
    r"|vertebral heart",
    re.I,
)

_ABDOMEN = re.compile(
    r"\bliver\s*:"
    r"|\bspleen\s*:"
    r"|\bkidneys\s*:"
    r"|urinary bladder\s*:"
    r"|peritoneum\s*:"
    r"|retroperitoneum\s*:"
    r"|gastrointestinal tract\s*:"
    r"|serosal detail"
    r"|small intestin(?:e|al)"
    r"|the liver and spleen"
    r"|abdominal (?:serosal|detail)",
    re.I,
)

_PATTERNS: dict[str, re.Pattern[str]] = {"thorax": _THORAX, "abdomen": _ABDOMEN}


def evaluates_region(text: str, region: Region) -> bool:
    """True when the report substantively evaluates that body region."""
    return bool(_PATTERNS[region].search(text))


def report_text(header: list[str], row: list[str]) -> str:
    """The two source columns joined - the only text a labeller may read."""
    findings = row[column_index(header, COL_FINDINGS)]
    conclusions = row[column_index(header, COL_CONCLUSIONS)]
    return f"{findings}\n{conclusions}"


def eligible_case_ids(sheet: Sheet) -> list[str]:
    """CaseIDs whose report evaluates this sheet's region, in file order.

    A CaseID is returned once even if the sheet lists it twice; the duplicate row on
    the feline sheet carries the same report and would otherwise be scored twice.
    """
    header, rows = read_sheet(gold_csv_path(sheet))
    case_col = column_index(header, COL_CASE_ID)

    seen: set[str] = set()
    eligible: list[str] = []
    for row in rows:
        case_id = row[case_col]
        if case_id in seen:
            continue
        if evaluates_region(report_text(header, row), sheet.region):
            seen.add(case_id)
            eligible.append(case_id)
    return eligible
