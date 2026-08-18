"""The order cases get read in.

A flat random sample of 300 would satisfy the count and fail the brief: the rare
findings would not appear at all, and a disease never scored abnormal cannot measure
whether a model detects it. So the worklist leads with candidates mined for the
rarest column first, then falls back to a seeded random draw over everything left.

Ordering rather than choosing is the point. The reader works down the list and stops
at 300, so cases rejected on reading cost a slot from the tail rather than needing a
second selection pass, and the tail stays an unbiased sample of the eligible pool.
"""

import json
import random

from pydantic import BaseModel

from gold_standard.csv_io import column_index, read_sheet
from gold_standard.keywords import candidate_pattern
from gold_standard.screen import eligible_case_ids, report_text
from gold_standard.sheets import COL_CASE_ID, Sheet, gold_csv_path, worklist_path

DEFAULT_SEED = 20260818

# How many mined candidates to queue per column. The coverage target is three, and
# roughly half of the hits are rejected on reading, so eight leaves room to spare
# without the coverage block crowding out the random tail.
CANDIDATES_PER_COLUMN = 8


class WorkItem(BaseModel):
    """One case queued for reading, and why it was queued."""

    case_id: str
    reason: str  # "coverage:<column>" or "random"
    position: int


def build_worklist(sheet: Sheet, seed: int = DEFAULT_SEED) -> list[WorkItem]:
    """Rank the eligible pool: mined candidates first, then a seeded random tail."""
    eligible = eligible_case_ids(sheet)
    eligible_set = set(eligible)

    header, rows = read_sheet(gold_csv_path(sheet))
    case_col = column_index(header, COL_CASE_ID)
    texts = {
        row[case_col]: report_text(header, row)
        for row in rows
        if row[case_col] in eligible_set
    }

    hits_by_column = {
        column: [cid for cid in eligible if candidate_pattern(column).search(texts[cid])]
        for column in sheet.judged_columns
    }
    # Rarest column first, so a case that satisfies several columns is credited to the
    # one that needs it most.
    rarest_first = sorted(sheet.judged_columns, key=lambda c: len(hits_by_column[c]))

    ordered: list[WorkItem] = []
    taken: set[str] = set()
    for column in rarest_first:
        queued = 0
        for case_id in hits_by_column[column]:
            if case_id in taken:
                continue
            ordered.append(
                WorkItem(case_id=case_id, reason=f"coverage:{column}", position=len(ordered))
            )
            taken.add(case_id)
            queued += 1
            if queued == CANDIDATES_PER_COLUMN:
                break

    remaining = [cid for cid in eligible if cid not in taken]
    random.Random(seed).shuffle(remaining)
    for case_id in remaining:
        ordered.append(WorkItem(case_id=case_id, reason="random", position=len(ordered)))

    return ordered


def save_worklist(sheet: Sheet, items: list[WorkItem]) -> None:
    worklist_path(sheet).write_text(
        json.dumps([i.model_dump() for i in items], indent=2), encoding="utf-8"
    )


def load_worklist(sheet: Sheet) -> list[WorkItem]:
    raw = json.loads(worklist_path(sheet).read_text(encoding="utf-8"))
    return [WorkItem(**item) for item in raw]
