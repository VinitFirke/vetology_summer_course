"""What was judged abnormal, and the sentence that decided it.

This store, not the CSV, is the source of truth. The CSV is a projection of it, so an
interrupted run resumes from the last saved batch and a rebuild can never disagree
with the judgements already made.

Only abnormal findings are recorded. Under the agreed rubric a finding the report does
not mention is normal, so listing every normal column would be several thousand cells
of restated default carrying no information - while every abnormal cell, the ones that
actually claim something, is backed by a verbatim quote and a reason.
"""

import json

from pydantic import BaseModel

from gold_standard.sheets import ABNORMAL, NORMAL, Sheet, evidence_path


class Judgement(BaseModel):
    """Why one finding was called abnormal on one case."""

    evidence: str  # verbatim from the report
    reasoning: str  # one sentence


class CaseRecord(BaseModel):
    """One case's abnormal findings. Anything absent is normal."""

    case_id: str
    abnormal: dict[str, Judgement] = {}
    note: str = ""  # e.g. why a worklist case was skipped
    skipped: bool = False


def expand(sheet: Sheet, record: CaseRecord) -> dict[str, str]:
    """Project one record into a full {column: label} row for the sheet."""
    for finding in record.abnormal:
        if finding in sheet.derived:
            raise ValueError(
                f"{finding!r} is derived and must not be judged directly "
                f"(case {record.case_id})"
            )
        if finding not in sheet.label_columns:
            raise ValueError(
                f"{finding!r} is not a label column of {sheet.name} (case {record.case_id})"
            )

    labels = {column: NORMAL for column in sheet.label_columns}
    for finding in record.abnormal:
        labels[finding] = ABNORMAL
    for summary, inputs in sheet.derived.items():
        labels[summary] = ABNORMAL if any(labels[i] == ABNORMAL for i in inputs) else NORMAL
    return labels


def expand_all(sheet: Sheet, records: dict[str, CaseRecord]) -> dict[str, dict[str, str]]:
    """Project every scored record. Skipped cases are left out of the sheet entirely."""
    return {
        case_id: expand(sheet, record)
        for case_id, record in records.items()
        if not record.skipped
    }


def scored_count(records: dict[str, CaseRecord]) -> int:
    return sum(1 for r in records.values() if not r.skipped)


def load_evidence(sheet: Sheet) -> dict[str, CaseRecord]:
    path = evidence_path(sheet)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {case_id: CaseRecord(**record) for case_id, record in raw.items()}


def save_evidence(sheet: Sheet, records: dict[str, CaseRecord]) -> None:
    path = evidence_path(sheet)
    payload = {case_id: record.model_dump() for case_id, record in records.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
