"""Collecting replicate samples, and the JSONL they are written to.

Append-only JSONL is what makes a run resumable: count what is already on disk and
request only the shortfall. A crash at case 40 then costs nothing, which matters when
one provider's entire budget is a single run.
"""

import json
import threading
from pathlib import Path

_WRITE_LOCK = threading.Lock()


def append_record(path: Path, record: dict) -> None:
    """Append one JSON object as a line. Thread-safe; creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_samples(path: Path) -> dict[str, list[dict]]:
    """Read a samples file into {case_id: [records ordered by replicate]}.

    Unparseable lines are skipped. A run killed mid-write can leave a truncated final
    line, and that must not make the rest of an expensive file unreadable.
    """
    if not path.exists():
        return {}

    by_case: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        case_id = record.get("case_id")
        if case_id is None:
            continue
        by_case.setdefault(case_id, []).append(record)

    for records in by_case.values():
        records.sort(key=lambda r: r.get("replicate", 0))

    return by_case


def replicate_shortfall(
    existing: dict[str, list[dict]],
    case_ids: list[str],
    replicates: int,
) -> dict[str, int]:
    """How many more replicates each case still needs. Complete cases are omitted."""
    shortfall: dict[str, int] = {}
    for case_id in case_ids:
        missing = replicates - len(existing.get(case_id, []))
        if missing > 0:
            shortfall[case_id] = missing
    return shortfall


def labels_from_record(record: dict) -> dict[str, str]:
    """The {finding: label} mapping stored on one replicate record."""
    return record["labels"]
