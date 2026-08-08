"""Collecting replicate samples, and the JSONL they are written to.

Append-only JSONL is what makes a run resumable: count what is already on disk and
request only the shortfall. A crash at case 40 then costs nothing, which matters when
one provider's entire budget is a single run.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from classifier.classify import classify_case
from classifier.prompt import Prompt
from classifier.schemas import NORMAL, FindingName, RadiologyCase
from uncertainty.schemas import CaseConfidence, CaseLabels

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


def render_ce_messages(
    prompt: Prompt,
    case: RadiologyCase,
    labels: dict[str, str],
) -> list[BaseMessage]:
    """Build the step-2 CE messages: the case, plus the labels being rated.

    Two-step CE resubmits the whole question-and-answer pair, which the paper found
    outperforms asking for the answer and the confidence in one shot.

    All 19 findings are listed even if `labels` is missing some, so the model always
    rates the same set in the same order that FindingName declares them.
    """
    proposed = "\n".join(
        f"- {finding.value}: {labels.get(finding.value, NORMAL)}" for finding in FindingName
    )
    user_text = prompt.user_template.format(
        case_id=case.case_id,
        findings=case.findings_text,
        conclusions=case.conclusions_text,
        proposed_labels=proposed,
        n_findings=len(FindingName),
    )
    return [SystemMessage(content=prompt.system), HumanMessage(content=user_text)]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sample_replicate(
    model: BaseChatModel,
    prompt: Prompt,
    case: RadiologyCase,
    provider: str,
    tier: str,
    replicate: int,
) -> dict:
    """One replicate call, as a JSONL-ready record.

    Reuses classify_case's retry loop by handing it the slim schema, so there is exactly
    one backoff implementation in the codebase.
    """
    result = classify_case(model, prompt, case, schema=CaseLabels)
    returned = {vote.finding.value: vote.label for vote in result.classification.findings}

    # Any finding the model omitted falls back to normal, matching csv_io.build_label_row.
    # Writing all 19 every time means build_rows can index by name without checking.
    labels = {finding.value: returned.get(finding.value, NORMAL) for finding in FindingName}

    return {
        "provider": provider,
        "tier": tier,
        "case_id": case.case_id,
        "replicate": replicate,
        "labels": labels,
        "logprobs": None,
        "usage": result.usage.model_dump(),
        "timestamp": _timestamp(),
    }


def elicit_confidence(
    model: BaseChatModel,
    ce_prompt: Prompt,
    case: RadiologyCase,
    labels: dict[str, str],
    provider: str,
    tier: str,
) -> dict:
    """Step 2 of CE: ask the model to rate the labels it just produced.

    Raises rather than returning a partial record, so the caller logs a failure instead
    of writing scores that were never actually elicited.
    """
    messages = render_ce_messages(ce_prompt, case, labels)
    structured = model.with_structured_output(CaseConfidence, include_raw=True)
    response = structured.invoke(messages)

    parsed = response.get("parsed")
    if parsed is None:
        raise RuntimeError(
            f"CE for case {case.case_id} was unparseable: {response.get('parsing_error')}"
        )

    raw = response.get("raw")
    usage = getattr(raw, "usage_metadata", None) or {}

    return {
        "provider": provider,
        "tier": tier,
        "case_id": case.case_id,
        "scores": {item.finding.value: item.score for item in parsed.scores},
        "usage": usage,
        "timestamp": _timestamp(),
    }
