"""JSONL persistence, resume arithmetic, and the two calls that spend money."""

import pytest

from classifier import config as classifier_config
from classifier.prompt import load_prompt
from classifier.schemas import FindingName, RadiologyCase
from uncertainty.config import CE_PROMPT_FILE, TIERS
from uncertainty.sample import (
    append_record,
    elicit_confidence,
    labels_from_record,
    plan_run,
    read_samples,
    render_ce_messages,
    replicate_shortfall,
    sample_replicate,
)
from uncertainty.schemas import CaseConfidence, CaseLabels, FindingConfidence, FindingVote

CASE = RadiologyCase(
    case_id="12345",
    findings_text="A minimal to mild diffuse bronchial pattern is present.",
    conclusions_text="1. Minimal-mild diffuse bronchial pulmonary pattern.",
)


class FakeRaw:
    usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    response_metadata: dict = {}


class FakeModel:
    """Returns one canned parsed object, recording the messages it was sent."""

    def __init__(self, parsed):
        self.parsed = parsed
        self.messages = None

    def with_structured_output(self, _schema, include_raw=False):
        return self

    def invoke(self, messages):
        self.messages = messages
        return {"parsed": self.parsed, "raw": FakeRaw(), "parsing_error": None}


@pytest.fixture
def classification_prompt():
    return load_prompt(classifier_config.PROMPT_FILE)


@pytest.fixture
def ce_prompt():
    return load_prompt(CE_PROMPT_FILE)


def _record(case_id: str, replicate: int, **labels) -> dict:
    return {
        "provider": "kimi",
        "tier": "low",
        "case_id": case_id,
        "replicate": replicate,
        "labels": {"cardiomegaly": "normal", **labels},
        "logprobs": None,
        "usage": {"input_tokens": 1180, "output_tokens": 640},
        "timestamp": "2026-08-08T18:40:11",
    }


def test_append_then_read_round_trips(tmp_path):
    path = tmp_path / "samples.jsonl"
    append_record(path, _record("A", 1))
    append_record(path, _record("A", 2))

    samples = read_samples(path)

    assert list(samples) == ["A"]
    assert len(samples["A"]) == 2


def test_reading_a_missing_file_gives_an_empty_result(tmp_path):
    assert read_samples(tmp_path / "nothing.jsonl") == {}


def test_append_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "samples.jsonl"
    append_record(path, _record("A", 1))
    assert path.exists()


def test_records_are_ordered_by_replicate_number(tmp_path):
    """Replicate 1 is the SinglePass answer, so it must come back first."""
    path = tmp_path / "samples.jsonl"
    append_record(path, _record("A", 3))
    append_record(path, _record("A", 1))
    append_record(path, _record("A", 2))

    assert [r["replicate"] for r in read_samples(path)["A"]] == [1, 2, 3]


def test_a_corrupt_line_is_skipped_rather_than_killing_the_read(tmp_path):
    """A run killed mid-write can leave a half-line; that must not cost the whole file."""
    path = tmp_path / "samples.jsonl"
    append_record(path, _record("A", 1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"case_id": "B", "replic\n')
    append_record(path, _record("A", 2))

    samples = read_samples(path)

    assert len(samples["A"]) == 2
    assert "B" not in samples


def test_a_record_without_a_case_id_is_skipped(tmp_path):
    path = tmp_path / "samples.jsonl"
    append_record(path, {"replicate": 1, "labels": {}})
    append_record(path, _record("A", 1))

    assert list(read_samples(path)) == ["A"]


def test_shortfall_counts_what_is_still_missing(tmp_path):
    path = tmp_path / "samples.jsonl"
    append_record(path, _record("A", 1))
    append_record(path, _record("A", 2))

    shortfall = replicate_shortfall(read_samples(path), ["A", "B"], replicates=5)

    assert shortfall == {"A": 3, "B": 5}


def test_a_complete_case_is_absent_from_the_shortfall(tmp_path):
    path = tmp_path / "samples.jsonl"
    for replicate in range(1, 6):
        append_record(path, _record("A", replicate))

    assert replicate_shortfall(read_samples(path), ["A"], replicates=5) == {}


def test_shortfall_ignores_cases_not_asked_for(tmp_path):
    """--limit 2 must not schedule work for the other 48."""
    path = tmp_path / "samples.jsonl"
    append_record(path, _record("Z", 1))

    assert replicate_shortfall(read_samples(path), ["A"], replicates=5) == {"A": 5}


def test_labels_are_extracted_from_a_record():
    assert labels_from_record(_record("A", 1))["cardiomegaly"] == "normal"


# --- the two calls that spend money -------------------------------------------------


def test_ce_messages_include_every_proposed_label(ce_prompt):
    messages = render_ce_messages(
        ce_prompt, CASE, {"cardiomegaly": "abnormal", "pneumonia": "normal"}
    )

    text = messages[-1].content
    assert "cardiomegaly: abnormal" in text
    assert "pneumonia: normal" in text
    assert CASE.findings_text in text
    assert CASE.conclusions_text in text


def test_ce_messages_list_all_nineteen_findings(ce_prompt):
    """Even findings absent from the label dict get a line, defaulted to normal."""
    messages = render_ce_messages(ce_prompt, CASE, {"cardiomegaly": "abnormal"})

    text = messages[-1].content
    for finding in FindingName:
        assert f"- {finding.value}: " in text


def test_ce_messages_have_a_system_and_a_user_message(ce_prompt):
    messages = render_ce_messages(ce_prompt, CASE, {"cardiomegaly": "normal"})
    assert len(messages) == 2


def test_sample_replicate_builds_a_complete_record(classification_prompt):
    parsed = CaseLabels(
        case_id="ignored",
        findings=[FindingVote(finding=FindingName.cardiomegaly, label="abnormal")],
    )

    record = sample_replicate(
        FakeModel(parsed), classification_prompt, CASE, "kimi", "low", replicate=1
    )

    assert record["case_id"] == "12345"
    assert record["replicate"] == 1
    assert record["provider"] == "kimi"
    assert record["tier"] == "low"
    assert record["labels"]["cardiomegaly"] == "abnormal"
    assert record["usage"]["output_tokens"] == 50
    assert record["logprobs"] is None
    assert "timestamp" in record


def test_unreturned_findings_default_to_normal(classification_prompt):
    """Matches csv_io.build_label_row: silence means normal."""
    record = sample_replicate(
        FakeModel(CaseLabels(case_id="ignored", findings=[])),
        classification_prompt,
        CASE,
        "kimi",
        "low",
        replicate=1,
    )

    assert len(record["labels"]) == 19
    assert set(record["labels"].values()) == {"normal"}


def test_every_record_carries_all_nineteen_findings(classification_prompt):
    """build_rows indexes labels by name, so a partial dict would drop findings."""
    parsed = CaseLabels(
        case_id="ignored",
        findings=[FindingVote(finding=FindingName.pneumonia, label="abnormal")],
    )
    record = sample_replicate(
        FakeModel(parsed), classification_prompt, CASE, "kimi", "low", replicate=3
    )

    assert set(record["labels"]) == {f.value for f in FindingName}


def test_elicit_confidence_builds_a_score_record(ce_prompt):
    parsed = CaseConfidence(
        case_id="ignored",
        scores=[FindingConfidence(finding=FindingName.cardiomegaly, score=85)],
    )

    record = elicit_confidence(
        FakeModel(parsed), ce_prompt, CASE, {"cardiomegaly": "abnormal"}, "kimi", "low"
    )

    assert record["case_id"] == "12345"
    assert record["scores"]["cardiomegaly"] == 85
    assert record["provider"] == "kimi"
    assert record["tier"] == "low"


def test_unparseable_ce_raises_rather_than_writing_a_blank_score(ce_prompt):
    class Unparseable(FakeModel):
        def invoke(self, messages):
            return {"parsed": None, "raw": FakeRaw(), "parsing_error": "bad json"}

    with pytest.raises(RuntimeError, match="CE for case 12345 failed after 3 attempts"):
        elicit_confidence(
            Unparseable(None), ce_prompt, CASE, {"cardiomegaly": "abnormal"},
            "kimi", "low", max_attempts=3,
        )


def test_ce_retries_a_transient_failure_then_succeeds(ce_prompt):
    """Kimi intermittently fences its JSON in ```; one retry clears it.

    Regression test: CE originally had no retry, and lost ~11% of Kimi's responses to
    exactly this while the replicate calls, which do retry, lost none.
    """
    good = CaseConfidence(
        case_id="ignored",
        scores=[FindingConfidence(finding=FindingName.cardiomegaly, score=85)],
    )

    class FlakyThenFine(FakeModel):
        def __init__(self):
            super().__init__(good)
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"parsed": None, "raw": FakeRaw(), "parsing_error": "```{...}"}
            return {"parsed": good, "raw": FakeRaw(), "parsing_error": None}

    model = FlakyThenFine()
    record = elicit_confidence(
        model, ce_prompt, CASE, {"cardiomegaly": "abnormal"}, "kimi", "low"
    )

    assert model.calls == 2
    assert record["scores"]["cardiomegaly"] == 85


# --- run planning -------------------------------------------------------------------


def _cases(n: int) -> list[RadiologyCase]:
    return [
        RadiologyCase(case_id=str(i), findings_text="text", conclusions_text="text")
        for i in range(n)
    ]


def test_a_fresh_plan_requests_every_replicate(tmp_path):
    plan = plan_run("openai", ("low",), _cases(3), replicates=5, uq_dir=tmp_path)

    assert plan.work == {("low", "0"): 5, ("low", "1"): 5, ("low", "2"): 5}
    assert plan.ce_needed == {("low", "0"), ("low", "1"), ("low", "2")}
    assert plan.remaining_calls == 18  # 15 replicates + 3 CE


def test_a_plan_subtracts_work_already_on_disk(tmp_path):
    path = tmp_path / "samples_openai_low.jsonl"
    append_record(path, _record("0", 1))
    append_record(path, _record("0", 2))

    plan = plan_run("openai", ("low",), _cases(2), replicates=5, uq_dir=tmp_path)

    assert plan.work[("low", "0")] == 3
    assert plan.work[("low", "1")] == 5


def test_a_finished_case_drops_out_of_the_plan_entirely(tmp_path):
    samples = tmp_path / "samples_openai_low.jsonl"
    for replicate in range(1, 6):
        append_record(samples, _record("0", replicate))
    append_record(tmp_path / "ce_openai_low.jsonl", {"case_id": "0", "scores": {}})

    plan = plan_run("openai", ("low",), _cases(1), replicates=5, uq_dir=tmp_path)

    assert plan.work == {}
    assert plan.ce_needed == set()
    assert plan.is_empty
    assert plan.estimate.dollars == 0.0


def test_ce_alone_can_remain_outstanding(tmp_path):
    """Replicates complete but the CE call failed - only CE should be rescheduled."""
    samples = tmp_path / "samples_openai_low.jsonl"
    for replicate in range(1, 6):
        append_record(samples, _record("0", replicate))

    plan = plan_run("openai", ("low",), _cases(1), replicates=5, uq_dir=tmp_path)

    assert plan.work == {}
    assert plan.ce_needed == {("low", "0")}
    assert not plan.is_empty
    assert plan.remaining_calls == 1


def test_each_tier_is_planned_independently(tmp_path):
    for replicate in range(1, 6):
        append_record(tmp_path / "samples_openai_low.jsonl", _record("0", replicate))

    plan = plan_run("openai", ("low", "high"), _cases(1), replicates=5, uq_dir=tmp_path)

    assert ("low", "0") not in plan.work
    assert plan.work[("high", "0")] == 5


def test_the_plan_costs_only_the_remaining_work(tmp_path):
    full = plan_run("openai", TIERS, _cases(50), replicates=5, uq_dir=tmp_path)
    partial = plan_run("openai", ("low",), _cases(2), replicates=5, uq_dir=tmp_path)

    assert full.remaining_calls == 50 * 5 * 3 + 50 * 3
    assert full.estimate.dollars > partial.estimate.dollars
    assert partial.estimate.dollars > 0


def test_a_half_done_run_costs_less_than_a_fresh_one(tmp_path):
    fresh = plan_run("kimi", ("low",), _cases(4), replicates=5, uq_dir=tmp_path)
    for case_id in ("0", "1"):
        for replicate in range(1, 6):
            append_record(tmp_path / "samples_kimi_low.jsonl", _record(case_id, replicate))

    resumed = plan_run("kimi", ("low",), _cases(4), replicates=5, uq_dir=tmp_path)

    assert resumed.estimate.dollars < fresh.estimate.dollars
