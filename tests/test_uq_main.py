"""The sampling loop, driven end to end against a fake model.

run_one_tier is where a bug costs real money - it decides how many calls to make and
which ones to skip. These tests exercise it offline, counting calls instead of paying
for them.
"""

import pytest

from classifier.schemas import FindingName, RadiologyCase
from uncertainty import config as uq_config
from uncertainty.sample import plan_run, read_samples
from uncertainty.schemas import CaseConfidence, CaseLabels, FindingConfidence, FindingVote

import uq_main


class FakeRaw:
    usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    response_metadata: dict = {}


class CountingModel:
    """Answers either schema, tallying how many calls of each kind it served."""

    def __init__(self):
        self.replicate_calls = 0
        self.ce_calls = 0
        self._schema = None

    def with_structured_output(self, schema, include_raw=False):
        self._schema = schema
        return self

    def invoke(self, _messages):
        if self._schema is CaseConfidence:
            self.ce_calls += 1
            parsed = CaseConfidence(
                case_id="ignored",
                scores=[FindingConfidence(finding=f, score=80) for f in FindingName],
            )
        else:
            self.replicate_calls += 1
            parsed = CaseLabels(
                case_id="ignored",
                findings=[FindingVote(finding=f, label="normal") for f in FindingName],
            )
        return {"parsed": parsed, "raw": FakeRaw(), "parsing_error": None}


def _cases(n: int) -> list[RadiologyCase]:
    return [
        RadiologyCase(case_id=f"case{i}", findings_text="text", conclusions_text="text")
        for i in range(n)
    ]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every UQ path at a temp dir and swap in a counting fake model."""
    monkeypatch.setattr(uq_config, "UQ_DIR", tmp_path)
    monkeypatch.setattr(
        uq_config, "samples_path", lambda p, t: tmp_path / f"samples_{p}_{t}.jsonl"
    )
    monkeypatch.setattr(uq_config, "ce_path", lambda p, t: tmp_path / f"ce_{p}_{t}.jsonl")
    monkeypatch.setattr(
        uq_config, "failures_path", lambda p, t: tmp_path / f"failures_{p}_{t}.jsonl"
    )
    monkeypatch.setattr(uq_main, "log_tier_usage", lambda *a, **k: None)

    model = CountingModel()
    monkeypatch.setattr(uq_main, "build_tier_model", lambda *a, **k: model)
    return tmp_path, model


def _run(tmp_path, cases, replicates=5, workers=1):
    plan = plan_run("openai", ("low",), cases, replicates, tmp_path)
    uq_main.run_one_tier("openai", "low", settings=None, cases=cases, plan=plan, workers=workers)
    return plan


def test_a_fresh_tier_makes_exactly_the_planned_calls(sandbox):
    tmp_path, model = sandbox
    cases = _cases(3)

    _run(tmp_path, cases)

    assert model.replicate_calls == 15  # 3 cases x 5 replicates
    assert model.ce_calls == 3  # one per case


def test_every_replicate_lands_on_disk(sandbox):
    tmp_path, model = sandbox
    cases = _cases(3)

    _run(tmp_path, cases)

    samples = read_samples(tmp_path / "samples_openai_low.jsonl")
    assert len(samples) == 3
    assert {len(v) for v in samples.values()} == {5}
    assert [r["replicate"] for r in samples["case0"]] == [1, 2, 3, 4, 5]


def test_a_second_run_makes_no_calls_at_all(sandbox):
    """The resume path must not re-charge for completed work."""
    tmp_path, model = sandbox
    cases = _cases(3)
    _run(tmp_path, cases)
    before = (model.replicate_calls, model.ce_calls)

    _run(tmp_path, cases)

    assert (model.replicate_calls, model.ce_calls) == before


def test_a_partial_run_charges_only_for_the_shortfall(sandbox):
    tmp_path, model = sandbox
    cases = _cases(3)

    _run(tmp_path, cases, replicates=3)
    after_first = model.replicate_calls
    _run(tmp_path, cases, replicates=5)

    assert after_first == 9  # 3 cases x 3
    assert model.replicate_calls == 9 + 6  # then 2 more each, not 15 again


def test_replicate_numbering_continues_rather_than_restarting(sandbox):
    tmp_path, _ = sandbox
    cases = _cases(1)

    _run(tmp_path, cases, replicates=3)
    _run(tmp_path, cases, replicates=5)

    samples = read_samples(tmp_path / "samples_openai_low.jsonl")
    assert [r["replicate"] for r in samples["case0"]] == [1, 2, 3, 4, 5]


def test_a_failing_replicate_is_logged_and_the_run_continues(sandbox, monkeypatch):
    tmp_path, _ = sandbox
    cases = _cases(3)

    calls = {"n": 0}

    def flaky(model, prompt, case, provider, tier, replicate):
        calls["n"] += 1
        if case.case_id == "case1":
            raise RuntimeError("provider exploded")
        return {
            "provider": provider, "tier": tier, "case_id": case.case_id,
            "replicate": replicate, "labels": {f.value: "normal" for f in FindingName},
            "logprobs": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            "timestamp": "2026-08-08T00:00:00",
        }

    monkeypatch.setattr(uq_main, "sample_replicate", flaky)
    _run(tmp_path, cases)

    samples = read_samples(tmp_path / "samples_openai_low.jsonl")
    failures = (tmp_path / "failures_openai_low.jsonl").read_text(encoding="utf-8")
    assert set(samples) == {"case0", "case2"}  # case1 never landed
    assert failures.count("provider exploded") == 5  # all 5 of its replicates logged
    assert calls["n"] == 15  # the run did not abort early


def test_ce_is_skipped_for_a_case_with_no_surviving_replicates(sandbox, monkeypatch):
    """CE rates replicate 1's labels, so with no replicate there is nothing to rate."""
    tmp_path, model = sandbox
    cases = _cases(2)

    def always_fails(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(uq_main, "sample_replicate", always_fails)
    _run(tmp_path, cases)

    assert model.ce_calls == 0


def test_the_thread_pool_produces_the_same_result_as_one_worker(sandbox):
    tmp_path, model = sandbox
    cases = _cases(4)

    _run(tmp_path, cases, workers=4)

    samples = read_samples(tmp_path / "samples_openai_low.jsonl")
    assert len(samples) == 4
    assert {len(v) for v in samples.values()} == {5}
    assert model.replicate_calls == 20
