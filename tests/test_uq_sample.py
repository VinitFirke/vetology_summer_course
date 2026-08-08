"""JSONL persistence and the resume arithmetic that makes a crash cost nothing."""

from uncertainty.sample import (
    append_record,
    labels_from_record,
    read_samples,
    replicate_shortfall,
)


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
