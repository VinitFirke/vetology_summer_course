"""diseased_lungs is derived, so it must never contradict its inputs."""

from classifier.csv_io import build_label_row, derive_diseased_lungs
from classifier.schemas import (
    ABNORMAL,
    DISEASED_LUNGS,
    DISEASED_LUNGS_INPUTS,
    NORMAL,
    FindingName,
)


def all_normal() -> dict[str, str]:
    return {finding.value: NORMAL for finding in FindingName}


def test_all_normal_gives_normal_lungs():
    assert derive_diseased_lungs(all_normal()) == NORMAL


def test_any_single_lung_finding_flips_it():
    for finding in DISEASED_LUNGS_INPUTS:
        labels = all_normal()
        labels[finding.value] = ABNORMAL
        assert derive_diseased_lungs(labels) == ABNORMAL, finding.value


def test_non_lung_findings_do_not_flip_it():
    excluded = [f for f in FindingName if f not in DISEASED_LUNGS_INPUTS]
    assert excluded, "expected some findings to be excluded from the rollup"
    for finding in excluded:
        labels = all_normal()
        labels[finding.value] = ABNORMAL
        assert derive_diseased_lungs(labels) == NORMAL, finding.value


def test_cardiomegaly_does_not_make_lungs_diseased():
    labels = all_normal()
    labels[FindingName.cardiomegaly.value] = ABNORMAL
    labels[FindingName.pleural_effusion.value] = ABNORMAL
    assert derive_diseased_lungs(labels) == NORMAL


def test_build_label_row_fills_all_20_columns():
    row = build_label_row({FindingName.bronchitis.value: ABNORMAL})
    assert len(row) == 20
    assert row[FindingName.bronchitis.value] == ABNORMAL
    assert row[DISEASED_LUNGS] == ABNORMAL


def test_missing_findings_fall_back_to_normal():
    """Silence means normal, so a model omitting a finding must not blank the cell."""
    row = build_label_row({})
    assert set(row.values()) == {NORMAL}
    assert len(row) == 20
