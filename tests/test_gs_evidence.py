import pytest

from gold_standard.evidence import (
    CaseRecord,
    Judgement,
    expand,
    expand_all,
    load_evidence,
    save_evidence,
)
from gold_standard.sheets import get_sheet


def _record(**abnormal):
    return CaseRecord(
        case_id="123",
        abnormal={
            k: Judgement(evidence=f"quote for {k}", reasoning="because") for k in abnormal
        },
    )


def test_unlisted_findings_expand_to_normal():
    sheet = get_sheet("canine_abdomen")
    labels = expand(sheet, _record(ascites=True))
    assert labels["ascites"] == "abnormal"
    assert labels["hepatomegaly"] == "normal"
    assert set(labels) == set(sheet.label_columns)


def test_diseased_lungs_is_abnormal_when_a_lung_finding_is():
    sheet = get_sheet("canine_thorax")
    assert expand(sheet, _record(bronchitis=True))["diseased_lungs"] == "abnormal"


def test_diseased_lungs_stays_normal_for_a_non_lung_finding():
    sheet = get_sheet("canine_thorax")
    labels = expand(sheet, _record(pleural_effusion=True, cardiomegaly=True))
    assert labels["diseased_lungs"] == "normal"


def test_feline_alveolar_column_rolls_into_diseased_lungs():
    sheet = get_sheet("feline_thorax")
    record = _record(**{"Alveolar_interstitial_pattern": True})
    assert expand(sheet, record)["diseased_lungs"] == "abnormal"


def test_a_derived_column_cannot_be_judged_directly():
    sheet = get_sheet("canine_thorax")
    with pytest.raises(ValueError, match="derived"):
        expand(sheet, _record(diseased_lungs=True))


def test_an_unknown_finding_is_rejected():
    sheet = get_sheet("canine_abdomen")
    with pytest.raises(ValueError, match="not a label column"):
        expand(sheet, _record(cardiomegaly=True))


def test_skipped_cases_are_not_projected_into_the_sheet():
    sheet = get_sheet("canine_abdomen")
    records = {
        "1": _record(ascites=True),
        "2": CaseRecord(case_id="2", skipped=True, note="not an abdominal study"),
    }
    assert set(expand_all(sheet, records)) == {"1"}


def test_evidence_survives_a_save_and_load(tmp_path, monkeypatch):
    import gold_standard.evidence as module

    sheet = get_sheet("canine_abdomen")
    monkeypatch.setattr(module, "evidence_path", lambda s: tmp_path / f"{s.name}.json")
    save_evidence(sheet, {"123": _record(ascites=True)})
    loaded = load_evidence(sheet)
    assert loaded["123"].abnormal["ascites"].evidence == "quote for ascites"


def test_loading_a_missing_store_gives_an_empty_dict(tmp_path, monkeypatch):
    import gold_standard.evidence as module

    sheet = get_sheet("canine_abdomen")
    monkeypatch.setattr(module, "evidence_path", lambda s: tmp_path / "absent.json")
    assert load_evidence(sheet) == {}
