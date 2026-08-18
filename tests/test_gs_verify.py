from gold_standard.sheets import ABNORMAL, NORMAL, get_sheet
from gold_standard.verify import coverage_counts, rollup_problems, scored_case_ids


def test_rollup_problems_flags_a_contradicting_summary():
    sheet = get_sheet("canine_thorax")
    row = {c: NORMAL for c in sheet.label_columns}
    row["bronchitis"] = ABNORMAL
    row["diseased_lungs"] = NORMAL  # contradicts the roll-up
    assert rollup_problems(sheet, {"1": row}) == [
        "case 1: diseased_lungs is normal but bronchitis is abnormal"
    ]


def test_rollup_problems_flags_an_unsupported_summary():
    sheet = get_sheet("canine_thorax")
    row = {c: NORMAL for c in sheet.label_columns}
    row["diseased_lungs"] = ABNORMAL
    assert rollup_problems(sheet, {"1": row}) == [
        "case 1: diseased_lungs is abnormal but no lung finding is"
    ]


def test_rollup_problems_accepts_a_consistent_row():
    sheet = get_sheet("canine_thorax")
    row = {c: NORMAL for c in sheet.label_columns}
    row["bronchitis"] = ABNORMAL
    row["diseased_lungs"] = ABNORMAL
    assert rollup_problems(sheet, {"1": row}) == []


def test_pleural_effusion_alone_does_not_make_lungs_diseased():
    sheet = get_sheet("canine_thorax")
    row = {c: NORMAL for c in sheet.label_columns}
    row["pleural_effusion"] = ABNORMAL
    assert rollup_problems(sheet, {"1": row}) == []


def test_scored_and_coverage_read_the_live_sheet():
    sheet = get_sheet("canine_abdomen")
    assert isinstance(scored_case_ids(sheet), list)
    counts = coverage_counts(sheet)
    assert set(counts) == set(sheet.label_columns)
