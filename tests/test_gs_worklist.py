from gold_standard.sheets import SHEETS, get_sheet
from gold_standard.worklist import build_worklist


def test_worklist_has_no_duplicate_cases():
    for sheet in SHEETS.values():
        items = build_worklist(sheet)
        ids = [i.case_id for i in items]
        assert len(set(ids)) == len(ids), f"{sheet.name} repeats a case"


def test_worklist_is_at_least_the_target_length():
    for sheet in SHEETS.values():
        assert len(build_worklist(sheet)) >= sheet.target_cases


def test_coverage_candidates_come_before_random_fill():
    items = build_worklist(get_sheet("canine_thorax"))
    reasons = [i.reason for i in items]
    last_coverage = max(i for i, r in enumerate(reasons) if r.startswith("coverage:"))
    first_random = min(i for i, r in enumerate(reasons) if r == "random")
    assert last_coverage < first_random


def test_every_judged_column_is_represented_among_candidates():
    for sheet in SHEETS.values():
        items = build_worklist(sheet)
        covered = {i.reason.removeprefix("coverage:") for i in items if ":" in i.reason}
        missing = [c for c in sheet.judged_columns if c not in covered]
        assert missing == [], f"{sheet.name}: no candidates found for {missing}"


def test_selection_is_reproducible_for_a_fixed_seed():
    sheet = get_sheet("feline_thorax")
    assert [i.case_id for i in build_worklist(sheet, seed=7)] == [
        i.case_id for i in build_worklist(sheet, seed=7)
    ]


def test_a_different_seed_changes_the_random_tail():
    sheet = get_sheet("feline_thorax")
    tail_a = [i.case_id for i in build_worklist(sheet, seed=1) if i.reason == "random"]
    tail_b = [i.case_id for i in build_worklist(sheet, seed=2) if i.reason == "random"]
    assert tail_a != tail_b
