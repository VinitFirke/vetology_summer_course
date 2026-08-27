from pathlib import Path

from gold_standard.keywords import SEARCH_TERMS, candidate_pattern
from gold_standard.sheets import SHEETS


def test_every_judged_column_has_search_terms():
    for sheet in SHEETS.values():
        for column in sheet.judged_columns:
            assert SEARCH_TERMS.get(column), f"{column} has no search terms"


def test_derived_columns_are_not_mined():
    assert "diseased_lungs" not in SEARCH_TERMS


def test_candidate_pattern_matches_the_sign_not_the_column_name():
    """Reports say 'bronchial pattern', never 'bronchitis abnormal'."""
    pattern = candidate_pattern("bronchitis")
    assert pattern.search("There is a mild generalized bronchointerstitial pattern.")
    assert pattern.search("A minimal to mild diffuse bronchial pattern is present.")


def test_candidate_pattern_is_case_insensitive():
    assert candidate_pattern("pleural_effusion").search("PLEURAL EFFUSION is present")


def test_criteria_document_covers_every_column():
    text = Path("dataset_gold_standard/criteria.md").read_text(encoding="utf-8")
    for sheet in SHEETS.values():
        for column in sheet.label_columns:
            assert f"### `{column}`" in text, f"criteria.md has no section for {column}"
