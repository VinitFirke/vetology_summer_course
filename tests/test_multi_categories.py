"""Category definitions must match the CSVs they claim to describe."""

import pandas as pd
import pytest

from classifier_multi import config
from classifier_multi.categories import CATEGORIES, Category, get_category


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_input_csv_exists(name):
    category = get_category(name)
    assert config.input_csv_path(category).exists(), (
        f"{category.input_csv} not found in {config.DATA_DIR}"
    )


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_label_columns_match_csv_header_in_order(name):
    """The label columns are the CSV's trailing columns, in file order."""
    category = get_category(name)
    df = pd.read_csv(config.input_csv_path(category), dtype=str, nrows=0)
    assert list(df.columns)[10:] == list(category.label_columns)


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_derived_columns_are_real_columns(name):
    category = get_category(name)
    for derived, sources in category.derived.items():
        assert derived in category.label_columns
        for source in sources:
            assert source in category.label_columns, f"{source} not a {name} column"


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_asked_findings_excludes_derived(name):
    category = get_category(name)
    for derived in category.derived:
        assert derived not in category.asked_findings


def test_category_has_no_gold_standard_fields():
    """Gold knowledge lives in evaluate.py, not in the category definition."""
    fields = set(Category.model_fields)
    assert "gold_csv" not in fields
    assert "gold_value_aliases" not in fields
