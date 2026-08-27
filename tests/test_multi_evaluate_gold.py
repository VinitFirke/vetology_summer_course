"""evaluate.py is the only gold-aware module, and _originals is off limits."""

from pathlib import Path

import pytest

from classifier_multi import evaluate
from classifier_multi.categories import get_category


def test_gold_csv_path_names_the_category_file():
    path = evaluate.gold_csv_path(get_category("canine_thorax"))
    assert path.name == "canine_thorax_gold_standard.csv"
    assert path.parent == evaluate.GOLD_DIR


def test_vhs_alias_survived_the_move_from_categories():
    """The manual scorers wrote the vertebral heart score as normal/enlarged."""
    aliases = evaluate.gold_value_aliases(get_category("canine_thorax"))
    assert aliases["vhs"]["enlarged"] == "abnormal"


def test_categories_without_aliases_get_an_empty_map():
    assert evaluate.gold_value_aliases(get_category("canine_abdomen")) == {}


def test_originals_is_refused():
    forbidden = evaluate.GOLD_DIR / "_originals" / "anything.csv"
    with pytest.raises(evaluate.OriginalsAccessError):
        evaluate.reject_originals(forbidden)


def test_originals_is_refused_anywhere_in_the_path():
    with pytest.raises(evaluate.OriginalsAccessError):
        evaluate.reject_originals(Path("a") / "_originals" / "b" / "c.csv")


def test_ordinary_gold_path_is_allowed():
    evaluate.reject_originals(evaluate.gold_csv_path(get_category("feline_thorax")))


def test_confusion_matrix_path_carries_the_variant():
    zero = evaluate.confusion_matrix_path(get_category("feline_thorax"), "zeroshot")
    few = evaluate.confusion_matrix_path(get_category("feline_thorax"), "fewshot")
    assert zero != few
    assert few.name == "confusion_matrix_feline_thorax.xlsx"
    assert few.parent.name == "fewshot"


def test_vhs_aliases_reach_the_scoring():
    """An 'enlarged' gold value must count as a hit against an 'abnormal' prediction."""
    aliases = evaluate.gold_value_aliases(get_category("canine_thorax"))
    counts = evaluate.confusion_counts(
        [("enlarged", "abnormal"), ("normal", "normal")], aliases["vhs"]
    )
    assert counts.true_positive == 1
    assert counts.true_negative == 1
    assert counts.false_positive == 0
    assert counts.false_negative == 0
