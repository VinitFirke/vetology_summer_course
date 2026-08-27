"""Output paths must separate prompt variants and never touch the gold standard."""

import pytest

from classifier_multi import config
from classifier_multi.categories import get_category

CATEGORY = get_category("canine_thorax")


def test_short_name_strips_the_cloud_prefix():
    assert config.short_name("cloud_gemma") == "gemma"
    assert config.short_name("cloud_nemotron") == "nemotron"


def test_predictions_path_is_named_as_specified():
    path = config.predictions_path(CATEGORY, "cloud_gemma", "fewshot")
    assert path.name == "canine_thorax_classified_gemma.csv"
    assert path.parent.name == "fewshot"
    assert path.parent.parent == config.DATA_DIR


def test_variant_separates_predictions():
    zero = config.predictions_path(CATEGORY, "cloud_gemma", "zeroshot")
    few = config.predictions_path(CATEGORY, "cloud_gemma", "fewshot")
    assert zero != few, "a few-shot run must not overwrite the zero-shot baseline"


def test_predictions_never_collide_with_the_input_csv():
    for variant in config.VARIANTS:
        for provider in config.PROVIDERS:
            path = config.predictions_path(CATEGORY, provider, variant)
            assert path != config.input_csv_path(CATEGORY)


def test_reasoning_path_sits_beside_its_predictions():
    predictions = config.predictions_path(CATEGORY, "cloud_qwen", "fewshot")
    reasoning = config.reasoning_path(CATEGORY, "cloud_qwen", "fewshot")
    assert reasoning.parent == predictions.parent
    assert reasoning.name == "canine_thorax_reasoning_qwen.json"


@pytest.mark.parametrize(
    "variant,expected",
    [("zeroshot", "canine_thorax.json"), ("fewshot", "canine_thorax_fewshot.json")],
)
def test_prompt_path_selects_the_variant_file(variant, expected):
    assert config.prompt_path(CATEGORY, variant).name == expected


def test_prompt_path_defaults_to_zeroshot():
    assert config.prompt_path(CATEGORY).name == "canine_thorax.json"


def test_every_variant_prompt_file_exists():
    """A variant that names a missing file would fail only at run time."""
    for name in ("feline_thorax", "canine_thorax", "canine_abdomen"):
        category = get_category(name)
        for variant in config.VARIANTS:
            assert config.prompt_path(category, variant).exists()


def test_config_has_no_gold_standard_surface():
    assert not hasattr(config, "GOLD_DIR")
    assert not hasattr(config, "gold_csv_path")
