"""The CLI must parse both subcommands and keep evaluate out of the classify path."""

import ast
from pathlib import Path

import pytest

from classifier_multi import main as main_module


def test_classify_subcommand_parses():
    args = main_module.build_parser().parse_args(
        ["classify", "--category", "canine_thorax",
         "--provider", "cloud_gemma", "--variant", "fewshot"]
    )
    assert args.command == "classify"
    assert args.category == "canine_thorax"
    assert args.provider == "cloud_gemma"
    assert args.variant == "fewshot"


def test_classify_variant_defaults_to_zeroshot():
    args = main_module.build_parser().parse_args(
        ["classify", "--category", "canine_thorax", "--provider", "cloud_gemma"]
    )
    assert args.variant == "zeroshot"


def test_evaluate_subcommand_parses():
    args = main_module.build_parser().parse_args(
        ["evaluate", "--category", "feline_thorax", "--variant", "fewshot"]
    )
    assert args.command == "evaluate"
    assert args.category == "feline_thorax"


def test_unknown_category_is_rejected():
    with pytest.raises(SystemExit):
        main_module.build_parser().parse_args(
            ["classify", "--category", "equine_thorax", "--provider", "cloud_gemma"]
        )


def test_unknown_variant_is_rejected():
    with pytest.raises(SystemExit):
        main_module.build_parser().parse_args(
            ["classify", "--category", "canine_thorax",
             "--provider", "cloud_gemma", "--variant", "tenshot"]
        )


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        main_module.build_parser().parse_args([])


def _main_ast():
    return ast.parse(Path(main_module.__file__).read_text(encoding="utf-8"))


def test_evaluate_is_not_imported_at_module_scope():
    """Parsed, not grepped: prose in the docstring must not affect the result."""
    for node in _main_ast().body:
        if isinstance(node, ast.Import):
            assert all("evaluate" not in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "evaluate" not in (node.module or "")
            assert all("evaluate" not in alias.name for alias in node.names)


def test_evaluate_is_imported_lazily_somewhere():
    """The counterpart: the import exists, it is just not at module scope."""
    tree = _main_ast()
    module_scope = set(tree.body)
    lazy = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node not in module_scope
        and (
            "evaluate" in (getattr(node, "module", "") or "")
            or any("evaluate" in alias.name for alias in node.names)
        )
    ]
    assert lazy, "expected evaluate to be imported inside a function"
