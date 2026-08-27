"""The gold standard must be unreachable from the classification path.

Two independent checks. The source scan catches a module that names the gold standard;
the import check catches one that reaches it through another module. Either alone can
be worked around by accident, so both are here.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "classifier_multi"

# Every module that runs during classification. main.py is deliberately absent: it is
# the one module that names both sides, and it is covered by the import check instead.
CLASSIFICATION_MODULES = (
    "categories.py",
    "config.py",
    "prompt.py",
    "schemas.py",
    "llm.py",
    "classify.py",
    "csv_io.py",
    "run_fewshot.py",
)

FORBIDDEN = ("gold", "_originals", "dataset_gold_standard")


@pytest.mark.parametrize("module", CLASSIFICATION_MODULES)
def test_module_never_mentions_the_gold_standard(module):
    source = (PACKAGE / module).read_text(encoding="utf-8").lower()
    for term in FORBIDDEN:
        assert term not in source, (
            f"{module} mentions {term!r}. The gold standard is reachable only from "
            f"evaluate.py - see docs/superpowers/specs/"
            f"2026-08-19-classifier-multi-restructure-design.md"
        )


@pytest.mark.parametrize("module", CLASSIFICATION_MODULES)
def test_module_never_imports_evaluate(module):
    source = (PACKAGE / module).read_text(encoding="utf-8")
    assert "import evaluate" not in source
    assert "from classifier_multi.evaluate" not in source


def test_importing_classify_does_not_load_evaluate():
    """Proves main.py's lazy import holds, and that no transitive path exists."""
    code = (
        "import classifier_multi.classify, classifier_multi.csv_io, sys; "
        "print('classifier_multi.evaluate' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=PACKAGE.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "importing the classification path pulled in evaluate.py"
    )


def test_evaluate_is_allowed_to_mention_gold():
    """The counterpart: the boundary is a boundary, not a ban."""
    source = (PACKAGE / "evaluate.py").read_text(encoding="utf-8").lower()
    assert "gold" in source
