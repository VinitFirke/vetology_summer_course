# classifier_multi Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repoint `classifier_multi/` at the three `*_TOBE_classified(in).csv` files in `dataset_LLM_classification/`, and make the separation from `dataset_gold_standard/` structural and test-enforced.

**Architecture:** The package keeps its current module layout. Two gold-standard couplings (`Category.gold_csv`, `Category.gold_value_aliases`) and one path constant (`config.GOLD_DIR`) move into `evaluate.py`, which becomes the only gold-aware module. A new `main.py` exposes `classify` and `evaluate` as separate subcommands and imports `evaluate` lazily, so a classification run never loads a gold-aware module. Prompt variant becomes a subfolder under the data directory so a few-shot run cannot overwrite the zero-shot baseline.

**Tech Stack:** Python 3.12, pydantic v2, pandas, langchain-core, pytest, argparse (stdlib).

**Spec:** `docs/superpowers/specs/2026-08-19-classifier-multi-restructure-design.md`

## Global Constraints

- The gold standard is read only by `evaluate.py`. No other module in `classifier_multi/` may contain the strings `gold`, `_originals`, or `dataset_gold_standard`. `main.py` is the one documented exception and imports `evaluate` lazily inside its `evaluate` handler.
- `dataset_gold_standard/_originals/` must never be read. Any resolved path containing that directory name raises.
- Only `Findings (original radiologist report)` and `Conclusions (original radiologist report)` are fed to the model. The `(AI report)` columns and `Recommendations` are never read.
- `dataset_LLM_classification/*_TOBE_classified(in).csv` are inputs. Never write to them.
- Provider keys keep the `cloud_` prefix. The prefix is stripped only when building filenames.
- Prompt variants are exactly `"zeroshot"` and `"fewshot"`.
- Label-column order in `categories.py` must equal the CSV header order.

## Pre-existing failures — do not "fix" these

Before starting, run `python -m pytest tests/ -q` and note the baseline. These already fail and are **out of scope**:

- `tests/test_gs_*.py`, `tests/test_uq_*.py` — collection errors, `ModuleNotFoundError: No module named 'gold_standard'` / `'uncertainty'`. Those packages are not present.
- `tests/test_schema.py::test_real_csv_matches_the_enum` — `FileNotFoundError`. It belongs to the single-category `classifier` package and points at a dataset file that no longer exists.

Every existing test file targets `classifier`, not `classifier_multi`. Nothing in this plan changes the `classifier` package.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `classifier_multi/categories.py` | What differs per study type | Modify — drop 2 gold fields, repoint `input_csv`, reorder feline |
| `classifier_multi/config.py` | Classification paths, providers, keys | Modify — drop gold, add variant-aware output paths |
| `classifier_multi/evaluate.py` | Scoring against gold | Modify — takes ownership of all gold knowledge |
| `classifier_multi/main.py` | CLI entry point | Create |
| `tests/test_multi_categories.py` | Category definitions match the CSVs | Create |
| `tests/test_multi_config.py` | Path construction | Create |
| `tests/test_multi_evaluate_gold.py` | Gold path resolution and `_originals` refusal | Create |
| `tests/test_gold_boundary.py` | The constraint itself | Create |
| `tests/test_multi_prompt.py` | Few-shot message rendering | Create |

Test files are prefixed `test_multi_` so they cannot be confused with the existing `classifier` tests of the same concern (`tests/test_prompt.py` etc.).

---

### Task 1: Category definitions repointed and gold-free

**Files:**
- Modify: `classifier_multi/categories.py`
- Test: `tests/test_multi_categories.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Category` without `gold_csv` or `gold_value_aliases`; `Category.input_csv` naming the `*_TOBE_classified(in).csv` files; `FELINE_THORAX.label_columns` in CSV header order. `Category.asked_findings` and `Category.prompt_filename(variant)` keep their current signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_categories.py`:

```python
"""Category definitions must match the CSVs they claim to describe."""

import pandas as pd
import pytest

from classifier_multi import config
from classifier_multi.categories import CATEGORIES, get_category


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
    fields = set(get_category("canine_thorax").model_fields)
    assert "gold_csv" not in fields
    assert "gold_value_aliases" not in fields
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_categories.py -v`
Expected: FAIL — `test_input_csv_exists` fails for all three (files named `*_manual_final_50_appended_*` do not exist), and `test_category_has_no_gold_standard_fields` fails because both fields are still present.

- [ ] **Step 3: Rewrite the Category model**

In `classifier_multi/categories.py`, replace the `Category` class docstring and field block. Delete the `gold_value_aliases` paragraph from the docstring and both fields:

```python
class Category(BaseModel):
    """One study type: its data file, its findings, and its prompt.

    `derived` maps a column the model is never asked about to the columns that roll
    up into it. A derived column is abnormal when any of its inputs is abnormal.
    Deriving in code rather than asking the model means the summary column can never
    contradict the findings underneath it.

    Nothing here knows the gold standard exists. Gold filenames and the vocabulary
    the manual scorers used live in evaluate.py, which is the only module allowed to
    read dataset_gold_standard/.
    """

    name: CategoryName
    input_csv: str
    label_columns: tuple[str, ...]
    derived: dict[str, tuple[str, ...]] = {}

    @property
    def asked_findings(self) -> tuple[str, ...]:
        """The findings put to the model: every label column that is not derived."""
        return tuple(c for c in self.label_columns if c not in self.derived)

    def prompt_filename(self, variant: str | None = None) -> str:
        """The prompt file for this category, optionally a named variant of it.

        The zero-shot prompt is the bare category name; a variant such as "fewshot"
        is a suffix, so the baseline file is never the one edited when a new prompt
        style is tried and the two stay comparable run to run.
        """
        stem = self.name if variant is None else f"{self.name}_{variant}"
        return f"{stem}.json"
```

- [ ] **Step 4: Repoint the three category instances**

Make three edits per instance, leaving every existing `label_columns` tuple in place for now:

1. Replace the `input_csv=` value:
   - `FELINE_THORAX` &rarr; `input_csv="feline_thorax_TOBE_classified(in).csv",`
   - `CANINE_THORAX` &rarr; `input_csv="canine_thorax_TOBE_classified(in).csv",`
   - `CANINE_ABDOMEN` &rarr; `input_csv="canine_abdomen_TOBE_classified(in).csv",`
2. Delete the `gold_csv="..."` line from all three instances.
3. Delete this line from `CANINE_THORAX`, together with the two comment lines directly above it:

```python
    # The manual scorers wrote the vertebral heart score as normal/enlarged. Both
    # spellings of the positive class are listed so casing in the sheet cannot matter.
    gold_value_aliases={"vhs": {"enlarged": "abnormal"}},
```

Task 3 restores that knowledge in `evaluate.py`. Do not do this step without doing Task 3 — the `vhs` alias is load-bearing for canine thorax scoring and deleting it alone would silently mis-score every VHS row.

`CANINE_THORAX` and `CANINE_ABDOMEN` label columns already match their CSV headers exactly and must not be touched. Only feline changes, in the next step.

- [ ] **Step 5: Reorder the feline label columns to the CSV header order**

Replace `FELINE_THORAX`'s `label_columns` with exactly this tuple. Six positions differ from what was there:

```python
    label_columns=(
        "pulmonary_nodules",
        "esophagitis",
        "pneumonia",
        "bronchitis",
        "interstitial",
        "diseased_lungs",
        "hypo_plastic_trachea",
        "cardiomegaly",
        "pleural_effusion",
        "perihilar_infiltrate",
        "focal_caudodorsal_lung",
        "right_sided_cardiomegaly",
        "focal_perihilar",
        "left_sided_cardiomegaly",
        "bronchiectasis",
        "pulmonary_vessel_enlargement",
        "thoracic_lymphadenopathy",
        "pulmonary_hypoinflation",
        "pericardial_effusion",
        "Alveolar_interstitial_pattern",
    ),
```

Also update the module docstring's second paragraph, which still claims the tuples are copied from the CSV headers "because csv_io writes them back by name and verify_columns fails loudly if the spreadsheet and this file ever drift apart" — that sentence stays true, but append:

```
The order is the CSV's own header order. It sets the order findings are put to the
model and the order of the answer schema; it does not affect the written file, which
csv_io fills in by column name.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_categories.py -v`
Expected: PASS, 13 tests (4 parametrized × 3 categories, plus 1).

- [ ] **Step 7: Commit**

```bash
git add classifier_multi/categories.py tests/test_multi_categories.py
git commit -m "refactor(multi): repoint categories at TOBE_classified CSVs, drop gold fields"
```

---

### Task 2: Classification paths in config

**Files:**
- Modify: `classifier_multi/config.py`
- Test: `tests/test_multi_config.py`

**Interfaces:**
- Consumes: `Category` from Task 1 (no `gold_csv` field).
- Produces: `Variant` (`Literal["zeroshot", "fewshot"]`), `VARIANTS: tuple[Variant, ...]`, `short_name(provider) -> str`, `predictions_path(category, provider, variant) -> Path`, `reasoning_path(category, provider, variant) -> Path`, `prompt_path(category, variant="zeroshot") -> Path`. `gold_csv_path` and `GOLD_DIR` no longer exist. `confusion_matrix_path` is removed from this module — Task 3 defines it in `evaluate.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_config.py`:

```python
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


def test_config_has_no_gold_standard_surface():
    assert not hasattr(config, "GOLD_DIR")
    assert not hasattr(config, "gold_csv_path")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_config.py -v`
Expected: FAIL — `AttributeError: module 'classifier_multi.config' has no attribute 'short_name'`, and `test_config_has_no_gold_standard_surface` fails because `GOLD_DIR` still exists.

- [ ] **Step 3: Delete the gold surface from config**

In `classifier_multi/config.py`, delete these three things:

```python
GOLD_DIR = PROJECT_ROOT / "dataset_gold_standard"     # delete this line

def gold_csv_path(category: Category) -> Path:        # delete this function
    return GOLD_DIR / category.gold_csv

def confusion_matrix_path(category: Category) -> Path:   # delete, moves to evaluate.py
    """Written next to the predictions, so the gold standard folder stays read-only."""
    return DATA_DIR / f"confusion_matrix_{category.name}.xlsx"
```

Update the module docstring: its opening list says paths are "a function of a Category". Append a third bullet:

```
  * nothing here knows the gold standard exists. Its location, its filenames and the
    vocabulary its scorers used belong to evaluate.py, so a classification run cannot
    reach the answers even by accident.
```

- [ ] **Step 4: Add the variant and path helpers**

Replace the old `output_csv_path` and `reasoning_json_path` functions with:

```python
# The two prompt styles. The variant is a directory rather than a filename suffix, so
# a run of one style can never land on top of the other's results.
Variant = Literal["zeroshot", "fewshot"]

VARIANTS: tuple[Variant, ...] = ("zeroshot", "fewshot")


def short_name(provider: Provider) -> str:
    """`cloud_gemma` -> `gemma`. Filenames only; provider keys keep the prefix."""
    return provider.removeprefix("cloud_")


def variant_dir(variant: Variant) -> Path:
    """One directory per prompt style, under the data directory."""
    return DATA_DIR / variant


def predictions_path(category: Category, provider: Provider, variant: Variant) -> Path:
    """Where one run's labelled CSV goes. Never the input file."""
    return variant_dir(variant) / (
        f"{category.name}_classified_{short_name(provider)}.csv"
    )


def reasoning_path(category: Category, provider: Provider, variant: Variant) -> Path:
    """The per-finding evidence and reasoning, beside its predictions."""
    return variant_dir(variant) / (
        f"{category.name}_reasoning_{short_name(provider)}.json"
    )


def prompt_path(category: Category, variant: Variant = "zeroshot") -> Path:
    """The prompt file for a category and style."""
    suffix = None if variant == "zeroshot" else variant
    return PROMPT_DIR / category.prompt_filename(suffix)
```

`Literal` is already imported at the top of the file. Delete the now-unused old `prompt_path` definition that took `variant: str | None`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_config.py tests/test_multi_categories.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add classifier_multi/config.py tests/test_multi_config.py
git commit -m "refactor(multi): variant-aware output paths, remove gold surface from config"
```

---

### Task 3: evaluate.py takes ownership of the gold standard

**Files:**
- Modify: `classifier_multi/evaluate.py`
- Test: `tests/test_multi_evaluate_gold.py`

**Interfaces:**
- Consumes: `config.PROJECT_ROOT`, `config.DATA_DIR`, `config.Variant` from Task 2; `Category` from Task 1.
- Produces: `GOLD_DIR`, `gold_csv_path(category) -> Path`, `gold_value_aliases(category) -> dict[str, dict[str, str]]`, `confusion_matrix_path(category, variant) -> Path`, `OriginalsAccessError`. Existing `confusion_counts`, `matrix_row`, `build_matrix_rows`, `sensitivity`, `specificity`, `normalise`, `Counts` keep their current signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_evaluate_gold.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_evaluate_gold.py -v`
Expected: FAIL — `AttributeError: module 'classifier_multi.evaluate' has no attribute 'gold_csv_path'`.

- [ ] **Step 3: Add the gold knowledge to evaluate.py**

`evaluate.py` currently imports only `from pydantic import BaseModel` (line 20). Add below it:

```python
from pathlib import Path

from classifier_multi.categories import Category
from classifier_multi.config import DATA_DIR, PROJECT_ROOT, Variant

# This module is the only place in the package that knows the gold standard exists.
# Every other module is checked by tests/test_gold_boundary.py for even a mention of
# it, so a classification run cannot reach the answers by accident.
GOLD_DIR = PROJECT_ROOT / "dataset_gold_standard"

# Off limits everywhere, including here: the untouched source spreadsheets.
FORBIDDEN_DIR = "_originals"

GOLD_CSV_NAMES: dict[str, str] = {
    "feline_thorax": "feline_thorax_gold_standard.csv",
    "canine_thorax": "canine_thorax_gold_standard.csv",
    "canine_abdomen": "canine_abdomen_gold_standard.csv",
}

# Vocabulary the manual scorers used that the models are never asked to produce.
# Canine VHS is scored normal/enlarged there; every model returns normal/abnormal.
GOLD_VALUE_ALIASES: dict[str, dict[str, dict[str, str]]] = {
    "canine_thorax": {"vhs": {"enlarged": "abnormal"}},
}


class OriginalsAccessError(RuntimeError):
    """Raised on any attempt to reach dataset_gold_standard/_originals."""


def reject_originals(path: Path) -> None:
    """Refuse a path that reaches into the untouched source spreadsheets.

    Checked rather than merely never written, so a future edit that constructs such a
    path fails immediately instead of quietly reading files that are not ours to use.
    """
    if FORBIDDEN_DIR in Path(path).parts:
        raise OriginalsAccessError(
            f"{FORBIDDEN_DIR} is off limits and must never be read: {path}"
        )


def gold_csv_path(category: Category) -> Path:
    path = GOLD_DIR / GOLD_CSV_NAMES[category.name]
    reject_originals(path)
    return path


def gold_value_aliases(category: Category) -> dict[str, dict[str, str]]:
    """Per-condition value maps for this category, empty when none are needed."""
    return GOLD_VALUE_ALIASES.get(category.name, {})


def confusion_matrix_path(category: Category, variant: Variant) -> Path:
    """Written beside the predictions it scores, so the gold folder stays read-only."""
    return DATA_DIR / variant / f"confusion_matrix_{category.name}.xlsx"
```

Update the module docstring's second paragraph. It currently says the alias map comes from "the category's" aliases; change that sentence to:

```
One thing differs from the single-category scorer: the gold standard does not always
spell the positive class the way the models do. Canine VHS is scored normal/enlarged
by the manual scorers while every model returns normal/abnormal, so gold values are
put through the alias map defined in this module. The map lives here rather than on
Category because it describes the gold standard, and nothing outside this module is
allowed to know the gold standard exists.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_evaluate_gold.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Run the whole multi suite**

Run: `python -m pytest tests/test_multi_categories.py tests/test_multi_config.py tests/test_multi_evaluate_gold.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add classifier_multi/evaluate.py tests/test_multi_evaluate_gold.py
git commit -m "refactor(multi): evaluate.py owns all gold standard knowledge"
```

---

### Task 4: The boundary test

**Files:**
- Create: `tests/test_gold_boundary.py`

**Interfaces:**
- Consumes: the module layout established by Tasks 1–3.
- Produces: nothing importable. This task exists to make the constraint fail loudly.

- [ ] **Step 1: Write the test**

This task inverts the usual order: the test is the deliverable, and Tasks 1–3 are what make it pass. Create `tests/test_gold_boundary.py`:

```python
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
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_gold_boundary.py -v`
Expected: PASS if Tasks 1–3 are complete. If any module still names the gold standard, the failure message says which.

If `test_importing_classify_does_not_load_evaluate` fails at this point, `classify.py` or `csv_io.py` has an import that reaches `evaluate`. Find it with `python -c "import classifier_multi.classify, sys; print([m for m in sys.modules if 'evaluate' in m])"`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gold_boundary.py
git commit -m "test(multi): enforce the gold standard boundary"
```

---

### Task 5: The CLI entry point

**Files:**
- Create: `classifier_multi/main.py`
- Test: `tests/test_multi_main.py`

**Interfaces:**
- Consumes: `config.PROVIDERS`, `config.VARIANTS`, `config.load_settings`, `config.predictions_path`, `config.reasoning_path`, `config.prompt_path` (Task 2); `categories.CATEGORY_NAMES`, `get_category` (Task 1); `evaluate.gold_csv_path`, `evaluate.gold_value_aliases`, `evaluate.confusion_matrix_path` (Task 3); existing `csv_io.read_cases`, `csv_io.build_label_row`, `csv_io.write_labeled_csv`, `classify.classify_case`, `llm.build_model`, `prompt.load_prompt`, `schemas.labels_from`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_main.py`:

```python
"""The CLI must parse both subcommands and keep evaluate out of the classify path."""

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
    import ast
    from pathlib import Path

    return ast.parse(Path(main_module.__file__).read_text(encoding="utf-8"))


def test_evaluate_is_not_imported_at_module_scope():
    """Parsed, not grepped: prose in the docstring must not affect the result."""
    import ast

    for node in _main_ast().body:
        if isinstance(node, ast.Import):
            assert all("evaluate" not in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "evaluate" not in (node.module or "")
            assert all("evaluate" not in alias.name for alias in node.names)


def test_evaluate_is_imported_lazily_somewhere():
    """The counterpart: the import exists, it is just not at module scope."""
    import ast

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'classifier_multi.main'`.

- [ ] **Step 3: Write main.py**

Create `classifier_multi/main.py`:

```python
"""Command line entry point: one command to classify, one to score.

They are separate subcommands rather than one run because that is what makes the gold
standard boundary observable. `classify` never imports evaluate - the import sits
inside the evaluate handler - so a classification run cannot read the answers even by
accident. tests/test_gold_boundary.py checks that this holds.
"""

import argparse
import json

from classifier_multi import config
from classifier_multi.categories import CATEGORY_NAMES, get_category
from classifier_multi.classify import classify_case
from classifier_multi.csv_io import build_label_row, read_cases, write_labeled_csv
from classifier_multi.llm import build_model
from classifier_multi.prompt import load_prompt
from classifier_multi.schemas import labels_from


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classifier_multi")
    subcommands = parser.add_subparsers(dest="command", required=True)

    classify_parser = subcommands.add_parser(
        "classify", help="label one category's cases with one provider"
    )
    classify_parser.add_argument("--category", required=True, choices=CATEGORY_NAMES)
    classify_parser.add_argument("--provider", required=True, choices=config.PROVIDERS)
    classify_parser.add_argument(
        "--variant", default="zeroshot", choices=config.VARIANTS
    )

    evaluate_parser = subcommands.add_parser(
        "evaluate", help="score predictions against the gold standard"
    )
    evaluate_parser.add_argument("--category", required=True, choices=CATEGORY_NAMES)
    evaluate_parser.add_argument(
        "--variant", default="zeroshot", choices=config.VARIANTS
    )

    return parser


def run_classify(args: argparse.Namespace) -> int:
    category = get_category(args.category)
    settings = config.load_settings()
    model = build_model(args.provider, settings)
    prompt = load_prompt(config.prompt_path(category, args.variant))
    cases = read_cases(config.input_csv_path(category), category)

    labels_by_case: dict[str, dict[str, str]] = {}
    reasoning: list[dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        result = classify_case(model, prompt, category, case)
        labels = labels_from(result.classification)
        labels_by_case[case.case_id] = build_label_row(category, labels)
        reasoning.append(result.classification.model_dump(mode="json"))

    predictions = config.predictions_path(category, args.provider, args.variant)
    write_labeled_csv(
        config.input_csv_path(category), predictions, category, labels_by_case
    )

    reasoning_file = config.reasoning_path(category, args.provider, args.variant)
    reasoning_file.parent.mkdir(parents=True, exist_ok=True)
    reasoning_file.write_text(json.dumps(reasoning, indent=2), encoding="utf-8")

    print(f"wrote {predictions}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    # Imported here, not at module scope: this is the only code path allowed to know
    # the gold standard exists.
    from classifier_multi import evaluate

    category = get_category(args.category)
    print(f"scoring {category.name} ({args.variant}) against {evaluate.gold_csv_path(category)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "classify":
        return run_classify(args)
    return run_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

`run_evaluate` is intentionally a stub that resolves paths and prints. Wiring the full
matrix write is a follow-up; the spec's scope here is the boundary, not new scoring
behaviour.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_main.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Verify the boundary still holds with main.py present**

Run: `python -m pytest tests/test_gold_boundary.py -v`
Expected: PASS. In particular `test_importing_classify_does_not_load_evaluate` must still print `False`.

- [ ] **Step 6: Check the CLI responds**

Run: `python -m classifier_multi.main --help`
Expected: usage text listing `classify` and `evaluate`. This makes no API calls.

Run: `python -m classifier_multi.main classify --help`
Expected: usage text listing `--category`, `--provider`, `--variant`.

Do **not** run a real `classify` command. Running one costs 200 API calls per invocation and is the user's decision.

- [ ] **Step 7: Commit**

```bash
git add classifier_multi/main.py tests/test_multi_main.py
git commit -m "feat(multi): CLI with separate classify and evaluate commands"
```

---

### Task 6: Few-shot rendering tests

**Files:**
- Create: `tests/test_multi_prompt.py`

**Interfaces:**
- Consumes: `prompt.load_prompt`, `prompt.render_messages`, `prompt.example_answer`, `prompt.PromptExample` (already implemented); `config.prompt_path` (Task 2); `get_category` (Task 1).
- Produces: nothing importable.

The few-shot machinery in `prompt.py` is already written but has no test — `tests/test_prompt.py` covers the single-category `classifier` package, not this one. This task closes that gap.

- [ ] **Step 1: Write the test**

Create `tests/test_multi_prompt.py`:

```python
"""Few-shot examples must become real conversation turns, and must match the schema."""

import json

import pytest

from classifier_multi import config
from classifier_multi.categories import CATEGORIES, get_category
from classifier_multi.prompt import (
    PromptExample,
    example_answer,
    load_prompt,
    render_messages,
)
from classifier_multi.schemas import RadiologyCase

CASE = RadiologyCase(
    case_id="TEST-1", findings_text="Findings text.", conclusions_text="Conclusions text."
)


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_both_prompt_variants_load(name):
    category = get_category(name)
    for variant in config.VARIANTS:
        prompt = load_prompt(config.prompt_path(category, variant))
        assert prompt.version
        assert "veterinary radiologist" in prompt.system.lower()


def test_zero_shot_prompt_is_two_messages():
    category = get_category("canine_abdomen")
    prompt = load_prompt(config.prompt_path(category, "zeroshot"))
    assert prompt.examples == []
    assert len(render_messages(prompt, category, CASE)) == 2
    for example in prompt.examples:
        answer = json.loads(example_answer(category, example))
        assert [f["finding"] for f in answer["findings"]] == list(category.asked_findings)
        assert all(f["label"] in ("normal", "abnormal") for f in answer["findings"])


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_example_answers_leave_evidence_and_reasoning_empty(name):
    """Inventing quotes in the examples would teach the model to invent quotes."""
    category = get_category(name)
    prompt = load_prompt(config.prompt_path(category, "fewshot"))
    for example in prompt.examples:
        answer = json.loads(example_answer(category, example))
        assert all(f["evidence"] == "" for f in answer["findings"])
        assert all(f["reasoning"] == "" for f in answer["findings"])


def test_unknown_finding_name_raises():
    category = get_category("canine_abdomen")
    bad = PromptExample(
        case_id="X", findings="f", conclusions="c", labels={"not_a_finding": "abnormal"}
    )
    with pytest.raises(ValueError, match="not_a_finding"):
        example_answer(category, bad)


def test_derived_column_is_rejected_as_a_label():
    """diseased_lungs is computed in code, so it must never appear in an example."""
    category = get_category("feline_thorax")
    bad = PromptExample(
        case_id="X", findings="f", conclusions="c", labels={"diseased_lungs": "abnormal"}
    )
    with pytest.raises(ValueError, match="diseased_lungs"):
        example_answer(category, bad)


def test_omitted_findings_default_to_normal():
    category = get_category("canine_abdomen")
    example = PromptExample(
        case_id="X", findings="f", conclusions="c", labels={"colitis": "abnormal"}
    )
    answer = json.loads(example_answer(category, example))
    by_name = {f["finding"]: f["label"] for f in answer["findings"]}
    assert by_name["colitis"] == "abnormal"
    assert by_name["gastritis"] == "normal"
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_multi_prompt.py -v`
Expected: PASS, 17 tests. Every assertion covers code that already exists; a failure here means Task 1's category reorder broke something, or a prompt JSON is malformed.

- [ ] **Step 3: Run the complete new suite**

Run: `python -m pytest tests/test_multi_categories.py tests/test_multi_config.py tests/test_multi_evaluate_gold.py tests/test_gold_boundary.py tests/test_multi_main.py tests/test_multi_prompt.py -v`
Expected: all PASS.

- [ ] **Step 4: Confirm no pre-existing test regressed**

Run: `python -m pytest tests/ -q`
Expected: the same failures listed in "Pre-existing failures" and no others. The `classifier` package was not touched.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_prompt.py
git commit -m "test(multi): cover few-shot message rendering"
```

---

## Self-review notes

Checked against the spec:

- Every "Decisions" row maps to a task: scope (all), input columns (untouched — Task 1 leaves `schemas.COL_FINDINGS` / `COL_CONCLUSIONS` alone), boundary (Tasks 3, 4), outputs (Task 2), providers (Task 2 `short_name`), entry point (Task 5), column order (Task 1).
- The spec's `test_categories.py` is delivered as `tests/test_multi_categories.py`; the rename avoids colliding with the existing `classifier` tests.
- The spec's "Out of scope" items (resume, concurrency, cost control) have no task, by design.
- `run_evaluate` is a stub. The spec's scope is the boundary and the repointing, not new scoring behaviour; `build_matrix_rows` already exists and is unchanged.
- Signature consistency: `predictions_path`, `reasoning_path`, `prompt_path`, `short_name`, `gold_csv_path`, `gold_value_aliases`, `confusion_matrix_path`, `reject_originals`, `build_parser`, `main` are each defined in exactly one task and referenced with the same name and argument order everywhere else.
