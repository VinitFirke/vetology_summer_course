# classifier_multi Restructure — Design

**Date:** 2026-08-19
**Status:** approved

## Purpose

Repoint `classifier_multi/` at the three `*_TOBE_classified(in).csv` files now in
`dataset_LLM_classification/`, and make the separation from `dataset_gold_standard/`
structural rather than conventional.

The package is currently **unrunnable against this dataset**: every `input_csv` in
`categories.py` names a `*_manual_final_50_appended_*` file that no longer exists.

Two constraints came from the user and drive most of the design:

1. The gold standard is not consulted until the very end, when confusion matrices and
   validation metrics are produced.
2. `dataset_gold_standard/_originals/` must never be read by this code, at any point.
   (Verified: nothing in the repo references it today.)

Every ambiguity below was settled with the user before any code was written.

## Data

Three CSVs in `dataset_LLM_classification/`, all 200 rows, all label columns empty:

| File | Rows | Source cols | Label cols |
|---|---|---|---|
| `canine_abdomen_TOBE_classified(in).csv` | 200 | 10 | 10 |
| `canine_thorax_TOBE_classified(in).csv` | 200 | 10 | 20 (incl. `vhs`) |
| `feline_thorax_TOBE_classified(in).csv` | 200 | 10 | 20 (incl. `diseased_lungs`) |

Facts established by inspection:

- CaseIDs are unique within each file (200/200) and **do not overlap between
  categories** — the three sets are disjoint.
- All label columns are entirely empty (non-null count 0). These are inputs awaiting
  prediction, not partially scored sheets.
- The 10 source columns are identical across all three files: `CaseID`,
  `Link to AI report`, `Link to Rad report`, `Findings (original radiologist report)`,
  `Conclusions (original radiologist report)`,
  `Recommendations (original radiologist report)`, `Original Radiologist`, and the
  three `(AI report)` columns.
- `Recommendations (original radiologist report)` is missing on 3 (abdomen), 11
  (canine thorax) and 5 (feline thorax) rows. It is not read, so this does not matter.
- `canine_abdomen` and `canine_thorax` label-column order matches `categories.py`
  exactly. **`feline_thorax` does not** — same 20 names, six positions rearranged.

## Decisions

Recorded as decisions, not assumptions. Each was put to the user explicitly.

| Question | Decision |
|---|---|
| Depth of restructure | Repoint + enforce the boundary. Module layout otherwise unchanged. |
| Which columns feed the model | `Findings` + `Conclusions` of the **radiologist** report only. AI-report columns stay unread. |
| Boundary enforcement | Structural, plus a test that fails on violation. |
| Output location | Prompt variant as a subfolder; flat filenames inside it. |
| Provider naming | Keep `cloud_*` keys; strip the prefix only when building filenames. |
| Entry point | New `main.py` with separate `classify` and `evaluate` subcommands. |
| Feline column order | The CSV header wins; `categories.py` is corrected to match. |

## Design

### Module responsibilities after the change

| Module | Change |
|---|---|
| `categories.py` | Drop `gold_csv` and `gold_value_aliases`. Repoint `input_csv`. Reorder feline `label_columns`. |
| `config.py` | Drop `GOLD_DIR` and `gold_csv_path()`. Add `Variant`, `short_name()`, `predictions_path()`, `reasoning_path()`. |
| `prompt.py` | No change. `prompt_path(category, variant)` already exists. |
| `schemas.py` | No change. |
| `llm.py` | No change. |
| `classify.py` | No change. |
| `csv_io.py` | No change. Writes by column name, so CSV column order is preserved regardless. |
| `evaluate.py` | Becomes the only gold-aware module: gains `GOLD_DIR`, gold filenames, the `vhs` alias map, the confusion-matrix path, and an `_originals` refusal. |
| `main.py` | New. Two subcommands. |
| `tests/test_gold_boundary.py` | New. Enforces the constraint. |

### Category definitions

`Category` loses two fields. Both are gold-standard concerns and move to `evaluate.py`:

```python
class Category(BaseModel):
    name: CategoryName
    input_csv: str
    label_columns: tuple[str, ...]
    derived: dict[str, tuple[str, ...]] = {}
    # gold_csv           REMOVED - evaluate.py owns gold filenames
    # gold_value_aliases REMOVED - evaluate.py owns gold vocabulary
```

`input_csv` values become:

- `canine_abdomen_TOBE_classified(in).csv`
- `canine_thorax_TOBE_classified(in).csv`
- `feline_thorax_TOBE_classified(in).csv`

Feline `label_columns` is rewritten to the CSV header order. The corrected sequence,
positions 9–14 being the ones that move:

```
pulmonary_nodules, esophagitis, pneumonia, bronchitis, interstitial,
diseased_lungs, hypo_plastic_trachea, cardiomegaly,
pleural_effusion, perihilar_infiltrate, focal_caudodorsal_lung,
right_sided_cardiomegaly, focal_perihilar, left_sided_cardiomegaly,
bronchiectasis, pulmonary_vessel_enlargement, thoracic_lymphadenopathy,
pulmonary_hypoinflation, pericardial_effusion, Alveolar_interstitial_pattern
```

This changes the order findings are listed in the prompt and in the answer schema. It
does not change the output CSV, which is written by column name.

`derived` is unchanged: `diseased_lungs` still rolls up from the parenchymal findings
on both thorax categories, and the abdominal sheet still has no derived column.

### Paths

`config.py` holds classification paths only. `GOLD_DIR` is deleted outright — its
absence is what the boundary test checks for.

```python
DATA_DIR = PROJECT_ROOT / "dataset_LLM_classification"

Variant = Literal["zeroshot", "fewshot"]

def short_name(provider: Provider) -> str:
    """cloud_gemma -> gemma. Filenames only; provider keys keep the prefix."""

def predictions_path(category, provider, variant) -> Path:
    # dataset_LLM_classification/<variant>/<category>_classified_<short>.csv

def reasoning_path(category, provider, variant) -> Path:
    # dataset_LLM_classification/<variant>/<category>_reasoning_<short>.json

def prompt_path(category, variant) -> Path:
    # zeroshot -> <category>.json,  fewshot -> <category>_fewshot.json
```

Resulting layout:

```
dataset_LLM_classification/
  canine_thorax_TOBE_classified(in).csv        input, never written to
  zeroshot/
    canine_thorax_classified_gemma.csv
    canine_thorax_reasoning_gemma.json
  fewshot/
    canine_thorax_classified_gemma.csv
```

The variant subfolder is what stops a few-shot run overwriting the zero-shot baseline —
the failure mode present in the code today, where paths key only on category and
provider.

### The gold boundary

`evaluate.py` becomes the single point of contact with `dataset_gold_standard/`. It
owns:

- `GOLD_DIR`, and the per-category gold filenames
  (`{category}_gold_standard.csv`);
- the vocabulary map `{"vhs": {"enlarged": "abnormal"}}`, which exists because the
  manual scorers wrote the vertebral heart score as normal/enlarged while every model
  returns normal/abnormal;
- the confusion-matrix output path;
- an explicit refusal: any resolved path containing `_originals` raises, rather than
  relying on that directory simply never being named.

No classification module imports `evaluate`. The dependency runs one way only.

### Entry point

```
python -m classifier_multi.main classify \
    --category canine_thorax --provider cloud_gemma --variant fewshot

python -m classifier_multi.main evaluate \
    --category canine_thorax --variant fewshot
```

`classify` must not import `evaluate`, directly or transitively. Keeping them as
separate subcommands is what makes the boundary observable at the command line: the
only invocation that can touch the gold standard is the one named `evaluate`.

`main.py` is necessarily the one module that names both sides, so it imports
`evaluate` **lazily, inside the `evaluate` subcommand handler**. Running `classify`
therefore never loads a gold-aware module. This is the single documented exception to
the rule below, and the boundary test encodes it rather than waiving it.

## Error handling

Unchanged from current behaviour, which is already adequate:

- `verify_columns()` fails loudly if the spreadsheet and `categories.py` disagree.
- `classify.py` retries transient provider failures with capped backoff, and raises
  `ProviderUnavailable` immediately on 401/402/403/404 rather than burning six minutes
  per case on a dead provider.
- `write_labeled_csv()` leaves cells empty for cases absent from the results, so a
  partial run produces a partial file rather than a wrong one.
- `confusion_counts()` skips pairs where either side is blank, so an unfinished run
  scores only the cases it completed.

New:

- `example_answer()` raises `ValueError` on a finding name the category does not ask
  about. Already implemented.
- `evaluate` raises on any path resolving inside `_originals`.

## Testing

| Test | Asserts |
|---|---|
| `test_gold_boundary.py` | Two assertions. (a) Source scan: none of `categories`, `config`, `prompt`, `schemas`, `llm`, `classify`, `csv_io` contains `gold`, `_originals`, or `dataset_gold_standard`, and none imports `evaluate`. `main.py` is excluded from this scan by design — see the exception above. (b) Import check: importing `classifier_multi.classify` in a clean interpreter leaves `classifier_multi.evaluate` absent from `sys.modules`, which is what proves the lazy import in `main.py` actually holds. |
| `test_categories.py` | Each category's `label_columns` equals the real CSV header slice, in order. Catches drift between sheet and code. |
| `test_prompt.py` (extend) | Empty `examples` yields 2 messages; N examples yield 2N+2; unknown finding name raises; rendered examples cover every asked finding in schema order. |
| existing suite | Continues to pass. |

The `test_categories.py` check reads only `dataset_LLM_classification/`, so it does not
itself breach the boundary.

## Out of scope

Recorded so the omissions are deliberate, not oversights.

- **Resume / incremental writing.** Predictions are collected in memory and written
  once at the end. At 200 cases a late failure loses the whole run. The user chose to
  keep the run machinery unchanged; this is the first thing to add if a run is lost.
- **Concurrency.** Cases are classified serially.
- **Cost control.** One category × provider × variant is 200 calls; the full grid
  across 3 categories × 4 providers × 2 variants is 4,800, with few-shot prepending two
  long reports to every one of them.

## Known unknowns

- **Gold alignment is unverified.** Per the user's instruction the gold CSVs have not
  been opened, so whether their 200 CaseIDs match the `TOBE_classified` CaseIDs is
  unknown. It will surface the first time `evaluate` runs. If they do not align,
  `build_matrix_rows()` will silently score only the intersection — worth an explicit
  count check when evaluation is first run.
- **Only feline has few-shot examples reviewed for threshold consistency.**
  `canine_thorax` example `2721429` leaves `bronchitis` normal for "minimal to mild
  diffuse bronchial pattern", while feline `2538154` labels it abnormal for "mild
  diffuse bronchial pattern". Both are defensible; few-shot examples teach exactly this
  threshold. Flagged to the user, left as their call.
