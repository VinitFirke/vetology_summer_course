# How I would build this, and how it differs from the existing codebase

**A design-rationale document. Nothing here is a work order — it is the reasoning you asked
for, to explain how another engineer would approach the same problem.**

Premise: I am handed one CSV. 50 feline thoracic radiology reports, columns 0–9 carrying
text and metadata, columns 10–29 empty and named after 20 findings. A separate manually
scored gold file exists. Goal: fill the label columns with an LLM, then produce a confusion
matrix.

---

## 1. The biggest difference is the order, not the tools

The existing project was built classifier-first: schema → prompt → `classify_case` → full
run → *then* Phase 2 built `evaluate.py` and the confusion matrix.

**I would build evaluation first, and I would not call a model until it existed.**

This is not tidiness. It is the difference between knowing and not knowing what your number
means. The existing project reached 97.2% agreement before anyone computed that labelling
every cell `normal` scores **95.1%**. The whole apparatus — three providers, 150 calls,
structured output, retry logic — is worth 2.1 percentage points of accuracy, and that fact
was unavailable until months of work had been done.

An evaluation-first order surfaces it on day one, for free, from the gold file alone.

### The roadmap

| # | Step | Why here | Cost |
|---|---|---|---|
| 1 | **Read the data by hand.** 10 reports, end to end. Count label prevalence in the gold file. | You cannot choose a metric without knowing the class balance. 30 minutes here reprices the entire project. | free |
| 2 | **Name the decision the system serves.** Who consumes the output; is a false positive or a false negative worse. | Determines the operating point. Without it, "accuracy" is a number with no owner. | free |
| 3 | **Build the scorer.** Pure functions, gold in, metrics out. Test with hand-built fixtures. | It is the only component whose correctness you can fully verify without an API. | free |
| 4 | **Score the trivial baselines through it**: all-normal, then keyword match, then keyword + negation. | Now you have a number the LLM must beat, and a scoring path already proven by three systems. | free |
| 5 | **Then the LLM**: schema → provider adapter → prompt → one case → smoke → full run. | Everything downstream already exists and is tested. | ~$0.05 |
| 6 | **Ablations as flags from the start**, not retrofitted. | Retrofitting a flag means re-running everything. | ~$0.05 each |

Steps 1–4 cost nothing and answer most of the interesting questions. In the existing
project they happened last, partially, or not at all.

---

## 1a. What "evaluation-first" actually means

Not "write tests first". It means **the scorer defines the contract, and predictors plug
into it** — so the LLM is one implementation among several rather than the centre of the
system.

### The structural difference

In the existing codebase the LLM is the centre. `classify_case` is the thing; `main.py`
writes `feline_thorax_labeled_{provider}.csv`; and `evaluate_main.py` later reads that CSV
back, joins on `CaseID`, and applies `GOLD_TO_PREDICTION` to reconcile a column rename. The
**file format is the interface** between producing labels and scoring them.

Evaluation-first inverts it. Define the interface as a type:

```python
class Predictor(Protocol):
    name: str
    def predict(self, case: RadiologyCase) -> dict[str, Label]: ...
```

and one runner that takes any list of them:

```python
def score_all(predictors, cases, gold) -> pd.DataFrame   # one row per (predictor, finding)
```

`AllNormal`, `KeywordMatch`, `KeywordNegation` and `LLMPredictor` are then peers. Adding a
second prompt variant is a new `Predictor`, not a new CSV convention.

### The order of work

**Day 1 — no API, no cost**

1. Write the scorer and its tests against hand-built fixtures.
2. Write `AllNormal`. It is three lines.
3. Run it end to end: data → predictor → scorer → metrics table.
4. Read the number. **95.1%.** The project is now correctly priced before a prompt exists.

**Day 2 — still no cost.** Add the keyword and keyword+negation predictors, score them.

**Day 3 — first API call.** `LLMPredictor` drops into a harness already validated by three
other systems.

### What it buys, concretely

- **You cannot build something whose contribution you can't measure.** The 95.1% question
  answers itself on day one instead of never.
- **The scorer is proven before it judges anything expensive.** `evaluate.py` here was
  written after the runs; a bug in it would have been debugged against paid data.
- **Prompt comparison becomes free.** Two prompts = two `Predictor` instances and the table
  already exists. Today it means two CSVs and a manual diff.
- **The CSV stops being load-bearing.** It becomes an output sink rather than the seam
  between two halves of the system.

---

## 1b. Which trivial baselines, specifically

### Tier 0 — degenerate (a few lines each)

| # | Baseline | Result on this data | What it tells you |
|---|---|---|---|
| 1 | **All-normal** | acc 95.1%, **F1 0.000** | The floor. Reprices the headline number. |
| 2 | **All-abnormal** | recall 1.000, precision 0.049 | Brackets the range; shows why precision matters at 5% prevalence. |
| 3 | **Prevalence-matched random** | F1 ≈ 0.05 | What you get from knowing the base rate and nothing else. Any claim of signal must beat it. |

### Tier 1 — the real comparator

**4. Keyword presence.** Per finding, a short list of surface terms; label `abnormal` if any
appears in findings + conclusions.

The data point that makes this non-optional: **"bronchial" appears in 26 of 50 reports; the
gold standard has 18 `bronchitis` positives; the LLM called 24.** Plain keyword matching is
already in the right neighbourhood on the highest-prevalence finding in the set. No claim
about clinical reasoning survives until it has beaten this.

**5. Keyword + negation.** The same, minus matches falling inside a negation scope. This is
the serious baseline — the classic NegEx approach from clinical NLP, routinely within a few
points of an LLM on extraction tasks.

Negation cues present in these reports: `no`, `no evidence of`, `without`, `negative for`,
`not identified`, `not appreciated`, `unremarkable`, `within normal limits`, `is normal`,
`absent`.

Hedge cues needing a separate, explicit decision: `possible`, `suspect`,
`cannot be excluded`, `versus artifact`, `mild`, `minimal`. The corpus contains
*"Mild generalized cardiomegaly versus artifact"* — how that single phrase is treated moves
FP/FN, and no one has yet decided.

### Tier 2 — optional

**6. Section-scoped keyword** — search only the relevant section header. Limited here: just
12 of 50 reports use the structured `Cardiac silhouette:` layout.

**7. TF-IDF + logistic regression per finding**, cross-validated. Marginal at n=50, but for
`bronchitis` (18 positives) it is feasible and would be informative.

### What I would actually build: 1, 4 and 5

Three baselines, no API cost, roughly an afternoon.

**And the interpretation is the point.** If keyword+negation reaches F1 0.70 and the LLM
reaches 0.77, that is a different result from "the LLM achieves 0.77". It says the LLM buys
seven points over a free, instant, deterministic, fully auditable alternative — and for a
clinical tool a reviewer might reasonably prefer the auditable one. That conversation is
impossible without the baseline.

---

## 2. Design choices, side by side

| Decision | Existing codebase | What I would do | Why |
|---|---|---|---|
| **Build order** | classifier first, evaluation second | evaluation and baselines first | 95.1% baseline is invisible otherwise |
| **Provider layer** | LangChain `with_structured_output` | `openai` + `mistralai` SDKs behind a ~60-line `Protocol` | see below |
| **Answer schema** | `findings: list[FindingLabel]` | one required field per finding, generated from the enum | a list permits omissions; a fixed key set cannot |
| **Schema enforcement** | client-side parsing | server-side strict JSON schema | the model cannot emit a malformed answer at all |
| **Missing finding** | silently defaults to `normal` in `build_label_row` | structurally impossible; if it happened, it would raise | silent defaults hide model failure as data |
| **Evidence quotes** | collected, never checked | asserted to appear verbatim in the source text | free hallucination gate |
| **Glossary** | deliberately absent, untested | a flag, and an ablation run on day one | it is the largest untested prompt variable |
| **Metrics** | accuracy, sensitivity, specificity | precision / recall / F1, prevalence in every table | 5% prevalence makes accuracy uninformative |
| **Model IDs** | `mistral-medium-latest` | pinned snapshots, recorded per output row | `-latest` is not reproducible |
| **Baselines** | none | three, scored identically | otherwise the LLM's contribution is unmeasured |

### Why not LangChain

Three OpenAI-compatible endpoints do not need a framework, and the abstraction demonstrably
leaked twice in this codebase:

- `reasoning_effort` was silently transferred into `model_kwargs` on Mistral with only a
  `UserWarning` — a parameter central to the study, accepted without confirmation that it
  reached the API.
- Kimi intermittently wraps responses in a markdown fence; LangChain's parser fails on it.

Direct SDKs give explicit provider errors, exact control over retries, and OpenAI's native
strict-schema path which enforces structure **server-side** rather than parsing hopefully on
the client. The cost is roughly 60 lines of adapter.

### Why a fixed-key schema

`list[FindingLabel]` permits duplicates, omissions, extras and arbitrary order.
`build_label_row` absorbs omissions by defaulting to `normal` — silently, and without
counting how often it fires. Nobody knows whether that path has ever executed.

Generating one required field per finding from `FindingName` keeps the enum as the single
source of truth while making "the model skipped a finding" a schema violation instead of a
silent `normal`.

### Why check the evidence quotes

The schema already demands a verbatim quote and the models comply — **590 of 592 quotes
appear verbatim in the source (99.7%)**. This is collected and never validated.

The two that fail are both `abnormal` calls on findings with **zero** gold positives
(`right_sided_cardiomegaly`, `Alveolar_interstitial_pattern`). A one-line assertion would
have flagged two of the 26 false positives at no cost.

---

## 3. What the existing codebase gets right

A rebuild should copy these rather than reinvent them:

- **`FindingName` as the single source of truth**, cross-checked against CSV headers at
  startup by `csv_io.verify_columns`. Renaming a spreadsheet column produces a loud error
  instead of a silently unlabelled finding. Genuinely good.
- **`evaluate.py` as pure functions over label strings** — no files, no API, fully testable.
- **The model is built by the caller and injected**, which is the reason `test_classify.py`
  runs with no network.
- **Prompt text in a versioned JSON file** rather than buried in code.
- **Input CSV read-only, one output copy per provider**, so a failed run cannot damage source
  data.
- **`diseased_lungs` derived in code**, never asked of the model, so it cannot contradict the
  findings it is built from.

One caveat on the prompt file: the prompt actually lives in **two** places, because the
`Field(description=...)` strings in `schemas.py` are shipped to the model as part of the JSON
schema. Worth knowing when tuning wording.

---

## 4. The two methodological gaps

Both were raised and consciously set aside. They are the sharpest points for any explanation.

**The baseline.** All-normal scores 95.1% on the 19 judged findings; the models score 97.2%.
Reported as accuracy, the project looks like a 97% success. Reported on the positive class it
looks like this:

| system | precision | recall | F1 |
|---|---|---|---|
| all-normal | — | 0.000 | **0.000** |
| `gpt-5.6-luna` | 0.639 | 0.979 | **0.773** |

The second framing is both more honest and more flattering. Accuracy was hiding a good
result behind a meaningless one.

**The label distribution.** 9 of 19 findings have zero positives in 50 cases and account for
450 of the 950 scored cells; sensitivity is undefined for all nine. The study effectively
rests on three conditions: `bronchitis` (18 positives), `cardiomegaly` (10),
`pulmonary_nodules` (5). Any per-condition table should show prevalence beside the metric,
and conditions below ~5 positives should be reported separately rather than averaged in.

**On the framing choice.** Retaining the "apply your own clinical judgement" persona is
defensible, but it has a consequence worth stating once: the model cannot see the images and
is reading another radiologist's finished text, so a disagreement with gold is not
unambiguously a model error. The confusion matrix then measures agreement-with-annotator
rather than extraction accuracy. FP=26 against FN=1 is the signature of a model inferring
rather than reporting — which may be exactly what the persona asked for.
