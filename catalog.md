# Catalog — Uncertainty Quantification Feature

A running log of design decisions for adding LLM uncertainty quantification to the
feline thoracic radiology classifier. Each entry records what was decided, why, and
what was rejected — so a decision can be revisited later without re-deriving it.

This file is the *index*. The full design lives in `uncertainty-quantification-design.md`
(same directory), structured after `software-design-practices.md`. The implementation
plan lands in `plan.md`.

**Status:** design complete and approved 2026-08-08. Not yet implemented.

---

## 1. What is being built

An adaptation of Savage et al., *"Large language model uncertainty proxies:
discrimination and calibration for medical diagnosis and treatment"*
(JAMIA 2025;32(1):139–149, doi:10.1093/jamia/ocae254) onto this codebase.

The paper measures three proxies for how certain an LLM is about its own answer, then
asks two questions of each proxy:

- **Discrimination** — does the proxy separate answers the model got right from
  answers it got wrong? Measured by ROC AUC; the paper's usefulness threshold is 0.7.
- **Calibration** — when the proxy says "85% confident", is the model right 85% of the
  time? Measured by a calibration plot, Expected Calibration Error (ECE), and Brier score.

The three proxies:

| Proxy | Abbrev. | How it works |
|---|---|---|
| Confidence elicitation | CE | Ask the model to rate its own certainty 0–100 |
| Token-level probability | TLP | Read the probability the model assigned to the tokens it emitted |
| Sample consistency | SC | Run the same prompt N times; measure how much the answers agree |

The paper's headline result: **SC discriminates best** (ROC AUC 0.68–0.79), TLP and CE
are weaker, and CE is consistently over-confident.

## 2. Source material

All gitignored — reference material, not ours to redistribute.

| Path | What it is |
|---|---|
| `ocae254.pdf` | The paper |
| `ocae254_supplementary_data.docx` | Supplementary data |
| `25962529/Supplemental Information/` | The authors' released notebooks and result workbooks |
| `25962529/…/Uncertainty_Proxy_Generation_Steps.md` | Step-by-step notes on how each proxy's raw value is produced |
| `25962529/…/Supplemental_Information_VIII_Statistical_Analysis.ipynb` | Their ROC AUC / ECE / Brier / bootstrap code |

## 3. How our task differs from the paper's

This is the single most consequential difference and it simplifies several things.

| | Paper | This codebase |
|---|---|---|
| Task | Open-ended free-text diagnosis | Closed-set multi-label binary |
| Answer space | Any string | `{normal, abnormal}` × 19 findings |
| Extracting the answer | Regex `\[(.*?)\]` out of prose | Pydantic structured output |
| Deciding if two answers agree | GPT-4 annotation, or sentence-embedding cosine similarity | Exact string equality |
| Grading correctness | Two blinded physicians, third to break ties | `dataset_gold_standard/` already exists |

**Consequences.** Sample consistency becomes an exact majority-vote fraction — no
annotator LLM, no `pritamdeka/S-PubMedBert-MS-MARCO`, no `intfloat/e5-small-v2`, no
cosine similarity matrices. Two of the paper's five SC variants collapse into one
exactly-computable number. Correctness grading is free.

---

## Locked decisions

### D1 — Unit of analysis: one row per (case, finding)

**Decided 2026-08-07.**

A single data point is one finding on one case: 50 × 20 = 1000 rows per provider per
effort tier. Confidence comes from the proxy; correctness from the gold standard.

**Why.** `classifier/evaluate.py:120` `build_matrix_rows()` already scores per condition,
so this slots into the existing shape. At the per-case alternative (n=50) the ROC AUC
confidence interval would be roughly ±0.15 — every proxy would overlap every other one
and the analysis would conclude nothing. The paper's 10-decile ECE also needs more than
5 points per bin.

**Rejected.** Per-case (n=50, too noisy); both (the per-case half is the weak one).

**Carries a caveat.** The 20 findings within a case are correlated, so bootstrap CIs
must resample **by case**, not by row — see D9.

### D2 — Run matrix: 50 cases × N=5 replicates × 3 effort tiers × 3 providers

**Decided 2026-08-07.**

**Why N=5 rather than the paper's 15.** Budget. This is a portfolio project intended to
demonstrate the method, not to produce a valid measurement, and the user has ~$40 of
credit across three accounts. N=15 costs ~$217.

**Known limitation, stated deliberately.** With binary labels, SC confidence is a
majority-vote fraction, so N=5 yields only three distinct values: {0.6, 0.8, 1.0}. Every
ROC curve will have three points and the 10-bin calibration plot degenerates. This is
documented rather than hidden; the fix is one config change (raise N). N mattered *more*
here than in the paper, whose free-text answers produced finely-graded agreement scores.

### D3 — Provider and model lineup

**Decided 2026-08-07.** Groq dropped by the user before this design started.

| Provider | Model | Price in/out per 1M | Est. cost at D2 |
|---|---|---|---|
| openai | `gpt-5.6-luna` | $0.20 / $1.20 | ~$1.40 of $10 |
| mistral | `mistral-medium-3.5` | $1.50 / $7.50 | ~$7.40 of $10 |
| kimi | `kimi-k3` | $3.00 / $15.00 | ~$14.70 of $20 |
| | | **total** | **~$23.50** |

**Why luna and not `gpt-5.5`.** gpt-5.5 is $5.00/$30.00 — its output price alone puts
N=5 × 3 tiers × 50 cases at ~$28 against a $10 budget. luna runs the identical design
for ~$1.40, keeping all 50 cases and all three tiers so the results table stays uniform
across providers.

**Cost of that choice.** The existing `feline_thorax_labeled_openai.csv` and the
confusion matrix were produced on `gpt-5.5`, so the OpenAI column of the UQ study will
not correspond to them unless the base classification is re-run on luna (~$0.30).

**Rejected.** Keeping gpt-5.5 on 20 cases via the flex tier (ragged dataset); dropping
OpenAI entirely (loses a provider); topping up credit (~$120–217).

### D4 — Effort tiers replace temperature

**Decided 2026-08-07.** Mapping supplied by the user; to be verified by the probe (D6).

| Canonical tier | openai | mistral | kimi |
|---|---|---|---|
| low | `low` | `none` | `low` |
| medium | `medium` | `high` | `high` |
| high | `high` | `high` | `max` |

Lives as an `EFFORT_LEVELS: dict[Provider, dict[Tier, str]]` table in config, beside the
existing `REASONING_EFFORT` dict in `classifier/config.py:26`.

**A note on what the substitution changes.** In the paper, temperature did double duty:
it was the experimental condition *and* the source of the randomness SC depends on.
That's why they could only run SC at temperature 0.5 and 1.0 — at 0 there wasn't enough
variation between responses to measure. Reasoning effort is not a randomness knob at
all; the variation between replicates comes from ordinary sampling, which is present at
every tier. So SC is measurable at all three tiers here, which is cleaner than the
paper's design.

**Mistral's medium and high tiers are the same configuration** (`high` both times). Not
a defect — it is a free negative control. Any gap between those two columns is
run-to-run sampling noise, giving a noise floor to judge the other providers' effort
effects against. Costs ~$2.50 to run the duplicate.

### D5 — Proxies: CE and SC now, TLP gated on a capability probe

**Decided 2026-08-07.**

CE and SC work on any provider and get built first. TLP needs the API to return token
logprobs, which reasoning models often decline to do and which Mistral's API does not
offer at all. Rather than guess, a ~15-line script sends one case per provider with
`logprobs=True` and reports whether logprobs actually came back (~$0.05). TLP is built
only for providers that answered yes; the rest show "not supported" in the results,
which is itself a legitimate finding.

**Rejected.** Skipping TLP outright (loses the most technically interesting plumbing);
committing to TLP on OpenAI without checking (risks dead code).

#### D5 result — measured 2026-08-08

`python -m uncertainty.probe`, three calls, ~$0.05:

| provider | model | logprobs | detail |
|---|---|---|---|
| openai | `gpt-5.6-luna` | **NO** | response carried no logprobs field |
| mistral | `mistral-medium-latest` | **NO** | response carried no logprobs field |
| kimi | `kimi-k3` | **NO** | response carried no logprobs field |

**TLP is not buildable. CE and SC are the final proxy set.** Task 14 in `plan.md` is
skipped.

Worth noting *how* they failed: all three calls **succeeded** and simply came back
without the field — none raised, not even Mistral, whose API has no logprobs parameter
at all. That is the silent-omission failure mode the probe existed to catch. Had TLP been
built on the assumption it worked, it would have produced nothing and the gap would only
have surfaced during analysis, after a full paid run.

**Scope of this finding.** The probe exercises logprobs through
`with_structured_output`, which is the path this pipeline actually uses. A raw
Chat Completions call without structured output might behave differently on some
providers, but that is not a path this codebase takes, so the answer is decisive for our
purposes. Reasoning models declining logprobs is also the documented expectation for the
OpenAI o-series and gpt-5 families.

**Consequences.** The `logprobs` field stays on every JSONL record as `null` — it
documents that the question was asked. The results workbook will carry two proxies per
tier rather than three. This is reportable rather than a gap: the paper's own conclusion
is that SC beats TLP and CE nearly everywhere, and that TLP degraded on newer models —
finding it unavailable on three 2026-era reasoning models extends that trend.

### D6 — Architecture: new `uncertainty/` package, sibling to `classifier/`

**Decided 2026-08-07.**

```
uncertainty/
  config.py     effort tiers, N, output paths, price table
  schemas.py    CaseLabels, CaseConfidence  (slim, UQ-only)
  sample.py     run one case N times at one tier -> raw samples
  proxies.py    raw samples -> CE / SC / TLP confidence values
  stats.py      (confidence, correct) -> ROC AUC, ECE, Brier
  probe.py      the logprobs capability check
uq_main.py          collect samples   (spends money)
uq_analyze_main.py  proxies + stats + plots   (free, offline, re-runnable)
```

Reuses `classifier.llm.build_model`, `classifier.prompt`, `classifier.csv_io`,
`classifier.schemas` unchanged.

**The load-bearing part: sampling and analysis are separate programs with a file
between them.** `uq_main.py` writes raw JSONL and computes nothing; `uq_analyze_main.py`
reads JSONL and calls no API. On a $23 budget that is the difference between paying once
and paying every time an ECE bug gets fixed. Same principle as
`classifier/evaluate.py`, which is pure functions over label strings.

**Rejected.** Extending `classifier/` in place (turns one package into
classify + sample + inferential statistics, and grows `main.py` a second mode with its
own resume semantics — against REFERENCE.md REF3b). Notebooks like the authors' (the
tested, typed structure *is* the portfolio value; their notebooks are chained-assignment
and copy-pasted cells).

### D7 — Sampling stage details

**Decided 2026-08-07.**

**D7a — All 5 replicates use the same labels-only schema.** New `CaseLabels` in
`uncertainty/schemas.py`: `case_id` plus 19 `(finding, label)` pairs, no `evidence`, no
`reasoning`. Those two fields are ~45% of output tokens and no proxy reads them.

The deeper reason is a confound that was nearly designed in. Using the full schema for
replicate 1 (it is also the SinglePass answer) and the slim one for replicates 2–5 means
replicate 1 is generated by a different prompt than the others. Sample consistency
measures variation *between replicates* — so some of the measured disagreement would be
caused by the schema change rather than by model uncertainty, and the two are
indistinguishable after the fact. All five replicates must be generated identically.
The paper's 15 responses all came from one unchanged prompt for the same reason.

Existing `main.py` full-schema output is untouched and remains the headline
classification.

**D7b — `classify_case` gains a `schema` default argument.**
`classify_case(model, prompt, case, schema=CaseClassification, max_attempts=10)`, with
`classifier/classify.py:75` becoming `model.with_structured_output(schema, ...)`. Lets
the UQ sampler reuse the existing 10-attempt backoff loop (`backoff_seconds` 5→60s)
instead of growing a second copy that drifts. Every current caller and all seven tests
in `tests/test_classify.py` are unaffected. Line 88's `parsed.case_id = case.case_id`
works for both schemas.

**D7c — CE is two-step and effort-matched.** Step 1 is replicate 1, already paid for.
Step 2 is one call per case per tier: the case text plus replicate 1's 19 labels, asking
for a 0–100 certainty per finding, returned as structured output (19 integers, no
bracket convention and no `str.extract('(\d+)')`).

The paper crossed answer-temperature × CE-temperature for nine CE values per question
(`CE_bT_0_0` … `CE_bT_10_10`). We match CE effort to answer effort — three combinations,
not nine. The off-diagonal answers "does the effort you *rate* at matter separately from
the effort you *answered* at", which is not the question being asked here, and it costs
two-thirds of the CE budget.

**D7d — A cost guard runs before any API call.** `uq_main.py` prints planned call count,
estimated tokens and estimated dollars, then exits unless `--yes` is passed; `--dry-run`
prints and never proceeds. `main.py`'s `--smoke` protects against broken wiring, not
against correct wiring pointed at the wrong number — at 900 calls against a $10
account there is no undo. (Per provider: 750 replicate calls + 150 CE calls = 900.)

**D7e — N must be odd.** SC confidence is the fraction of replicates agreeing with the
majority. With an even N a binary label can split evenly with no majority; with N=5 the
smallest majority is 3, so no tie-break rule is needed anywhere in the code.

**D7f — Sampling runs through a thread pool, default 4 workers.** The full matrix is
2,700 calls; sequentially at ~5s that is nearly four hours, and four workers brings it
under an hour. Safe because
retry logic is per-call — the only shared state is the JSONL append, which takes a lock.

**Artifacts written** (all under `dataset_LLM_uncertainty/`, gitignored):

| File | Contents |
|---|---|
| `samples_{provider}_{tier}.jsonl` | one line per replicate call: labels, usage, logprobs |
| `ce_{provider}_{tier}.jsonl` | one line per CE call: 19 scores |
| `failures_{provider}_{tier}.jsonl` | cases that exhausted all retries |

Append-only JSONL gives resume for free: count existing replicates per
`(case_id, tier)` on startup and request only the shortfall. A crash at case 40 costs
nothing — which matters when the entire Kimi budget is a single run.

### D8 — Proxy computation

**Decided 2026-08-08.**

`uncertainty/proxies.py` is pure functions over parsed JSONL, emitting one long-format
table `uncertainty_proxies.csv` with columns
`provider, tier, case_id, finding, proxy, confidence, answer, gold, correct, n_replicates`
— roughly 17k–26k rows. One `groupby` away from every statistic in D9.

**SC** — `Counter(labels).most_common(1)` gives the majority label and its count;
confidence is `count / len(labels)`. That is the entire proxy. It replaces the paper's
GPT-4-annotation and sentence-embedding variants, both of which existed only to decide
whether two free-text diagnoses meant the same thing.

**CE** — answer is replicate 1's label, confidence is `score / 100`. The paper's scale is
0 = definitely uncertain, 100 = definitely certain, so higher means more confident and no
inversion is applied. Their headline CE finding was consistent over-confidence
(Figure 5 curves below the diagonal); if ours reproduce that, it is a replication.

**TLP** — answer is replicate 1's label; confidence is the probability of *that finding's
label token*, found by walking `logprobs.content` and taking the value token after each
`"label":` key.

Two deliberate departures from the paper:

1. They averaged (and took the minimum of) token probabilities across a multi-token
   free-text answer. Here each label is one token in a known position, so the per-finding
   label-token probability is simpler and strictly more informative than averaging it
   with the probabilities of `{`, `"finding"` and `,`.
2. Conversion uses `math.exp(logprob)`. Their released code uses `pow(10, logprob)`, but
   OpenAI returns natural log. Since `10^x` is monotonic in `x`, their **ROC AUC values
   are unaffected** — AUC depends only on ranking — but their **TLP calibration numbers
   would be shifted**, because ECE and Brier compare the value itself against observed
   accuracy.

**Correctness** joins the gold standard through the existing `config.GOLD_TO_PREDICTION`
map and compares with `evaluate.is_abnormal()` / `is_normal()`, so the UQ study and the
confusion matrix cannot disagree about what counts as a match.

**`diseased_lungs` is excluded.** Derived in `csv_io.derive_diseased_lungs()` from ten
other findings, never judged by the model — so it has no CE score and no token
probability. Including it for SC alone would give one row in twenty a different set of
available proxies. **The analysis covers 19 findings, so 50 × 19 = 950 rows per provider
per tier, not 1000** (corrects the figure quoted in D1).

**Partial cases** keep whatever replicates survived, with `n_replicates` recorded; rows
below 3 are dropped, matching the paper's exclusion of error responses.

### D9 — Statistics

**Decided 2026-08-08.**

`uncertainty/stats.py` — pure functions over `confidence`, `correct` and `case_id`
arrays. No file I/O, no plotting.

**ROC AUC with a clustered bootstrap CI.** Point estimate from
`sklearn.metrics.roc_auc_score`. The interval resamples **cases** with replacement,
taking all 19 findings of each drawn case. Row-level resampling would pretend there are
950 independent observations when there are 50 clusters, giving intervals that are too
narrow. This is D1's caveat cashed in.

Two departures from the authors' notebook: 1,000 bootstrap iterations rather than 4,000
(enough for a percentile interval, 4× faster to iterate against), and percentiles
2.5/97.5 rather than their 5/95 — their code takes the 5th and 95th percentiles and
prints the result as `"95% Confidence Interval"`, which is a 90% interval.

**ECE with adaptive binning.** `ECE = Σ_b (n_b/N) · |mean_conf_b − acc_b|`. The authors
bin with `pd.qcut(confidence, 10)`. SC confidence at N=5 has exactly three distinct
values, so `qcut` into 10 bins raises `ValueError: Bin edges must be unique` — a
certainty, not a risk. Binning therefore adapts: bin by distinct value when the count of
distinct values is at or below the requested bin count, otherwise use quantile bins. SC
gets 3 bins, CE up to 10, one function handles both.

**Brier score** is `mean((confidence − correct)²)` — no binning, nothing to break.

**The 0.7 threshold** is reported, not enforced. The paper computes calibration only for
proxies clearing ROC AUC 0.7; at N=5 that may be nowhere, so calibration is computed for
everything and a `meets_threshold` column flags the ones that cleared it.

**Outputs:** `uncertainty_results.xlsx` with one sheet per provider (tier, proxy, n, AUC,
CI bounds, ECE, Brier, mean confidence, observed accuracy, meets 0.7), plus
`figures/calibration_{provider}_{tier}.png` in the paper's Figure 5 layout and
`figures/roc_{provider}.png`.

**New dependencies:** `scikit-learn` and `matplotlib` in `requirements.txt`.

### D10 — Error handling and tests

**Decided 2026-08-08.**

**Sampling never aborts.** A case exhausting all 10 retries writes a line to
`failures_{provider}_{tier}.jsonl` and the run continues — a Kimi run is ~$15 and one bad
case must not cost it. The cost guard (D7d) is the only thing that stops a run.

**Analysis fails loudly and early**, matching `evaluate_main.py:67`'s
`raise SystemExit("No prediction files found - run main.py first.")`. Missing sample
files name the provider, the tier, and the command that produces them.

**Degenerate groups are reported, not crashed.** No variation in correctness →
`N/A (no incorrect answers)` instead of a `roc_auc_score` exception. One distinct
confidence value → AUC is 0.5 by construction, and the cell says so.

**Tests** follow `tests/test_classify.py` conventions — fakes, no API, no files.

- `tests/test_proxies.py` — SC majority arithmetic; CE `90` → `0.90`;
  **TLP logprob `−0.03` → `0.970`, not `0.933`** (pins the `exp` vs `pow(10,·)` decision);
  gold `"Abnormal"` vs predicted `"abnormal"` scores correct; 2-replicate case dropped;
  `diseased_lungs` never appears.
- `tests/test_stats.py` — perfect separator → AUC 1.0; constant confidence → AUC 0.5;
  perfectly-calibrated synthetic set → ECE ≈ 0; confident perfect predictor → Brier 0;
  **three distinct values with 10 bins requested does not raise** (regression test for the
  `qcut` crash); clustered bootstrap gives a wider interval than row-level resampling on
  the same data, proving the clustering is real.
- `tests/test_uq_sample.py` — resume requests exactly the shortfall; a failing case writes
  to the failures file and the run continues; `--dry-run` makes zero calls.

**Acceptance criterion for the one change to existing code:**
`tests/test_classify.py` must pass unmodified.

**Security:** no new secrets and no new key handling. Every module in `uncertainty/`
takes an already-built model as an argument exactly as `classify_case` does, so nothing
in the new package touches a key. `.gitignore` gains `dataset_LLM_uncertainty/` and
`figures/`.

---

### D11 — Measured tokens, and dropping Mistral's duplicate tier

**Measured 2026-08-08** by the Task 11 smoke run: 36 calls, 3 providers × 3 tiers × 2
cases × 1 replicate, ~$0.34. **No failures anywhere.**

Mean tokens per replicate call, slim `CaseLabels` schema:

| provider | tier | effort | input | output |
|---|---|---|---|---|
| openai | low / medium / high | `low` / `medium` / `high` | 1009 | 416 / 474 / 904 |
| mistral | low | `none` | 1018 | 360 |
| mistral | medium / high | `high` (same config) | 1018 | pooled 3795 |
| kimi | low / medium / high | `low` / `high` / `max` | ~1173 | 308 / 1018 / 1330 |

**Three risks cleared.** Mistral accepts `none`, and it plainly works — 360 output tokens
against 3795 at `high`. No provider silently ignores the effort knob; all three show
clear separation across tiers. CE works live on all three, returning 19 valid scores
every time.

**An early sighting of the paper's headline.** Mean CE scores across all 18 CE calls ran
**83–100**. That is severe over-confidence before a single case has been scored.

**The estimator was restructured, not just re-tuned.** `BASELINE_TOKENS ×
SLIM_OUTPUT_FACTOR × EFFORT_OUTPUT_MULTIPLIER` is replaced by `MEASURED_TOKENS` and
`MEASURED_CE_TOKENS`, indexed per provider per tier. A single global multiplier could not
represent what was measured: the low tier is 0.88× medium on openai but **0.06×** on
mistral, whose low tier disables reasoning outright. CE output likewise spans 259–3171
tokens across tiers where the old estimate assumed a flat 800. Four constants removed,
replaced by data.

**The measurements moved the budget a long way:**

| provider | estimated | measured | budget | |
|---|---|---|---|---|
| openai | $1.31 | **$0.78** | $10 | ok |
| mistral | $7.03 | **$18.50** | $10 | **over** |
| kimi | $13.98 | **$14.96** | $20 | ok |

`test_full_run_lands_inside_the_budget` is what caught it — written speculatively in
Task 2, it went red on the honest numbers, which is exactly its job.

**Decision: Mistral runs low + high only.** Its medium tier maps to the same
`effort=high` string, so it was a duplicate condition kept as a sampling-noise control
while that was free. Measured at $8.61 it is not worth that. `PROVIDER_TIERS` now encodes
per-provider tier sets; Mistral lands at $9.76.

**Rejected.** Switching to `mistral-medium-3` at $0.40/$2.00 (~$4.94, but an older model
and unverified effort support); cutting Mistral to 25 cases (ragged dataset, CIs widen
~1.4×); dropping Mistral entirely.

**Caveat carried in the config.** Mistral's 3795 is the mean of `[2033, 9622, 1870,
1654]` — one runaway reasoning trace inflates it. The mean rather than the median is used
deliberately: a budget guard should over-estimate, and outliers are what you get billed
for. It also means $9.76 is a conservative ceiling; the median implies nearer $7.

**Consequence for the results table.** Mistral has two tiers where the others have three,
and the sampling-noise floor is no longer measured. One footnote.

### D12 — First results: OpenAI (`gpt-5.6-luna`)

**Run 2026-08-08.** 750 replicates + 150 CE across three tiers, zero failures, ~$0.78.
5,700 proxy rows. Bootstrap 1000, clustered by case.

| tier | proxy | n | AUC | 95% CI | ECE | Brier | mean conf | obs. acc | ≥0.7 |
|---|---|---|---|---|---|---|---|---|---|
| low | SC | 950 | 0.627 | 0.557–0.723 | 0.025 | 0.025 | 0.997 | 0.972 | no |
| low | **CE** | 950 | **0.787** | 0.641–0.891 | 0.019 | 0.024 | 0.956 | 0.974 | **yes** |
| medium | SC | 950 | 0.608 | 0.534–0.682 | 0.026 | 0.026 | 0.997 | 0.972 | no |
| medium | **CE** | 950 | **0.820** | 0.713–0.898 | 0.014 | 0.024 | 0.957 | 0.972 | **yes** |
| high | SC | 950 | 0.628 | 0.551–0.698 | 0.026 | 0.026 | 0.997 | 0.972 | no |
| high | **CE** | 950 | **0.775** | 0.679–0.867 | 0.017 | 0.025 | 0.956 | 0.971 | **yes** |

**CE beats SC at every tier, and only CE clears 0.7. This reverses the paper**, which
found SC best (ROC AUC 0.68–0.79) and CE worst.

**The mechanism is in the data, not speculation.** SC confidence is exactly `1.0` on
**2,817 of 2,850 rows (98.8%)** — only 33 rows saw any disagreement across five
replicates. SC measures *stochastic* variation between runs; on closed-set extraction
from a fixed text, a reasoning model is very nearly deterministic, so SC has almost
nothing to measure and saturates. CE spans 0.25–1.0 and keeps its signal. The paper's
open-ended free-text task had real answer variability for SC to detect; this one does
not. See [[D5 result]] for the same theme — the proxies that depend on the model
exposing variation are the ones that fail here.

**Reasoning effort changed essentially nothing:**

- **948 of 950 labels identical** between low and high effort (99.8%)
- **26 of 27 errors identical** across all three tiers; low and medium share the *same*
  error set exactly
- observed accuracy 0.972 at every tier

The paper reported no change in discriminative performance across temperature settings.
The effort substitution reproduces that, more starkly — the axis is real (token counts
differ 2.2× between low and high, D11) but it does not move the answers.

**Three caveats that must appear in any write-up.**

1. Accuracy is 97.2%, so the AUC rests on **27 errors per tier**. That is what the wide
   CIs are saying (CE low: 0.641–0.891).
2. The excellent ECE (0.014–0.026) is partly arithmetic: when accuracy is 97% and stated
   confidence is 96%, they agree almost by coincidence rather than by calibration.
3. CE here is slightly **under**-confident (0.956 stated, 0.974 observed) — the reverse
   of the paper's central finding, and again a consequence of the high base rate. Note
   this contradicts the raw CE scores looking extreme (83–100 in D11): a mean score of 96
   *is* well calibrated when the model is right 97% of the time.

**Clinically:** `bronchitis` accounts for 26 of the errors across tiers, well ahead of
`Alveolar_interstitial_pattern` (12) and `pulmonary_nodules` (10).

### D13 — CE had no retry (fixed mid-run)

**Found 2026-08-08 during the Kimi run.** Kimi intermittently wraps structured output in
a markdown fence, so the payload begins with ` ``` ` and fails JSON parsing. Its
*replicate* calls hit the same thing and were unaffected — 250/250 on the low tier —
because they route through `classify_case`, which retries ten times. `elicit_confidence`
invoked the model directly, so one fenced response permanently lost that case's CE.
~11% of CE calls were being dropped.

Fixed by extracting the loop into `classify.invoke_structured` and routing both callers
through it, rather than giving CE a second copy that would drift. This is the same
argument that put the `schema` argument on `classify_case` in D7b; CE should have gone
through it then, and the plan's omission is why it did not.

Cheap to repair thanks to D7's resume design: `plan_run` tracks replicates and CE
separately, so a resume pass re-requests only the failed CE calls.

### D14 — Final results, all three providers

**Run complete 2026-08-08.** 2,000 replicate calls + 400 CE calls across 8 provider-tier
combinations, every cell at 250/250 replicates and 50/50 CE. 15,200 proxy rows.

**Actual spend $24.62** against a $25.50 estimate and $40 of credit:

| provider | input | output | actual | estimated | budget |
|---|---|---|---|---|---|
| openai | 888,660 | 478,862 | $0.75 | $0.78 | $10 |
| mistral | 603,892 | 1,194,386 | $9.86 | $9.76 | $10 |
| kimi | 1,033,302 | 727,100 | $14.01 | $14.96 | $20 |
| | | | **$24.62** | $25.50 | $40 |

Every provider landed within 4% of its post-measurement estimate. Mistral came within
$0.14 of its $10 ceiling — the decision in D11 to drop its duplicate medium tier was
what made it fit.

#### Result 1 — CE beats SC everywhere. This reverses the paper.

| | SC AUC | CE AUC |
|---|---|---|
| openai low / medium / high | 0.627 / 0.608 / 0.628 | **0.787 / 0.820 / 0.775** |
| mistral low / high | 0.591 / 0.558 | **0.914** / 0.617 |
| kimi low / medium / high | 0.617 / 0.656 / 0.637 | **0.815 / 0.777 / 0.795** |

SC clears the paper's 0.7 usefulness threshold in **zero of eight** cells. CE clears it in
**six of eight**. Savage et al. found the opposite ordering (SC 0.68–0.79 best, CE worst)
on every model they tested.

#### Result 2 — why SC fails here, and why "fails" is the wrong word

SC is not uninformative. It is **high-precision, low-coverage**:

| provider | rows where all 5 replicates agreed | accuracy when unanimous | accuracy when split |
|---|---|---|---|
| openai | 98.8% | 0.978 | **0.394** (n=33) |
| mistral | 98.2% | 0.976 | **0.735** (n=34) |
| kimi | 98.0% | 0.980 | **0.607** (n=56) |

When these models disagree with themselves, accuracy collapses from ~98% to 39–74%. That
is a very strong error signal — it just fires on about 2% of rows. ROC AUC rewards
ranking across the whole distribution, so a proxy that is constant on 98% of rows scores
poorly no matter how sharp it is on the rest.

**The practical reading is the opposite of the AUC's:** "review every finding where the
model contradicted itself across five runs" would catch a real share of errors at a very
low false-alarm rate. AUC is the wrong lens for a low-coverage detector.

The cause is task shape, not model quality. SC measures *stochastic* variation between
runs. Closed-set extraction from a fixed report leaves reasoning models nearly
deterministic, so there is little variation to measure. The paper's open-ended free-text
diagnosis had genuine answer variability. This is the same theme as [[D5 result]]: the
proxies that need the model to expose variation are the ones that degrade here.

#### Result 3 — reasoning effort does almost nothing

| provider | accuracy by tier |
|---|---|
| openai | low 0.972 · medium 0.972 · high 0.972 |
| mistral | low 0.966 · high 0.977 |
| kimi | low 0.975 · medium 0.974 · high 0.971 |

OpenAI is identical to three decimal places across all three tiers; 948 of 950 labels are
the same between low and high, and 26 of its 27 errors are identical across tiers. Kimi
gets very slightly *worse* with more effort. Only Mistral improves, by 1.1 points.

The axis is real — D11 measured 2.2× to 9× more output tokens at high effort — so the
models genuinely think harder. It just does not change the answers. This reproduces the
paper's finding of no discriminative change across temperature settings, more starkly.

One notable interaction: **Mistral's CE discrimination collapses with effort**, 0.914 at
`none` down to 0.617 at `high`. Its best uncertainty estimate came with reasoning
switched off entirely.

#### Result 4 — the models are under-confident, not over-confident

CE stated confidence minus observed accuracy:

| provider | low | medium | high |
|---|---|---|---|
| openai | −0.017 | −0.014 | −0.015 |
| mistral | **+0.016** | — | −0.008 |
| kimi | −0.100 | −0.100 | −0.095 |

Seven of eight cells are under-confident. The paper's central calibration finding was that
"LLMs are consistently over-confident when verbalizing their confidence."

**This is where it would be easy to report the opposite of what happened.** The raw CE
scores look alarming — means of 87–99, and 83–100 in the smoke run. But a model stating
96% confidence *is* well calibrated when it is right 97% of the time. Savage et al. found
over-confidence because their models were right far less often on open-ended diagnosis.
The high base rate here does not merely add noise; it inverts the direction of the
finding.

Kimi is the outlier at −0.10, systematically understating confidence it has earned.
OpenAI is the best calibrated (ECE 0.014–0.019).

#### Standing caveats

- Accuracy is ~97%, so each AUC rests on roughly 25–30 errors per cell. The CIs say so
  (mistral/high CE: 0.511–0.717).
- N=5 was a budget decision (D2). SC confidence takes only three values, which caps how
  finely it can rank even where it does have signal.
- One dataset, 50 cases, one species, one modality. The paper used 723 questions across
  three datasets.
- Correctness is agreement with a single manual annotation pass, not the paper's two
  blinded physicians with a third adjudicating.

---

## Reference tables

### Prices per 1M tokens (checked 2026-08-07)

| Model | input | cached input | output |
|---|---|---|---|
| `gpt-5.5` | $5.00 | $0.50 | $30.00 |
| `gpt-5.6-terra` | $2.00 | $0.20 | $12.00 |
| `gpt-5.6-luna` | $0.20 | $0.02 | $1.20 |
| `mistral-medium-3.5` | $1.50 | — | $7.50 |
| `mistral-medium-3` | $0.40 | — | $2.00 |
| `kimi-k3` | $3.00 | $0.30 | $15.00 |

Cost estimates exclude retries. `classify_case` retries up to 10× and failed attempts
still bill; the 2026-08-02 Mistral run had 17/50 failures. Budget 10–20% headroom.

---

## Implementation progress

| Task | Status | Commit | Tests |
|---|---|---|---|
| — | Phase 1+2 baseline committed | `18b5db7` | 39 |
| 1 | Slim schemas, generic `classify_case` | `df92567` | 47 |
| 2 | Config, effort table, cost estimator | `042deb4` | 56 |
| 3 | SC and CE proxy primitives | `0beb4f3` | 63 |
| 4 | Row assembly, gold join, exclusions | `7ee2274` | 75 |
| 5 | ROC AUC, case-clustered bootstrap | `3500883` | 84 |
| 6 | Adaptive-bin ECE, Brier, calibration points | `8564b38` | 95 |
| 7 | JSONL persistence and resume | `dec03b1` | 105 |
| 8 | Replicate call, two-step CE call, CE prompt | `55b5fa3` | 113 |
| 9 | Sampling CLI, cost guard, thread pool | `90944e9` | 128 |
| 10 | Logprobs probe — **run, all three NO** | `2fa6ddd` | 136 |
| 13 | Analysis CLI, workbook, calibration figures | `28e3ed4` | 147 |
| 11 | Tier/token calibration — **run, ~$0.34** | `f3d4214` | 155 |
| — | CE retry fix, found mid-run | `e28a346` | 156 |
| 12 | **The paid run — complete, $24.62** | | |
| 14 | **skipped** — TLP not buildable, see D5 result | — | — |

All free work is done. The analysis half was verified end to end on synthetic JSONL
before any sampling: 50 cases, 950 rows per group, SC AUC rising 0.73 → 0.84 → 0.93
across tiers, and a rendered calibration figure showing CE below the diagonal — the
paper's over-confidence result. `--bootstrap` was added to `uq_analyze_main.py` because
the interval costs ~3.2s per group at 1000 iterations (about a minute for three
providers); lower it while iterating, leave it at the default for published numbers.

**Spent so far: ~$0.05** (the probe). Remaining budget ~$22 against ~$40.

**Added beyond the plan in Task 9:** `tests/test_uq_main.py`. `run_one_tier` decides how
many calls to make, so it is the one code path where a bug costs real dollars, and the
plan had no test for it. Eight tests drive it end to end against a counting fake model.
They passed first time — a weak signal — so they were mutation tested: breaking the
shortfall subtraction so it ignores existing work turns three of them red.

**A third correction, from Task 5.** The plan's bootstrap fixture was perfectly separable,
so every draw returned AUC 1.0, both intervals collapsed to zero width, and the
"clustered is wider" assertion compared `0 > 0`. Fixing it surfaced something the design
had glossed: modelling only *correctness* at the case level is not enough either — the
confidences stay independent within a case, so a drawn case behaves like 19 independent
rows and clustering buys nothing (measured: clustered 0.086 vs row 0.103, i.e.
*narrower*). The real structure needs a case-level random effect on **confidence** too,
since all 19 findings are read off one shared report. With that, clustered intervals come
out ~4× wider (0.43 vs 0.10) across every seed tried.

Practical upshot for D9: row-level resampling would have reported ±0.05 where the honest
interval is ±0.20.

**Two corrections Task 1 forced**, both recorded in `plan.md`:

1. `CaseResult.classification` was pinned to `CaseClassification`, so returning a
   `CaseLabels` raised `ValidationError`. `CaseResult` is now generic over the answer
   schema (`CaseResult[SchemaT]`), which keeps `main.py`'s type precision while admitting
   the slim shape. D7b understated the change as "one default argument"; it is two.
2. The retry-backoff sleep patch moved from `tests/test_classify.py` into `conftest.py`.
   A new test module did not inherit the module-local fixture, so the failure above slept
   5+10+20+40+60×6 ≈ seven minutes before reporting. The symptom (a hang) looked nothing
   like the cause (a validation error).

Measured estimator output at 50 cases × 5 replicates × 3 tiers: openai $1.31, mistral
$7.03, kimi $13.98 — **$22.31 total**, against the $23.50 hand estimate in D3.

## Open items

- [x] ~~Task 13 — `uq_analyze_main.py`~~ — done 2026-08-08
- [x] ~~Task 11 — smoke run per tier~~ — done 2026-08-08, ~$0.34, see D11
- [x] ~~Task 12 — the paid run~~ — done 2026-08-08, **$24.62**, see D14
- [x] ~~Record real spend~~ — in D14; every provider within 4% of estimate

**The study is complete.** Everything below is optional follow-on work.

- [ ] Decide whether to re-run base classification on `gpt-5.6-luna` for consistency with D3
- [ ] Consider reporting SC as a low-coverage error detector (precision/recall at the
      "not unanimous" threshold) alongside its AUC — see D14 Result 2. AUC understates it
      badly, and the practical framing is more useful than the headline number.
- [ ] Raise N to 15 to test whether SC's coverage problem is an artefact of N=5 or
      intrinsic to the task. At ~3× the budget this is the single most informative
      follow-up.
- [ ] Investigate why Kimi is systematically 10 points under-confident where the other
      two are within 2 points
- [x] ~~Run the logprobs probe (D5)~~ — done 2026-08-08, all three NO, Task 14 skipped

## Changelog

- **2026-08-07** — D1–D7 recorded. Paper, supplementary data and authors' code added to
  `.gitignore`.
- **2026-08-08** — D8–D10 recorded. Design approved and written to
  `uncertainty-quantification-design.md`. Corrected the row count in D1 from 1000 to 950
  (19 findings, not 20 — `diseased_lungs` is derived, see D8).
- **2026-08-08** — Implementation plan written as Phase 3 of `plan.md`, 14 TDD tasks.
  Two changes to D6's module list came out of planning: `build_tier_model` lives in a new
  `uncertainty/llm.py` rather than in `uq_main.py`, so `probe.py` can import it without a
  root script importing back into the package; and `uq_main.py` appends its real token
  spend to `logs/token_usage.md`, keeping both pipelines' costs in one file.
