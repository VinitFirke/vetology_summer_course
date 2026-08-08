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

## Reference tables

### Measured token baselines

From `logs/token_usage.md`, at medium effort, per case:

| Model | input | output |
|---|---|---|
| `gpt-5.5` | 1,064 | 1,384 |
| `mistral-large` | ~1,050 | ~1,050 |
| `kimi-k3` (at high) | 1,208 | 1,712 |

Effort multipliers on output are **estimates**: low ≈ 0.5×, high ≈ 2.5× vs medium.
To be replaced with measured values from a 2-case smoke run per tier before the full run.

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
| 11–12 | not started — **both spend money** | | |
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
- [ ] Task 11 — smoke run per tier (~$0.50): replace estimated effort multipliers with
      measured ones, **and** verify each provider accepts its three effort strings (D4),
      especially Mistral's `none`. If a provider's low and high output token counts match
      within ~10%, the effort knob is being ignored.
- [ ] Task 12 — the paid run (~$22)
- [ ] Decide whether to re-run base classification on `gpt-5.6-luna` for consistency with D3
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
