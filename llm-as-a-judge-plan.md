# LLM-as-a-Judge — integration plan for the radiology label pipeline

**Date:** 2026-08-14
**Status:** proposal, for discussion
**Source paper:** Gu J, Jiang X, Shi Z, et al. *A Survey on LLM-as-a-Judge.* arXiv:2411.15594v6, October 2025.
**Companion documents:** `deployment-plan.md` (productionisation, 2026-08-12) ·
`uncertainty-quantification-design.md` (UQ study) · `catalog.md` (decision log)
**Structured after** `REFERENCE.md`, like the other design docs in this repo.

> This document does not repeat `deployment-plan.md`. That one answers *"how do I ship
> this?"*. This one answers *"what does the survey tell me I have actually built, and
> what is provably missing from it?"* — and then re-scores the Vetology pitch in light of
> the answer. Where the two overlap, this document says so and points at the section.

---

## 0. The one-sentence version

You did not build a classifier that happens to use an LLM. You built an
**LLM-as-a-Judge system for data annotation** — the survey's §2.4.2 — and you have
implemented three and a half of the ten components of the reliability wrapper the survey says
such a system needs. Most of what is missing costs **$0**, because the data is already on
disk — and one piece of it is a gap the survey explicitly names as under-researched.

---

## 1. Design what you are building

### 1.1 The paper's formalism, filled in with your files

The survey opens with a formal definition (§2):

```
E  ←  P_LLM ( x ⊕ C )
```

| Symbol | Paper's meaning | Your system |
|---|---|---|
| `x` | the input being evaluated | the free-text radiologist report — `RadiologyCase.findings_text` + `.conclusions_text`, read from the two `(original radiologist report)` columns in `classifier_multi/schemas.py` |
| `C` | the context / prompt template | `classifier_multi/prompts/{category}.json` + the finding list injected by `prompt.finding_list_text()` + the `Field(description=...)` strings shipped inside the JSON schema |
| `⊕` | the combination operator | `render_messages()` — system message, then user template |
| `P_LLM` | the model | `llm.build_model()` across four providers |
| `E` | the final evaluation | 19 / 19 / 10 `{finding, label, evidence, reasoning}` objects per category, constrained by `classification_schema()` |

Then the paper introduces a **second, stronger** definition — the one the whole survey is
organised around:

```
R  ←  f_R ( P_LLM , x , C )
```

> *"`f_R`: A series of constraints and validation methods applied systematically to the
> basic LLM-as-a-Judge framework to enhance evaluation reliability. These include methods
> to mitigate biases, control variability, and confirm robustness against adversarial
> inputs."* — §2

**`f_R` is the thing you are missing a name for.** You have built parts of it without
calling it that:

| `f_R` component | Status in this repo |
| --- | --- |
| Structured/constrained output | ✅ `with_structured_output` + Pydantic enum — the paper's §2.3.1 constrained decoding |
| Control variability (multi-round) | ✅ `uncertainty/sample.py`, N=5 replicates — the paper's §3.3.1 "summarize by multiple rounds". **Feline thorax only, 3 providers** — it imports `classifier`, not `classifier_multi` |
| Confidence signal | ✅ CE in `uncertainty/proxies.py` — §2.1.1 score generation |
| Agreement with human judgment | ⚠️ partial — `classifier_multi/evaluate.py` gives sensitivity/specificity but **not** the paper's named metrics (§4.1) |
| Multi-model integration | ❌ four providers' labels sit on disk, never combined (§3.3.1) |
| Bias measurement | ❌ nothing (§4.2 — twelve named bias types) |
| Adversarial robustness | ❌ nothing (§4.3) |
| Self-verification | ❌ evidence quotes collected, never checked (§3.3.2) |
| Cascade / selective evaluation | ❌ designed in `deployment-plan.md` §1.3, not built (§3.3.1, Jung et al.) |
| Drift / temporal consistency | ❌ `mistral-medium-latest` is unpinned; no `model_id` in output rows (§6.5) |

Six of the ten rows are empty or partial, and **four of those six can be filled without
spending another dollar on API calls.** The two that cost money — bias and adversarial
robustness — come to about $1.10 between them.

### 1.2 Which of the four pipelines you are on

The survey (§2.4) splits LLM-as-a-Judge into four scenarios: *for models*, *for data*,
*for agents*, *for reasoning*. Almost all published meta-evaluation work — MTBench,
Chatbot Arena, FairEval, JudgeBench, all of Table 1 — is **for models**: which chatbot
gave the better answer.

You are on the **for data** branch. And the paper says this, verbatim, in §4.1:

> *"Current meta-evaluation primarily focuses on LLM-as-a-judge for models, while there is
> a lack of sufficient meta-evaluation when these LLM evaluators are used for
> automatically annotating large-scale datasets (Section 2.4.2). We advocate for more
> rigorous assessment of the alignment between LLM-as-a-judge and human judgment when they
> are employed for large-scale data annotation."*

That sentence is the strongest thing in the paper for your purposes. It is a
peer-reviewed survey saying *the exact study you have run is under-supplied in the
literature*. Every table you produce from here is aimed at a stated gap rather than at a
crowded benchmark. §7.6 and §7.10 (domain-specific reliable applications, which names
medical diagnosis) say the same thing from the future-work side.

### 1.3 The reframe in one sentence for a slide

> **A reliability-instrumented LLM-as-a-Judge for veterinary radiology annotation: it
> converts free-text radiologist reports into per-condition labels, and — the part almost
> nobody builds — it measures how much you should trust each label before a human sees
> it.**

---

## 2. Paper → codebase map

The single table this whole document exists to produce. **Cost** is marginal API spend.

| # | Paper | What it says | Your position | Build | Cost |
|---|---|---|---|---|---|
| 1 | §4.1 | Agreement is measured with **percent agreement, Cohen's Kappa, Spearman**; treat judging as classification and report **precision/recall** | `evaluate.py` reports sensitivity/specificity only. At 5% prevalence accuracy is nearly meaningless — you already proved this: all-normal scores 95.1% | Add Kappa + precision/recall/F1 + prevalence column to `evaluate.py` | **$0** |
| 2 | §3.3.1 | *"using multiple LLM evaluators to assess the contents simultaneously and integrating the results… can reduce biases introduced by LLMs"* — CPAD, Bai et al. peer review, EvalMORAAL | **10 labelled CSVs sit in `dataset_LLM_classification/`, never compared to each other** — 4 providers on feline thorax, 3 each on canine thorax and canine abdomen | `judge/panel.py`: vote + disagreement flag; 4-way on feline, 3-way on the other two | **$0** |
| 3 | §3.3.2 | Self-verification (TrueTeacher): keep only results that survive a verification pass | Schema demands a verbatim evidence quote; **590/592 appear verbatim on the feline run, and none is ever checked** (the other eight runs are unmeasured) | `judge/verify.py` — one `in` test per label; assert-fail becomes a defer | **$0** |
| 4 | §4.2.2 | **Position bias** — judges favour content by position; mitigate by shuffling and re-scoring (Wang et al., Auto-J, JudgeLM, PandaLM) | Findings are always sent in fixed `asked_findings` order. Never tested | Shuffle `finding_list` per case, re-run 50 cases, measure label flips | ~$0.30 |
| 5 | §4.2.2 | **Length / verbosity bias** and **concreteness bias** (numbers, jargon, citations inflate scores) | Untested — and testable *offline* by correlating report length / numeric-token count against abnormal-call rate, controlling for gold | `judge/bias.py` correlation report | **$0** |
| 6 | §4.2.1 | **Self-enhancement / source bias** — judges prefer text produced by themselves or by machines | Your CSVs carry **both** the Vetology AI report and the radiologist report per case. Nobody has asked whether the judge labels them differently | Label both columns, compare; the cleanest bias experiment available to you | ~$0.30 |
| 7 | §4.3 | **Adversarial robustness** — null-model attacks, phrase insertion, "meaningless statement" robustness | Nothing. But you already identified the clinical analogue: hedges (`possible`, `versus artifact`, `cannot be excluded`) and negation scopes (`no evidence of`, `within normal limits`) | `judge/adversarial.py`: negation-flip and hedge-injection suites over 50 reports | ~$0.50 |
| 8 | §3.3.1 | **Cascaded Selective Evaluation** (Jung et al. [57]): cheap model first, escalate to a stronger one only when confidence is low | This is *exactly* your deferral cascade in `deployment-plan.md` §1.3 — now with a citation and a name | `judge/cascade.py`; simulate offline from existing JSONL first | **$0** to simulate |
| 9 | §3.1.1 | **Decomposition of evaluation criteria** (BSM, HD-Eval): split coarse criteria into explicit sub-criteria | Your predicted failure mode — colitis/gastritis false positives are *definition* errors, not reading errors — is a criteria-decomposition problem | One-line clinical definition per condition in the prompt; ablate | ~$0.50 |
| 10 | §3.1.1 | Absolute scoring is less robust than **relative comparison** (PARIS, Liu et al.) | Your CE proxy is absolute 0–100 self-scoring — the format the paper calls least robust | Optional: pairwise CE variant on the ~30 hardest findings | ~$0.20 |
| 11 | §2.3.2, §3.3.2 | Logit normalisation and score smoothing require token-probability access | **You measured this and all three providers said no** (`catalog.md` D5) | Nothing to build — it is a *result*, see §6.2 below | done |
| 12 | §6.5 | **Evaluation drift** — judges change across model versions; meta-evaluation must be temporal | `mistral-medium-latest` is unpinned; rows carry `provider` and `tier` but no resolved `model_id` | Stamp `model_id` + `prompt_version` in every row; freeze a 50-case regression set | **$0** now, ~$0.10/run later |
| 13 | §2.2.2, §6.1 | Fine-tuned judges (PandaLM, JudgeLM, Prometheus) avoid API privacy leakage and version opacity, at the cost of generalisation | Not applicable at n=50 — but it is the right answer at 300k cases, which matters for the pitch | Describe only | — |

**Rows 1, 2, 3, 5, 8, 11, 12 cost nothing.** Row 2 is the one to do first, and §3.2
explains why.

---

## 3. Design the user experience

### 3.1 A new package, not a rewrite

```
classifier_multi/     # unchanged — produces labels
uncertainty/          # unchanged — produces confidence
judge/                # NEW — decides whether to believe the label
```

`judge/` follows the conventions that already work in this repo: pure functions over
parsed data, the model injected rather than constructed, no file I/O inside the
computation layer, and the expensive stage separated from the iterated stage by a file.

| File | Responsibility | Pure? | Costs money? |
|---|---|---|---|
| `judge/agreement.py` | Kappa, percent agreement, precision/recall/F1, prevalence | ✅ | no |
| `judge/panel.py` | Cross-provider vote and disagreement flag | ✅ | no |
| `judge/verify.py` | Evidence-grounding check; optional self-verification call | mostly | optional |
| `judge/bias.py` | Perturbation generators + bias correlation report | generators pure | on re-run |
| `judge/adversarial.py` | Negation-flip and hedge-injection suites | generators pure | on re-run |
| `judge/cascade.py` | Deferral policy, risk–coverage, cost simulation | ✅ | no |
| `judge/drift.py` | Frozen regression set, Kappa delta, alarm | ✅ | on re-run |
| `judge_eval_main.py` | CLI — offline, free, re-runnable | — | no |
| `judge_probe_main.py` | CLI — the perturbation runs, behind the existing cost guard | — | yes |

Two CLIs, mirroring `uq_main.py` / `uq_analyze_main.py`. Same rule: **the paid command
writes JSONL and computes nothing; the free command reads JSONL and calls no API.**

### 3.2 Day one, no API key: the panel

This is the highest-value single item in the document, so it gets its own section.

`dataset_LLM_classification/` currently holds **ten labelled CSVs** — four providers on
feline thorax, three on canine thorax, three on canine abdomen — plus nine
`reasoning_*.json` files carrying a verbatim evidence quote for every label. All produced
on identical inputs. **They have never been compared to each other.**

The paper (§3.3.1) calls combining them *integrating multi-source evaluation results* and
cites three systems that do it by voting. Doing it here answers a question your $24.62 UQ
study could not:

> **Sample consistency was high-precision and low-coverage** — `catalog.md` D14 Result 2:
> 98% of rows were unanimous across five replicates, so the detector fires on 2% of rows.
> That is *within-model* variation, and reasoning models on closed-set extraction have
> very little of it.
>
> **Cross-model disagreement is a different signal.** Four models with different training
> data fail in different places. The disagreement rate should be several times higher than
> 2% — and every disagreement is a candidate error, at zero marginal cost, because the
> calls are already paid for.

Concretely, per (case, finding) across providers: `n_abnormal / n_providers`. Then the same
risk–coverage analysis `deployment-plan.md` §1.1 already runs on CE. Three outcomes, all
publishable:

1. Panel disagreement beats CE at error detection → you have a **free** proxy that
   outperforms the one that cost $24.62. That is a genuinely good result.
2. It ties CE → you have an independent confirmation, and the cheaper CE wins on cost.
3. It underperforms → you have measured that provider diversity adds nothing on this task,
   which is itself a cost-saving finding (don't pay for provider diversity).

There is no losing branch. Do this first.

### 3.3 What the reviewer sees

`judge_eval_main.py` writes one workbook and one HTML page:

- **Per-condition judge card** — prevalence, Kappa, precision/recall/F1, sensitivity with
  CI, panel agreement, deferral threshold, drift status. One row per condition, laid out
  in the same shape as Vetology's public performance table so it reads as familiar.
- **Risk–coverage curve** with the deferral operating points marked.
- **Bias panel** — one small chart per bias type with the effect size and its CI.
- **Robustness table** — negation-flip pass rate, hedge-injection stability.

---

## 4. What Vetology actually does, from their public material

Everything in this section is sourced (links at the end). It matters because two of the
four ideas in §5 depend on details that are only visible if you read their news page.

| Fact | Detail |
|---|---|
| Core product | AI screening of canine/feline radiographs, structured findings + conclusions + recommendations, returned "within minutes" |
| Scale | **91 condition classifiers** — 27 canine thorax, 25 canine abdomen, 15 feline thorax, 18 feline abdomen, 6 spine/MSK |
| Validation base | "over 300,000 multi-image patient cases" from real practices, "not synthetic datasets or cherry-picked examples" |
| Ground truth | cases "diagnosed by United States board-certified veterinary radiologists" |
| Published metrics | sensitivity, specificity, 95% CIs, AUC, PPV/NPV, full confusion matrix, accuracy, balanced accuracy, **Matthews correlation coefficient**, prevalence, ground-truth counts, clinical urgency, model release date |
| Jan 2026 | First vet imaging AI company to publish full classifier performance metrics. President Eric Goldman: *"Complete transparency isn't a competitive advantage we're protecting, it's a professional obligation"* |
| **Mar 31, 2026** | Validation dashboard expanded **from 4 to 11 metrics** across 89+ classifiers, with **31 retrained models** |
| Cadence | "releasing new classifiers monthly"; monthly re-testing of existing ones |
| Second product | White-labelled teleradiology platform for private-practice radiologists — includes an **optional custom language model trained on the radiologist's historical cases** that "assists in generating conclusions and recommendations based on your findings" |
| Third product | Teleradiology read services by board-certified radiologists |
| Integrations | ezyVet, DaySmart Vet, VetRocket; AI scribes **ScribbleVet** (Oct 2025) and **CoVet** (Jan 2026) push SOAP notes into radiology requests |
| Validation partners | AMC New York, Tufts University; publicly invites independent validation |

**Three readings that matter:**

1. **Their bottleneck is labelled ground truth, and it recurs.** 31 models retrained in a
   single March release; new classifiers monthly. Each retrain needs a fresh scored test
   set, and each score is a human converting a free-text radiologist report into
   per-condition labels. That cost is paid again every month, forever.
2. **They have staked the brand on measurement, then raised the bar themselves.** Going
   4 → 11 metrics in ten weeks is a company that will keep adding metrics. The next
   obvious one is not another classifier metric.
3. **They already ship a generative LLM into a clinical workflow** — the custom language
   model writing conclusions on the radiologist platform — and nothing public describes a
   confidence, verification, or deferral layer on it.

---

## 5. Strategic advisor: four ways to add value

Ordered by how directly each attaches to something they have publicly committed to.

### Idea 1 — Be the measurement instrument behind their transparency programme

**Their problem.** Every number on the performance page rests on somebody hand-labelling a
radiologist's report. 91 classifiers, 300k cases, monthly releases, 31 retrains in one
month. That is a permanent, growing annotation cost, and it is the rate limiter on how
fast they can ship classifiers.

**What you bring.** A judge that reproduces the labelling step, with a measured deferral
policy that says which outputs need no human at all. Your own numbers (`deployment-plan.md`
§1.1): the *same* policy — one extra CE call per finding, defer below 0.90 — buys
**62% auto-accepted while catching 85% of all errors** on Kimi, or **91% auto-accepted
catching 46%** on OpenAI. Same mechanism, different operating point, chosen by which
provider you run. Those are the numbers that turn "an LLM can label reports" into "you can
retire most of this queue." Quote the catch rate alongside the auto rate every time — 91%
coverage looks better than 62% until you notice it catches half as many errors.

**Why it lands.** It does not ask them to trust your model over theirs. It asks them to
trust your model over their annotation backlog. Different, much easier sale.

**The ask.** Stratified enrichment. Half your conditions have zero positive cases in 50
consecutive reports, so sensitivity is undefined for them — pull n≥20 positives per
condition from the archive. That is a query against their case store, not a modelling
change, and only they can authorise it.

### Idea 2 — Metric #12: reliability of the ground truth itself

**Their problem.** All eleven published metrics measure *the classifier*. None measures
*the labelling that produced the ground truth those metrics are scored against*. Every
sensitivity figure inherits an unmeasured error bar from a single annotation pass. A
sceptical radiologist or an academic partner will eventually ask, and the honest answer
today is "one reviewer, unadjudicated."

**What you bring.** The survey's §4.1 metric set applied to the annotation layer:
Cohen's Kappa between the automated judge and the human annotator, per condition, with
prevalence beside it. Plus the §4.2 bias audit and the §4.3 robustness suite. Two
publishable statements fall out:

- *"Human/automated agreement on condition X is κ = 0.__ (n = __)"* — an inter-rater
  reliability number where there is currently none.
- *"The labelling layer is stable under report re-ordering, verbosity inflation, and
  hedge-phrase injection"* — a robustness claim nobody in veterinary AI is making.

**Why it lands.** It is the natural twelfth metric for a company that went from 4 to 11 in
ten weeks and calls transparency a professional obligation. And it is *cheap* — most of it
is arithmetic over data they already hold.

**Honest caveat to state on the call:** a judge agreeing with an annotator is not the same
as either being right. Kappa measures agreement, not truth. That framing is what stops
this being over-sold.

### Idea 3 — A confidence and verification layer on the radiologist platform's language model

**Their problem.** The radiologist platform offers a custom LLM that drafts conclusions and
recommendations from a radiologist's findings, trained on their historical cases. The
promise is "maintain complete authorship and control." But a generated conclusion arrives
with no signal about which sentences are well grounded in the findings and which the model
inferred. The radiologist has to read every word with equal suspicion — which erases much
of the time saving the feature exists to deliver.

**What you bring.** Two mechanisms straight out of the survey, both of which you have
either built or can build in days:

- **Evidence grounding** (§3.3.2, self-verification): every generated claim must quote the
  finding text it came from, and the quote is checked verbatim against the source. You
  already collect these quotes — 590 of 592 appear verbatim, and the two that don't are
  both false positives on conditions with zero gold positives. A one-line assertion would
  have caught them. Applied to generated conclusions, this becomes a UI feature:
  ungrounded sentences are flagged, grounded ones are highlighted back to their source.
- **Cascaded selective evaluation** (§3.3.1, Jung et al.): cheap pass on everything,
  escalate only what the confidence signal flags. Bounded cost, bounded latency.

**Why it lands.** It attaches to a feature they already ship, it makes the "AI + Human"
positioning literal, and it argues for *more* radiologist authority rather than less —
which is what their own copy says they care about.

### Idea 4 — Disagreement triage: turn evaluation exhaust into a work queue

**Their problem.** AI report and radiologist report exist for the same case. Today the
comparison happens once, in aggregate, to produce a performance metric. The per-case
disagreements — the interesting ones — are not surfaced to anyone.

**What you bring.** Label both reports with the judge, diff them, rank the disagreements by
panel confidence. Three products fall out of one pipeline:

- **QA queue** — cases where AI and radiologist disagree *and* the judge is confident,
  ranked. That is a triage list for the validation team.
- **Drift monitor** — with 31 models retrained in a month and monthly releases, run the
  frozen regression set after each release and alarm on a Kappa shift. The survey calls
  this *evaluation drift* (§6.5) and flags it as under-addressed. For a company shipping
  monthly, it is a real operational need, not a paper concern.
- **Training signal** — adjudicated disagreements append to the gold standard, thresholds
  re-tune on the grown set, and the system gets cheaper the longer it runs.

**Why it lands.** It is the only idea here that creates a new artefact rather than
improving an existing one, so it is the right thing to *mention* and the wrong thing to
build in ten days. Show the diagram; don't promise the product.

**Longer-horizon note worth one sentence on the call:** §2.2.2 of the survey observes that
API judges introduce privacy exposure and version opacity, and that fine-tuned judges fix
both. At 300k cases Vetology has more than enough data to fine-tune a small open-weights
judge that never sends a client's report to a third party. That is a two-year idea, not a
ten-day one, but knowing it exists is a differentiator.

---

## 6. Full-stack developer: presentable, scalable, deployable

### 6.1 Making the classification presentable

The current output is `confusion_matrix_*.xlsx`. It is correct and nobody will look at it.
Three changes, in order of return:

**a) The judge card.** One HTML page per category, one row per condition, columns in the
same order Vetology publishes theirs — prevalence, TP/FN/TN/FP, sensitivity + CI,
specificity, Kappa, panel agreement, deferral threshold, `model_id`, `prompt_version`,
last-run date. Generated by `judge_eval_main.py`, not hand-assembled. It reads as familiar
in about four seconds because it mirrors a page they already own. Worth knowing:
`classifier/evaluate.py` already computes Wilson intervals — that column exists, it just
never reaches an output anyone reads.

**b) The single-case view — the thing they'll actually click.** Paste a report, get back
the source text with **each evidence quote highlighted in place**, colour-coded by label,
with a confidence bar and a review flag per condition. This is nearly free to build: the
schema already returns a verbatim quote, so highlighting is `str.find` — with a fallback
for the ~0.3% that are not verbatim, which is exactly the hallucination gate from
`judge/verify.py` surfacing in the UI. It is the most
persuasive artefact in the whole project because it makes the model's reasoning legible
rather than asserted — which is precisely the survey's §6.4 complaint about judges being
opaque black boxes.

**c) Retire accuracy from every headline.** All-normal scores 95.1% on your data. Lead with
prevalence, then Kappa, then precision/recall. `deployment-plan.md` §4 already made this
argument; the survey (§4.1) supplies the citation.

Two things to *not* do: no dashboard framework, and no "explore the data yourself" UI. The
`REFERENCE.md` cautionary tale about the generic analytics dashboard nobody used applies
directly.

### 6.2 Results you already own that the survey lets you frame properly

Three findings in `catalog.md` gain a literature anchor from this paper. This costs nothing
and materially strengthens the write-up:

| Your finding | Survey section | The framing it unlocks |
|---|---|---|
| **D5** — no provider returns token logprobs through structured output | §2.3.2, §3.3.2 | The paper describes logit normalisation and score smoothing as standard post-processing, noting they "require the LLMs to be open-source or to provide interfaces that allow access to token probabilities." You measured that on three 2026 reasoning models the interface is gone. That is a small, dated, empirically-grounded contribution — *constrained decoding and logit access are in tension, and vendors chose constrained decoding.* |
| **D14 Result 2** — SC is high-precision, low-coverage; AUC understates it | §3.3.1 vs §4.1 | The mismatch is between *what the method is for* and *what the metric measures*. Report SC as a detector with precision/recall at the non-unanimous threshold, not as an AUC. Your own open item in `catalog.md` says this; the survey explains why it happens. |
| **D14 Result 1** — CE beats SC, reversing Savage et al. | §3.1.1 | The paper says absolute scoring is *less* robust than relative comparison, which predicts CE should be the weak one. It was the strong one — because closed-set extraction removes the answer variability SC needs. A result that contradicts two sources and has a mechanism is worth more than one that agrees with both. |

### 6.3 Scaling

The workload is throughput-bound, not latency-bound: 300k cases × ~20 conditions, tolerant
of hours, intolerant of cost. That shape dictates everything.

```
case store ──▶ queue ──▶ judge workers ──▶ results DB ──▶ judge cards
 (Postgres)   (per case)  ├─ label pass                    + risk–coverage
                          ├─ verify (evidence grounding)   + drift alarm
                          ├─ confidence (CE)
                          └─ escalate if deferred ─┐
                                                    └─▶ human review queue
```

Five levers, roughly in order of effect:

1. **Cascade before scale.** Cheap model on 100%, expensive model only on the deferred
   slice. Your own table: 91% auto-accepted at two calls per finding. Sizing the fleet
   before applying the cascade is sizing for a workload you are about to delete.
2. **Batch API** — ~50% off for 24h-tolerant work. A nightly revalidation job is exactly
   that shape.
3. **Prompt caching** — your price table shows cached input at 10× cheaper
   (`gpt-5.6-luna`: $0.20 → $0.02 per 1M). The system prompt and finding list are
   identical across every case in a category. This is the single largest structural saving
   and it needs no logic change beyond call ordering.
4. **Category as data, not code.** `classifier_multi/categories.py` already means a new
   study type is one entry, not a new branch. Three categories are defined today; going to
   Vetology's five study groups and 91 conditions is data entry, not new branching. Say this
   out loud — it is the best design decision in the repo.
5. **Idempotency you already have.** Append-only JSONL with shortfall counting on resume
   generalises directly to a queue with a `(case_id, finding, model_id, prompt_version)`
   idempotency key. Swap the file for Postgres; keep the semantics.

**Do not** reach for Kubernetes, a vector DB, model serving, or a fine-tuning pipeline.
There is no model artefact to serve — this is orchestration of API calls, and saying so
plainly is a signal in itself.

### 6.4 Deploying

Ten days, under $8, demo on synthetic reports throughout because the real dataset is
confidential.

| Day | Work | Output |
|---|---|---|
| 1 | `judge/agreement.py` + `judge/panel.py`; Kappa and the cross-provider panel over existing CSVs | **the free result** — panel vs CE risk–coverage |
| 2 | `judge/verify.py` over all nine `reasoning_*.json`; per-condition error taxonomy | grounding pass rate; the colitis/gastritis definition-error claim, proven or dropped |
| 3 | `judge/bias.py` offline half (length, concreteness); freeze the regression set; stamp `model_id` + `prompt_version` | bias correlation report; drift harness |
| 4 | Paid probes: order shuffle ($0.30), source bias ($0.30), negation-flip + hedge injection ($0.50) | ~$1.10 total; the bias + robustness tables |
| 5–6 | FastAPI `/v1/label`, `/v1/judge`, `/v1/health`; Dockerfile; GitHub Actions over the existing 156 tests | callable service, green CI badge |
| 7 | Single-case demo UI with evidence highlighting; synthetic report set; free-tier deploy | the public URL they click |
| 8 | Judge cards + risk–coverage page generated by `judge_eval_main.py` | the artefact that looks like their performance page |
| 9 | Two-page proposal: their annotation bottleneck, your instrument, the stratified-sampling ask | the document that gets the second conversation |
| 10 | README, architecture diagram, buffer | a repo a stranger can read |

**Budget:** order-shuffle ~$0.30 · source-bias ~$0.30 · adversarial suite ~$0.50 ·
criteria-decomposition ablation ~$0.50 · synthetic reports ~$1 · demo traffic ~$5 ·
hosting $0. **$7.60 — under $8.**

Gaps to close, unchanged from `deployment-plan.md` §5: HTTP layer (1 day), Dockerfile
(2h), CI (1h), `model_id`/`prompt_version` in every row (30 min), structured JSON logs
(3h), Postgres only if you go past the demo.

---

## 7. Identify ripple effects

| Item | Action |
|---|---|
| `classifier_multi/evaluate.py` | gains Kappa, precision/recall/F1, prevalence — **additive**; existing `MATRIX_COLUMNS` order is copied from the example workbook and must not change |
| `classifier_multi/classify.py` | records resolved `model_id` and `prompt_version` on `CaseResult` |
| `classifier_multi/prompts/*.json` | criteria-decomposition ablation adds a *second* versioned file per category; do not edit in place, or the ablation loses its control |
| `catalog.md` | new decision block per item, same D-numbering |
| `requirements.txt` | `scikit-learn` already present for Kappa; nothing new for the free work |
| `.gitignore` | add `dataset_LLM_judge/` |
| `tests/` | one known-answer test per pure function, per the existing convention |
| `deployment-plan.md` | still the productionisation document; this one supplies the judge layer that fills its §3 Tier A backlog |

**Two traps.** First, the prompt lives in two places — the JSON file *and* the
`Field(description=...)` strings, which ship to the model inside the JSON schema. A prompt
ablation that edits only the JSON has not fully changed the prompt. Second, if you tune the
deferral threshold on the same 50 cases you measure it on, the number is optimistic. At
n=50 a held-out split is barely viable; say so rather than quietly skipping it.

---

## 8. Understand the broader context

### Limitations of this plan

- **Every bias and robustness result will rest on 50 cases, one species, one modality.**
  These are *demonstrated methods with measured effect sizes*, not validated thresholds.
  A chart must not imply otherwise.
- **Kappa is hostile to low prevalence.** At 5% base rates it is unstable and can look
  poor where accuracy looks excellent. That is the point — but report `n` and prevalence
  beside every κ, and expect to defend it.
- **Agreement is not truth.** The judge is reading another radiologist's finished text and
  cannot see the image. A disagreement with gold is not unambiguously a model error, and
  FP=26 against FN=1 is the signature of a model inferring rather than reporting — which
  may be exactly what your persona asked it to do.
- **The Vetology bottleneck is inferred**, strongly, from their public performance page,
  their release cadence, and the shape of your data. It is not confirmed. Ask on the call
  before building a proposal around it.
- **The paper is a survey.** It gives vocabulary, taxonomy and citations. It does not
  supply a method that will lift your F1. Anything here framed as "the paper says this will
  work" is over-claiming; the correct frame is "the paper says this is the thing nobody
  measures."

### Moonshots

- **A veterinary meta-evaluation benchmark.** §7.5 calls for domain-specific
  meta-evaluation benchmarks and §7.10 names medical diagnosis. A public set of vet
  radiology reports with adjudicated labels *and* a bias/robustness perturbation suite
  would be the first of its kind in the field. Vetology has the cases, an academic
  partnership with Tufts and AMC New York, and a stated appetite for publishing.
- **A fine-tuned in-house judge** (§2.2.2) — 300k cases, no third-party API, reproducible
  across versions.
- **Self-improving gold standard** (§7.1.2 feedback loops): deferred rows go to a
  reviewer, corrections append to gold, thresholds re-tune, the system gets cheaper the
  longer it runs.

---

## Sources

- Gu J, Jiang X, Shi Z, et al. *A Survey on LLM-as-a-Judge.* arXiv:2411.15594v6, Oct 2025.
- [Vetology — AI Radiology Reports](https://vetology.net/ai/)
- [Vetology — AI Classifier Performance](https://vetology.net/ai-classifier-performance/)
- [Vetology — Platform for Private Practice Radiologists](https://vetology.net/private-practice-radiologists/)
- [Vetology — News](https://vetology.net/category/news/)
- [Vetology AI Releases Classifier Performance Metrics](https://vetology.net/vetology-ai-releases-classifier-performance-metrics/)
- [Vetology AI Becomes First Veterinary Imaging AI Company to Publicly Release Classifier Performance Metrics](https://finance.yahoo.com/news/vetology-ai-becomes-first-only-134700748.html)
- In-repo: `catalog.md` (D5, D14) · `deployment-plan.md` · `classifier-rebuild-design-notes.md` · `uncertainty-quantification-design.md`
