# Gold Standard Disease Labelling — Design

**Date:** 2026-08-18
**Source brief:** `gold_standard_prompt.md`
**Status:** approved

## Purpose

Produce a human-grade gold standard (ground truth) of normal/abnormal disease labels for
900 veterinary radiology cases, so that the four models driven by `classifier_multi/`
can be scored against it.

Every design decision below was settled with the user before any labelling began. Where
the brief was ambiguous, the resolution is recorded here rather than assumed.

## Data

Three CSVs in `dataset_gold_standard/`, all label columns empty at the start:

| Sheet | Rows | Label columns | Eligible pool |
|---|---|---|---|
| `canine_thorax_scoring_data_gold_standard(Sheet1).csv` | 3,999 | 20 (incl. `vhs`) | ~2,252 |
| `canine_abdomen_scoring_data_gold_standard(Sheet1).csv` | 3,751 | 10 | ~2,314 |
| `feline_thorax_scoring_data_gold_standard(Sheet1).csv` | 1,875 | 20 | ~1,073 |

Facts established by inspection:

- Every sheet mixes study types. Only ~56% of `canine_thorax` rows evaluate a thorax at
  all; the rest are abdominal, spinal or appendicular studies. The brief's "focused on"
  filter is therefore load-bearing, not a formality.
- Two report styles coexist: prose (`Study:` / `Findings:` / `Comments:`) and
  organ-headed (`Liver:` … / `Pulmonary parenchyma:` …).
- 1,564 CaseIDs appear in both canine sheets with byte-identical report text — one
  study scored once for its thorax and once for its abdomen.
- `feline_thorax` contains one duplicate CaseID.
- All three files are valid cp1252. `canine_thorax` holds 742 NBSP bytes and
  `feline_thorax` one `ü`; neither decodes as UTF-8.
- Row terminators are CRLF, but quoted fields contain bare LFs. Any rewrite must
  preserve both.

## Scope

- **300 cases per sheet, 900 total** (~15,000 individual finding judgements).
- Labels are written **into the original CSVs in place**. Unselected rows stay blank.

## Eligibility

A case is eligible for a sheet when its report substantively evaluates that region —
lungs, heart or pleura described for the thorax sheets; liver, spleen, GI tract or
peritoneum for the abdomen sheet. Combined thoracic/abdominal studies are eligible for
both, scored for their own region each time; the same CaseID may therefore appear in
both canine sheets.

A screening script proposes the pool. Screening is a filter, not an authority: if a case
turns out on reading not to evaluate the target region, it is dropped and replaced.

## Selection — coverage first, then random fill

1. Mine the eligible pool for candidate positives per label column, rarest first
   (`bronchiectasis`, `hypo_plastic_trachea`, `esophagitis`, `microhepatia`,
   `focal_perihilar`).
2. Read candidates until every column holds **at least 3 abnormal** cases.
3. Fill the remaining slots by seeded random sampling of the eligible pool. The seed is
   recorded so the selection is reproducible.

If the pool genuinely contains fewer than three defensible positives for a column, the
shortfall is documented rather than met by forcing a label. A wrong label corrupts the
ground truth for every model measured against it; a documented gap does not.

## Labelling rubric

| Rule | Decision |
|---|---|
| Source text | `Findings (original radiologist report)` + `Conclusions (original radiologist report)` only. The AI-report columns are never read. |
| Vocabulary | Exactly `normal` / `abnormal`, in every column including `vhs`. No third value. |
| Not mentioned, or structure not assessable | `normal` |
| Hedged findings — "suspected", "possible", "probable", "suspicious for" | `abnormal`. The radiologist saw something. |
| Hedged toward normal — "versus artifact", "versus positioning", "normal variation", "considered less likely" | `normal` |
| Disease-named columns — pneumonia, bronchitis, gastritis, colitis, pancreatitis, esophagitis | `abnormal` when the described pattern is that disease's radiographic signature (bronchial pattern → bronchitis; alveolar pattern → pneumonia). **Not** abnormal when the disease appears only in an unsupported differential list. |
| `interstitial` vs `Alveolar_interstitial_pattern` (feline) | `interstitial` = an interstitial component is described. `Alveolar_interstitial_pattern` = an alveolar component is described (air bronchograms, border effacement, lobar consolidation), typically mixed alveolar-interstitial. |
| `diseased_lungs` | Derived, never judged directly. Abnormal when any of: `pulmonary_nodules`, `pneumonia`, `bronchitis`, `interstitial`, `perihilar_infiltrate`, `focal_caudodorsal_lung`, `focal_perihilar`, `bronchiectasis`, `pulmonary_hypoinflation` (plus `Alveolar_interstitial_pattern` on the feline sheet). Matches `classifier_multi/categories.py` so gold and model derivations are identical. |
| `vhs` | `abnormal` when the report states the vertebral heart score is enlarged or above the reference range. Unstated → `normal`. |

Per-column radiographic decision rules, with citations to SignalPET / PMC / VIN / AVMA
sources, live in `dataset_gold_standard/criteria.md`.

## Write safety

The CSVs are edited in place, so the write path is the highest-risk part of this work.

- Originals are copied to `dataset_gold_standard/_originals/` before anything is written.
- Reading and writing both use `csv` at cp1252 with `newline=''` and an explicit CRLF
  terminator. `pandas` is not used for the write path: a round-trip through it silently
  renormalises quoting across ~3,700 untouched rows.
- Only label cells are mutated. Every other cell is passed through unchanged.
- After each write the file is re-parsed and **every non-label cell is compared against
  the backup**. Cell equality, not byte equality, is the invariant — quoting style may
  differ where a field was quoted unnecessarily in the original, but no value may change.

## Checkpointing

`dataset_gold_standard/evidence_<sheet>.json` is the source of truth, not the CSV:

```json
{ "<CaseID>": { "abnormal": { "<finding>": { "evidence": "verbatim quote", "reasoning": "one sentence" } } } }
```

Only abnormal findings are recorded. Everything not listed is `normal` by the
not-mentioned rule, and `diseased_lungs` is computed from the roll-up. This keeps the
audit trail proportional to the signal and mirrors the evidence convention the models
are held to.

Written after every batch of ~25 cases. The CSVs are regenerated from the evidence store,
so an interrupted run resumes without loss and every positive label is traceable to the
sentence that produced it.

## Verification

Work is not complete until all of these pass:

1. Every selected row has all label cells filled with `normal` or `abnormal`; no blanks,
   no other values.
2. Coverage: at least 3 abnormal per column, or a documented shortfall.
3. `diseased_lungs` agrees with its roll-up in every selected row, checked by script.
4. All non-label cells match the backup.
5. A blind re-label of a random 5% of cases, compared against the first pass to measure
   self-agreement.

## Deliverables

- The three CSVs, labelled in place
- `dataset_gold_standard/criteria.md` — decision rule and source per column
- `dataset_gold_standard/evidence_<sheet>.json` — audit trail
- `dataset_gold_standard/selection_log.csv` — case list, selection reason, RNG seed
- `dataset_gold_standard/coverage_report.md` — abnormal counts per column
- `plan.md` — implementation plan

## Out of scope

The brief says "stick to classification only". No model runs, no evaluation, no changes
to `classifier_multi/`. One note for later: that package still expects gold filenames of
the form `*_manual_final_50_appended(in).csv`, which no longer match these files.
