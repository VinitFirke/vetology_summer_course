"""The analysis pipeline, driven end to end on synthetic samples.

The paid run happens once. This exercises everything downstream of it beforehand, so the
first time uq_analyze_main sees real data is not the first time it runs at all.
"""

import random

import pandas as pd
import pytest

from classifier.schemas import FindingName
from uncertainty import config as uq_config
from uncertainty.sample import append_record

import uq_analyze_main
from uq_analyze_main import (
    UNDEFINED_AUC,
    collect_rows,
    load_gold,
    load_ce_scores,
    plot_calibration,
    summarise,
)

FINDINGS = [f.value for f in FindingName]


def test_load_gold_covers_every_enum_finding():
    """Reads the real gold standard: proves the column slice and the rename line up.

    GOLD_TO_PREDICTION maps Fe_Alveolar -> Alveolar_interstitial_pattern. If that broke,
    one finding would silently never be scored.
    """
    gold = load_gold()

    assert len(gold) == 50
    keys = set(next(iter(gold.values())))
    assert {f.value for f in FindingName} <= keys
    assert "diseased_lungs" in keys  # present but excluded downstream by build_rows


@pytest.fixture
def sampled(tmp_path, monkeypatch):
    """A synthetic run on disk: one provider, three tiers, 12 cases, 5 replicates."""
    monkeypatch.setattr(uq_config, "UQ_DIR", tmp_path)
    monkeypatch.setattr(uq_config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(
        uq_config, "samples_path", lambda p, t: tmp_path / f"samples_{p}_{t}.jsonl"
    )
    monkeypatch.setattr(uq_config, "ce_path", lambda p, t: tmp_path / f"ce_{p}_{t}.jsonl")

    rng = random.Random(0)
    gold: dict[str, dict[str, str]] = {}

    for index in range(12):
        case_id = f"case{index}"
        truth = {f: ("Abnormal" if rng.random() < 0.3 else "Normal") for f in FINDINGS}
        gold[case_id] = truth

        for tier in uq_config.TIERS:
            # Higher tiers agree with the truth more often, so effort has a visible effect.
            accuracy = {"low": 0.6, "medium": 0.75, "high": 0.9}[tier]
            for replicate in range(1, 6):
                labels = {
                    f: (truth[f].lower() if rng.random() < accuracy
                        else ("normal" if truth[f] == "Abnormal" else "abnormal"))
                    for f in FINDINGS
                }
                append_record(
                    tmp_path / f"samples_openai_{tier}.jsonl",
                    {
                        "provider": "openai", "tier": tier, "case_id": case_id,
                        "replicate": replicate, "labels": labels, "logprobs": None,
                        "usage": {"input_tokens": 1000, "output_tokens": 700},
                        "timestamp": "2026-08-08T00:00:00",
                    },
                )
            append_record(
                tmp_path / f"ce_openai_{tier}.jsonl",
                {
                    "provider": "openai", "tier": tier, "case_id": case_id,
                    "scores": {f: rng.randint(50, 100) for f in FINDINGS},
                    "usage": {}, "timestamp": "2026-08-08T00:00:00",
                },
            )

    return tmp_path, gold


def test_ce_scores_load_back_per_case(sampled):
    _, _ = sampled
    scores = load_ce_scores("openai", "low")

    assert len(scores) == 12
    assert set(scores["case0"]) == set(FINDINGS)


def test_collect_rows_covers_every_tier_and_both_proxies(sampled):
    _, gold = sampled

    rows = collect_rows("openai", gold)

    frame = pd.DataFrame([r.model_dump() for r in rows])
    assert set(frame["tier"]) == set(uq_config.TIERS)
    assert set(frame["proxy"]) == {"SC", "CE"}
    # 12 cases x 19 findings x 3 tiers x 2 proxies
    assert len(frame) == 12 * 19 * 3 * 2


def test_tlp_never_appears(sampled):
    """The probe found no provider returns logprobs; TLP must not silently reappear."""
    _, gold = sampled
    rows = collect_rows("openai", gold)

    assert "TLP" not in {r.proxy for r in rows}


def test_summarise_produces_one_row_per_tier_and_proxy(sampled):
    _, gold = sampled
    frame = pd.DataFrame([r.model_dump() for r in collect_rows("openai", gold)])

    summary = summarise(frame, bootstrap_iterations=50)

    assert len(summary) == 6  # 3 tiers x 2 proxies
    assert list(summary.columns) == uq_analyze_main.RESULT_COLUMNS
    assert summary["n"].tolist() == [12 * 19] * 6


def test_summary_statistics_are_in_range(sampled):
    _, gold = sampled
    frame = pd.DataFrame([r.model_dump() for r in collect_rows("openai", gold)])

    summary = summarise(frame, bootstrap_iterations=50)

    assert summary["AUC"].between(0, 1).all()
    assert summary["ECE"].between(0, 1).all()
    assert summary["Brier"].between(0, 1).all()
    assert (summary["CI low"] <= summary["AUC"]).all()
    assert (summary["AUC"] <= summary["CI high"]).all()


def test_sc_confidence_only_ever_takes_three_values(sampled):
    """At N=5 this is arithmetic, but it is the headline limitation - pin it."""
    _, gold = sampled
    rows = [r for r in collect_rows("openai", gold) if r.proxy == "SC"]

    assert {r.confidence for r in rows} <= {0.6, 0.8, 1.0}


def test_higher_effort_is_more_accurate_in_this_fixture(sampled):
    """Sanity check that the tier axis survives the whole pipeline intact."""
    _, gold = sampled
    frame = pd.DataFrame([r.model_dump() for r in collect_rows("openai", gold)])
    summary = summarise(frame, bootstrap_iterations=50).set_index(["tier", "proxy"])

    low = summary.loc[("low", "SC"), "observed accuracy"]
    high = summary.loc[("high", "SC"), "observed accuracy"]
    assert high > low


def test_a_group_with_no_incorrect_answers_reports_na(sampled):
    """A tier the model aced has no AUC; that must not crash the report."""
    _, gold = sampled
    frame = pd.DataFrame([r.model_dump() for r in collect_rows("openai", gold)])
    frame["correct"] = 1

    summary = summarise(frame, bootstrap_iterations=50)

    assert (summary["meets 0.7"] == UNDEFINED_AUC).all()
    assert summary["AUC"].isna().all()


def test_calibration_figures_are_written(sampled):
    tmp_path, gold = sampled
    frame = pd.DataFrame([r.model_dump() for r in collect_rows("openai", gold)])
    figures = tmp_path / "figures"

    for tier in uq_config.TIERS:
        plot_calibration(frame, "openai", tier, figures_dir=figures)

    written = sorted(p.name for p in figures.glob("*.png"))
    assert written == [
        "calibration_openai_high.png",
        "calibration_openai_low.png",
        "calibration_openai_medium.png",
    ]
    assert all((figures / name).stat().st_size > 5_000 for name in written)


def test_plotting_an_absent_tier_is_a_no_op(sampled):
    tmp_path, gold = sampled
    frame = pd.DataFrame([r.model_dump() for r in collect_rows("openai", gold)])
    figures = tmp_path / "empty"

    plot_calibration(frame[frame["tier"] == "nonexistent"], "openai", "nonexistent", figures)

    assert not figures.exists()
