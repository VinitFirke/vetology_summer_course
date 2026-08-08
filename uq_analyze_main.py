"""Turn collected samples into the results workbook and figures.

Free and offline - reads JSONL, calls no API, and can be re-run as often as you like.
That separation is the point: the sampling run is paid for once, and every bug fixed in
here costs nothing to re-check.

    python uq_analyze_main.py
    python uq_analyze_main.py --provider kimi

Proxies reported are SC and CE. TLP was dropped after the capability probe found that no
provider returns token logprobs - see the D5 result in catalog.md.
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # no display needed; write PNGs straight to disk

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from classifier import config as classifier_config  # noqa: E402
from classifier.csv_io import read_dataframe  # noqa: E402
from classifier.schemas import COL_CASE_ID, FindingName  # noqa: E402
from uncertainty import config as uq_config  # noqa: E402
from uncertainty.proxies import ProxyRow, build_rows  # noqa: E402
from uncertainty.sample import read_samples  # noqa: E402
from uncertainty.stats import (  # noqa: E402
    USEFUL_AUC,
    brier_score,
    calibration_points,
    clustered_bootstrap_ci,
    expected_calibration_error,
    roc_auc,
)

RESULT_COLUMNS = [
    "tier",
    "proxy",
    "n",
    "AUC",
    "CI low",
    "CI high",
    "ECE",
    "Brier",
    "mean confidence",
    "observed accuracy",
    "meets 0.7",
]

UNDEFINED_AUC = "N/A (no incorrect answers)"


def load_gold(gold_csv=None) -> dict[str, dict[str, str]]:
    """{case_id: {prediction_column_name: gold_label}}.

    Columns 10-29 are the label block. GOLD_TO_PREDICTION renames the one column the
    prediction files spell differently, so the keys line up with FindingName.
    """
    frame = read_dataframe(gold_csv or classifier_config.GOLD_CSV)
    gold_columns = list(frame.columns[10:30])
    rename = classifier_config.GOLD_TO_PREDICTION

    return {
        str(row[COL_CASE_ID]).strip(): {
            rename.get(column, column): str(row[column]).strip() for column in gold_columns
        }
        for _, row in frame.iterrows()
    }


def load_ce_scores(provider: str, tier: str) -> dict[str, dict[str, int]]:
    """{case_id: {finding: 0-100}} from one tier's CE file."""
    records = read_samples(uq_config.ce_path(provider, tier))
    return {case_id: entries[0].get("scores", {}) for case_id, entries in records.items()}


def collect_rows(provider: str, gold: dict[str, dict[str, str]]) -> list[ProxyRow]:
    """Every proxy row for one provider, across all tiers that have samples."""
    rows: list[ProxyRow] = []
    for tier in uq_config.TIERS:
        samples_file = uq_config.samples_path(provider, tier)
        if not samples_file.exists():
            print(f"  {provider}/{tier}: no samples, skipping")
            continue

        raw = read_samples(samples_file)
        samples = {cid: [r["labels"] for r in records] for cid, records in raw.items()}
        tier_rows = build_rows(samples, load_ce_scores(provider, tier), gold, provider, tier)
        print(f"  {provider}/{tier}: {len(raw)} cases -> {len(tier_rows)} rows")
        rows.extend(tier_rows)
    return rows


def summarise(frame: pd.DataFrame, bootstrap_iterations: int = 1000) -> pd.DataFrame:
    """One row per (tier, proxy) with every statistic.

    A group with no incorrect answers has no ROC AUC to report; that is recorded as N/A
    rather than allowed to raise, because a tier the model aced is a real outcome.

    The bootstrap dominates the runtime - about 3s per group at 1000 iterations on 950
    rows, so roughly a minute for three providers. Lower it while iterating on the report
    and put it back for the numbers you publish.
    """
    results = []
    for (tier, proxy), group in frame.groupby(["tier", "proxy"], sort=False):
        confidence = group["confidence"].tolist()
        correct = group["correct"].tolist()
        case_ids = group["case_id"].tolist()

        auc = roc_auc(confidence, correct)
        if auc is None:
            low = high = float("nan")
            verdict = UNDEFINED_AUC
        else:
            low, high = clustered_bootstrap_ci(
                confidence, correct, case_ids, iterations=bootstrap_iterations
            )
            verdict = "yes" if auc >= USEFUL_AUC else "no"

        results.append(
            {
                "tier": tier,
                "proxy": proxy,
                "n": len(group),
                "AUC": auc,
                "CI low": low,
                "CI high": high,
                "ECE": expected_calibration_error(confidence, correct),
                "Brier": brier_score(confidence, correct),
                "mean confidence": sum(confidence) / len(confidence),
                "observed accuracy": sum(correct) / len(correct),
                "meets 0.7": verdict,
            }
        )

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def plot_calibration(frame: pd.DataFrame, provider: str, tier: str, figures_dir=None) -> None:
    """Confidence against observed accuracy, proxies overlaid, with the diagonal.

    The paper's Figure 5 layout. Points below the diagonal are over-confidence.
    """
    subset = frame[frame["tier"] == tier]
    if subset.empty:
        return

    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")

    for proxy, group in subset.groupby("proxy"):
        points = calibration_points(group["confidence"].tolist(), group["correct"].tolist())
        if not points:
            continue
        axes.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=f"{proxy} (n={len(group)})",
        )

    axes.set_xlabel("stated confidence")
    axes.set_ylabel("observed accuracy")
    axes.set_title(f"{provider} - {tier} effort")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.legend(loc="upper left", fontsize="small")
    figure.tight_layout()

    target = figures_dir or uq_config.FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)
    figure.savefig(target / f"calibration_{provider}_{tier}.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=[*uq_config.UQ_PROVIDERS, "all"], default="all")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="bootstrap iterations for the AUC interval (default 1000, ~1 min for 3 "
        "providers; drop to 100 while iterating)",
    )
    args = parser.parse_args()

    providers = list(uq_config.UQ_PROVIDERS) if args.provider == "all" else [args.provider]
    gold = load_gold()
    print(f"Gold standard: {classifier_config.GOLD_CSV.name}, {len(gold)} cases")
    print(f"Findings analysed: {len(FindingName)} (diseased_lungs is derived, excluded)\n")

    all_rows: list[pd.DataFrame] = []
    sheets: dict[str, pd.DataFrame] = {}

    for provider in providers:
        rows = collect_rows(provider, gold)
        if not rows:
            continue

        frame = pd.DataFrame([row.model_dump() for row in rows])
        all_rows.append(frame)

        summary = summarise(frame, bootstrap_iterations=args.bootstrap)
        sheets[provider] = summary
        print(f"\n{provider}: {len(frame)} rows")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")

        for tier in uq_config.TIERS:
            plot_calibration(frame, provider, tier)

    if not sheets:
        raise SystemExit(
            "No samples found. Run:  python uq_main.py --provider all --yes"
        )

    uq_config.UQ_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows).to_csv(uq_config.PROXY_CSV, index=False)
    with pd.ExcelWriter(uq_config.RESULTS_XLSX, engine="openpyxl") as writer:
        for provider, summary in sheets.items():
            summary.to_excel(writer, sheet_name=provider, index=False)

    print(f"Wrote {uq_config.PROXY_CSV.name}")
    print(f"Wrote {uq_config.RESULTS_XLSX.name} with sheets: {', '.join(sheets)}")
    print(f"Wrote figures to {uq_config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
