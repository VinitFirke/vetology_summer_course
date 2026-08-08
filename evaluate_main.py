"""Score the LLM predictions against the manual gold standard.

Writes confusion_matrix_feline_thorax.xlsx with one sheet per model, each in the
column layout of the existing example workbook.

    python evaluate_main.py
    python evaluate_main.py --provider openai
"""

import argparse

import pandas as pd

from classifier import config
from classifier.csv_io import read_dataframe
from classifier.evaluate import MATRIX_COLUMNS, build_matrix_rows
from classifier.schemas import COL_CASE_ID


def labels_by_case(csv_path, condition_columns: list[str]) -> dict[str, dict[str, str]]:
    """Read a CSV into {case_id: {condition: label}}."""
    df = read_dataframe(csv_path)
    return {
        str(row[COL_CASE_ID]).strip(): {c: str(row[c]).strip() for c in condition_columns}
        for _, row in df.iterrows()
    }


def gold_condition_columns() -> list[str]:
    """The 20 label columns, named as the gold standard spells them."""
    gold = read_dataframe(config.GOLD_CSV)
    return list(gold.columns[10:30])


def build_sheet(provider: config.Provider, conditions: list[str]) -> pd.DataFrame | None:
    """One provider's confusion matrix, or None if its predictions are missing."""
    predictions_path = config.output_csv_path(provider)
    if not predictions_path.exists():
        print(f"  {provider}: no predictions at {predictions_path.name}, skipping")
        return None

    gold = labels_by_case(config.GOLD_CSV, conditions)
    prediction_columns = [config.GOLD_TO_PREDICTION.get(c, c) for c in conditions]
    predictions = labels_by_case(predictions_path, prediction_columns)

    rows = build_matrix_rows(gold, predictions, conditions, config.GOLD_TO_PREDICTION)
    scored = rows[0]["Check"] if rows else 0
    print(f"  {provider}: {len(rows)} conditions over {scored} scored cases")
    return pd.DataFrame(rows, columns=list(MATRIX_COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=[*config.PROVIDERS, "all"], default="all")
    args = parser.parse_args()

    providers = list(config.PROVIDERS) if args.provider == "all" else [args.provider]
    conditions = gold_condition_columns()

    print(f"Gold standard: {config.GOLD_CSV.name}")
    print(f"Conditions: {len(conditions)}")

    sheets = {p: build_sheet(p, conditions) for p in providers}
    sheets = {p: df for p, df in sheets.items() if df is not None}

    if not sheets:
        raise SystemExit("No prediction files found - run main.py first.")

    with pd.ExcelWriter(config.CONFUSION_MATRIX_XLSX, engine="openpyxl") as writer:
        for provider, df in sheets.items():
            df.to_excel(writer, sheet_name=provider, index=False)

    print(f"\nWrote {config.CONFUSION_MATRIX_XLSX.name} with sheets: {', '.join(sheets)}")


if __name__ == "__main__":
    main()
