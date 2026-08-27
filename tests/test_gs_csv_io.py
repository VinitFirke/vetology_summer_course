import pytest

from gold_standard.csv_io import (
    apply_labels,
    read_sheet,
    verify_untouched,
    write_sheet,
)
from gold_standard.sheets import SHEETS, get_sheet, gold_csv_path


def test_every_sheet_round_trips_byte_identically(tmp_path):
    """The write-safety invariant: reading and rewriting must change nothing."""
    for sheet in SHEETS.values():
        path = gold_csv_path(sheet)
        original = path.read_bytes()
        header, rows = read_sheet(path)
        out = tmp_path / sheet.csv_name
        write_sheet(out, header, rows)
        assert out.read_bytes() == original, f"{sheet.name} did not round-trip"


def test_header_carries_every_label_column():
    for sheet in SHEETS.values():
        header, _ = read_sheet(gold_csv_path(sheet))
        missing = [c for c in sheet.label_columns if c not in header]
        assert missing == [], f"{sheet.name} missing {missing}"


def test_judged_columns_exclude_derived():
    sheet = get_sheet("canine_thorax")
    assert "diseased_lungs" in sheet.label_columns
    assert "diseased_lungs" not in sheet.judged_columns


def test_apply_labels_writes_only_the_named_cells(tmp_path):
    sheet = get_sheet("canine_abdomen")
    src = gold_csv_path(sheet)
    work = tmp_path / sheet.csv_name
    work.write_bytes(src.read_bytes())
    header, rows = read_sheet(work)
    case_id = rows[0][0]

    written = apply_labels(work, sheet, {case_id: {c: "normal" for c in sheet.label_columns}})

    assert written == 1
    _, after = read_sheet(work)
    assert after[0][header.index("ascites")] == "normal"
    assert after[1] == rows[1]  # untouched row unchanged
    assert verify_untouched(work, src, sheet) == []


def test_apply_labels_rejects_a_value_outside_the_vocabulary(tmp_path):
    sheet = get_sheet("canine_abdomen")
    work = tmp_path / sheet.csv_name
    work.write_bytes(gold_csv_path(sheet).read_bytes())
    _, rows = read_sheet(work)
    with pytest.raises(ValueError, match="not a valid label"):
        apply_labels(work, sheet, {rows[0][0]: {"ascites": "enlarged"}})


def test_verify_untouched_reports_a_tampered_report_cell(tmp_path):
    sheet = get_sheet("canine_abdomen")
    src = gold_csv_path(sheet)
    work = tmp_path / sheet.csv_name
    header, rows = read_sheet(src)
    rows[0][header.index("Findings (original radiologist report)")] = "tampered"
    write_sheet(work, header, rows)

    problems = verify_untouched(work, src, sheet)

    assert len(problems) == 1
    assert "Findings" in problems[0]
