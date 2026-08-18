from gold_standard.screen import eligible_case_ids, evaluates_region

THORACIC = (
    "Pulmonary parenchyma: A mild diffuse bronchial pattern is present.\n"
    "Cardiac silhouette: The cardiac silhouette is normal in size and shape."
)
ABDOMINAL = (
    "Liver: The liver is subjectively normal in size.\n"
    "Spleen: The spleen is enlarged with an undulant dorsal margin."
)
SPINAL = (
    "Bones/Joints:\nL3-4 spondylosis deformans is present.\n"
    "Soft tissues: The included paraspinal soft tissues are normal."
)


def test_thoracic_report_evaluates_the_thorax():
    assert evaluates_region(THORACIC, "thorax") is True
    assert evaluates_region(THORACIC, "abdomen") is False


def test_abdominal_report_evaluates_the_abdomen():
    assert evaluates_region(ABDOMINAL, "abdomen") is True
    assert evaluates_region(ABDOMINAL, "thorax") is False


def test_spinal_study_evaluates_neither_region():
    assert evaluates_region(SPINAL, "thorax") is False
    assert evaluates_region(SPINAL, "abdomen") is False


def test_combined_study_evaluates_both_regions():
    combined = THORACIC + "\n" + ABDOMINAL
    assert evaluates_region(combined, "thorax") is True
    assert evaluates_region(combined, "abdomen") is True


def test_incidental_mention_does_not_count_as_evaluation():
    """'The included thorax is normal' in an abdominal study is not a thoracic read."""
    assert evaluates_region(ABDOMINAL + " The included thorax is normal.", "thorax") is False


def test_eligible_pools_are_large_enough_for_300_each():
    from gold_standard.sheets import SHEETS

    for sheet in SHEETS.values():
        pool = eligible_case_ids(sheet)
        assert len(pool) >= sheet.target_cases, f"{sheet.name}: only {len(pool)} eligible"
        assert len(set(pool)) == len(pool), f"{sheet.name}: duplicate CaseIDs in pool"
