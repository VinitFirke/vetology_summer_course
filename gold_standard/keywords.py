"""Search terms that find likely positives for each label column.

These are a *sampling aid*, not a classifier. The brief requires every disease to be
scored abnormal at least three times, and the rare ones - bronchiectasis, hypoplastic
trachea, microhepatia - appear in well under one percent of reports, so a random draw
of 300 would miss them entirely. These patterns pull candidates out of the pool; a
reader then decides each one against criteria.md, and plenty of hits are rejected.

The terms are the words radiologists actually write. A report never contains the
string "bronchitis abnormal"; it says "bronchial pattern". They also match negations -
"no pleural effusion" hits the pleural_effusion terms - which is fine and expected,
because finding a case worth reading is the only job here.

Keeping this file next to criteria.md means the vocabulary used to find cases and the
vocabulary used to judge them are revised together.
"""

import re

SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    # --- thoracic: lung parenchyma and airways ---
    "bronchitis": (
        "bronchial pattern",
        "bronchointerstitial",
        "broncho-interstitial",
        "peribronchial",
        "bronchial marking",
        "bronchial wall",
        "bronchitis",
    ),
    "interstitial": (
        "interstitial",
    ),
    "Alveolar_interstitial_pattern": (
        "alveolar",
        "air bronchogram",
        "consolidat",
        "border effacement",
        "lobar sign",
    ),
    "pneumonia": (
        "pneumonia",
        "alveolar pattern",
        "air bronchogram",
        "consolidat",
        "lobar sign",
        "aspiration",
    ),
    "pulmonary_nodules": (
        "nodule",
        "pulmonary mass",
        "miliary",
        "metastatic",
        "osteoma",
    ),
    "perihilar_infiltrate": (
        "perihilar",
        "hilar infiltrate",
    ),
    "focal_perihilar": (
        "focal perihilar",
        "perihilar nodule",
        "perihilar mass",
        "perihilar opacity",
        "perihilar soft tissue",
        "perihilar focus",
    ),
    "focal_caudodorsal_lung": (
        "caudodorsal",
        "caudal dorsal lung",
    ),
    "bronchiectasis": (
        "bronchiectasis",
        "bronchiectatic",
    ),
    "pulmonary_hypoinflation": (
        "hypoinflat",
        "hypo-inflat",
        "underinflat",
        "under-inflat",
        "incompletely inflated",
        "expiratory",
        "expiration",
        "reduced lung volume",
        "decreased lung volume",
    ),
    # --- thoracic: trachea, heart, pleura, mediastinum ---
    "hypo_plastic_trachea": (
        "hypoplastic trachea",
        "tracheal hypoplasia",
        "hypoplasia of the trachea",
        "tracheal diameter",
        "narrowed trachea",
        "small trachea",
    ),
    "cardiomegaly": (
        "cardiomegaly",
        "cardiac enlargement",
        "enlarged cardiac silhouette",
        "cardiac silhouette is enlarged",
        "heart is enlarged",
        "globoid",
    ),
    "left_sided_cardiomegaly": (
        "left atrial enlarge",
        "left atrium is enlarged",
        "left ventricular enlarge",
        "left-sided cardiomegaly",
        "left sided cardiomegaly",
        "left auricular",
        "mainstem bronchi",
        "tracheal elevation",
        "elevation of the carina",
        "left atrial",
    ),
    "right_sided_cardiomegaly": (
        "right atrial enlarge",
        "right ventricular enlarge",
        "right-sided cardiomegaly",
        "right sided cardiomegaly",
        "reverse d",
        "sternal contact",
        "right atrial",
    ),
    "vhs": (
        "vertebral heart",
        "vhs",
    ),
    "pleural_effusion": (
        "pleural effusion",
        "pleural fluid",
        "fissure line",
        "lung lobe retraction",
        "costophrenic",
    ),
    "pericardial_effusion": (
        "pericardial effusion",
        "pericardial fluid",
        "pericardi",
        "globoid",
    ),
    "pulmonary_vessel_enlargement": (
        "pulmonary venous distension",
        "pulmonary venous distention",
        "enlarged pulmonary",
        "distended pulmonary",
        "dilated pulmonary",
        "prominent pulmonary",
        "pulmonary vascular enlargement",
        "vascular enlargement",
        "enlarged pulmonary vasculature",
    ),
    "thoracic_lymphadenopathy": (
        "lymphadenopathy",
        "lymphadenomegaly",
        "lymph node enlarge",
        "enlarged lymph node",
        "sternal lymph",
        "tracheobronchial lymph",
        "hilar lymph",
    ),
    "esophagitis": (
        "megaesophagus",
        "mega-esophagus",
        "esophagitis",
        "esophageal dilat",
        "dilated esophagus",
        "esophageal distension",
        "esophageal distention",
        "distended esophagus",
        "esophageal gas",
        "gas within the esophagus",
        "gas-filled esophagus",
        "retained ingesta",
    ),
    # --- abdominal ---
    "hepatomegaly": (
        "hepatomegaly",
        "liver is enlarged",
        "enlarged liver",
        "hepatic enlargement",
        "rounded liver",
        "gastric axis",
    ),
    "microhepatia": (
        "microhepatia",
        "microhepatica",
        "small liver",
        "liver is small",
        "liver is subjectively small",
    ),
    "liver_mass": (
        "liver mass",
        "hepatic mass",
        "liver nodule",
        "hepatic nodule",
        "cranial abdominal mass",
    ),
    "splenomegaly": (
        "splenomegaly",
        "spleen is enlarged",
        "enlarged spleen",
        "splenic enlargement",
        "undulant",
    ),
    "splenic_mass": (
        "splenic mass",
        "splenic nodule",
        "spleen mass",
        "mid-abdominal mass",
        "midabdominal mass",
    ),
    "ascites": (
        "peritoneal fluid",
        "peritoneal effusion",
        "ascites",
        "serosal detail",
        "abdominal effusion",
        "free fluid",
    ),
    "small_intestinal_obstruction": (
        "obstruction",
        "obstructive",
        "ileus",
        "plicat",
        "foreign body",
        "dilated small intestin",
        "segmental dilation",
        "two populations",
    ),
    "gastritis": (
        "gastritis",
        "gastric wall",
        "gastric mucosal",
        "stomach wall",
    ),
    "colitis": (
        "colitis",
        "colonic wall",
        "colon wall",
        "typhlitis",
    ),
    "pancreatitis": (
        "pancreatitis",
        "pancreat",
        "sentinel loop",
        "duodenal",
        "right cranial abdominal",
    ),
}


def candidate_pattern(column: str) -> re.Pattern[str]:
    """A case-insensitive alternation over one column's search terms."""
    terms = SEARCH_TERMS.get(column)
    if not terms:
        raise ValueError(f"No search terms defined for {column!r}")
    return re.compile("|".join(re.escape(t) for t in terms), re.I)
