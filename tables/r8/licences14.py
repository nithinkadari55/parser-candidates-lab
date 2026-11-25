"""Milestone 14: which round-8 sets may train something the harness ships.

Policy: a set trains a shipped model only when the licence its creator states for the dataset (dataset card, paper or
repository LICENSE) allows commercial use of models derived from it (CDLA-Permissive-1.0/2.0, CC BY 4.0, CC0, MIT,
Apache-2.0, BSD). Non-commercial, evaluation-only, unlicensed or unknown sets are evaluated on, never trained on, never
used to choose a threshold, and none of their rows or ids go into the harness repository. The source documents keep
their own terms (PMC Open Access articles, arXiv papers, BRGM reports); nothing from them is shipped, only tree
thresholds over table-level ratios.

Checked 2026-11-26 from the primary sources (links in REPORT.md)."""

# set -> (dataset, licence, primary source)
TRAIN = {
    "r4pt": ("PubTables-1M (test tables; PMC source PDFs from the KDD 2026 bundle)", "CDLA-Permissive-2.0",
             "https://huggingface.co/datasets/bsmock/pubtables-1m"),
    "r4scitsr": ("SciTSR (test, incl. COMP)", "MIT", "https://github.com/Academic-Hammer/SciTSR"),
    "ptn": ("PubTabNet (val)", "CDLA-Permissive-1.0 (annotations; images under the PMC Open Access terms)",
            "https://github.com/ibm-aur-nlp/PubTabNet/blob/master/LICENSE.md"),
    "pt2": ("PubTables-v2 (cropped tables, test)", "CDLA-Permissive-2.0", "https://huggingface.co/datasets/kensho/PubTables-v2"),
    "ftnc": ("FinTabNet.c (test)", "CDLA-Permissive-2.0", "https://huggingface.co/datasets/bsmock/FinTabNet.c"),
    "synfin": ("SynFinTabs (test)", "MIT", "https://huggingface.co/datasets/ethanbradley/synfintabs"),
    "r4brgm": ("Table-BRGM (Soric et al., arXiv 2511.16134; created and published by the authors)", "MIT (dataset card)",
               "https://huggingface.co/datasets/marijanic/table-extraction-scientific-datasets"),
    "tabrec": ("TabRecSet (English part)", "CC BY 4.0 (attribution)", "https://doi.org/10.6084/m9.figshare.20647788.v9"),
}
EVAL_ONLY = {
    "r4icdar": ("ICDAR-2013 table competition (corrected, KDD 2026 bundle)", "no licence stated by the organisers; evaluation only",
                "https://huggingface.co/datasets/marijanic/table-extraction-scientific-datasets"),
    "ctdarm": ("ICDAR-2019 cTDaR modern", "no licence file in the repository; evaluation only",
               "https://github.com/cndplab-founder/ICDAR2019_cTDaR"),
    "ctdarh": ("ICDAR-2019 cTDaR historical", "no licence file in the repository; evaluation only",
               "https://github.com/cndplab-founder/ICDAR2019_cTDaR"),
    "wtw": ("WTW (Wired Table in the Wild), test", "CC BY-NC 4.0 (non-commercial)", "https://github.com/wangwen-whu/WTW-Dataset"),
}
# real text-layer PDFs (the harness reads their PDF words), and sets whose words are real PDF words on an image page
REAL_PDF = ["r4pt", "r4scitsr", "r4brgm", "r4icdar"]
REAL_WORDS = REAL_PDF + ["pt2", "ftnc"]


def source_set(row_id, s):
    """the set a table comes from; rotated copies (rotpdf_<set>_...) belong to their source set"""
    return row_id.split("_")[1] if s == "rotpdf" else s


def trainable(row_id, s):
    return source_set(row_id, s) in TRAIN


def markdown_table():
    out = ["| set | dataset | licence | use | primary source |", "|---|---|---|---|---|"]
    for use, d in (("training and evaluation", TRAIN), ("evaluation only", EVAL_ONLY)):
        for s, (name, lic, url) in d.items():
            out.append(f"| {s} | {name} | {lic} | {use} | {url} |")
    return "\n".join(out)
