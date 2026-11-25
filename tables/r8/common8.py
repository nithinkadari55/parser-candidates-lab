"""Round 8 shared definitions: the sets, their document types, document keys, the dev/test split and file helpers.

Every set lives in $TB_WORK/data/sets/<set>/: index.json (table ids), <id>.json (answer key) and the page PDFs the
tables sit on. Answer-key json (the round-4 format): id, doc, pdf, page_index, crop_pt ([l, t, r, b] page points,
top-left origin of the displayed page: the answer-key table box plus a 4 pt margin), html (PubTabNet structure tokens),
cells ([{text, bbox}], the answer key's cells in html order), tags (prep_sets.tag), kind (pdf | synth | scan) and,
for rotated pages, rot (dict(theta, mode))."""
import os, sys, json, zlib, re

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
for p in (os.path.join(TABLES, "r4"), os.path.join(TABLES, "r5"), HERE):
    if p not in sys.path: sys.path.insert(0, p)
from r4_common import *  # noqa: E402,F401  (paths, cluster_boot, wilcoxon_p, iou, jl, html_to_tokens)
import paths  # noqa: E402

R8OUT = os.path.join(OUT, "r8")

# name, document type, kind, label
SET_INFO = {
    "r4pt": ("scientific", "pdf", "PubTables-1M test, PMC source PDFs (KDD bundle)"),
    "r4scitsr": ("scientific", "pdf", "SciTSR test, arXiv source PDFs (incl. COMP)"),
    "ptn": ("scientific", "synth", "PubTabNet val (image + answer-key cell text as the text layer)"),
    "pt2": ("scientific", "synth", "PubTables-v2 cropped test (image + PDF words as the text layer)"),
    "ftnc": ("financial", "synth", "FinTabNet.c test (image + PDF words as the text layer)"),
    "synfin": ("financial", "synth", "SynFinTabs test (image + generator words as the text layer)"),
    "r4icdar": ("government/business", "pdf", "ICDAR-2013 corrected, EU/US government PDFs (KDD bundle)"),
    "r4brgm": ("government/business", "pdf", "Table-BRGM, FR/EN reports (KDD bundle)"),
    "ctdarm": ("government/business", "scan", "ICDAR-2019 cTDaR modern (images, OCR words)"),
    "ctdarh": ("scans/photos", "scan", "ICDAR-2019 cTDaR historical (images, OCR words)"),
    "tabrec": ("scans/photos", "scan", "TabRecSet English (scans and photos, incl. curved)"),
    "wtw": ("scans/photos", "scan", "WTW test (photos, scans, web; ModelScope repack)"),
    "rotpdf": ("rotated", "pdf", "Rotated text-PDF pages (90/180/270; /Rotate and content)"),
}
R4_SETS = ["r4pt", "r4scitsr", "r4icdar", "r4brgm"]
NEW_SETS = ["ptn", "pt2", "ftnc", "synfin", "ctdarm", "ctdarh", "tabrec", "wtw", "rotpdf"]
ALL_SETS = list(SET_INFO)
TYPES = ["scientific", "financial", "government/business", "scans/photos", "rotated"]
SCAN_SETS = [s for s, v in SET_INFO.items() if v[1] == "scan"]
DEV_SHARE = 3  # of 10 documents go to the dev split


def set_type(s): return SET_INFO[s][0]


def set_kind(s): return SET_INFO[s][1]


def set_ids(s):
    p = os.path.join(SETS, s, "index.json")
    if not os.path.exists(p): return []
    ids = json.load(open(p)); lim = int(os.environ.get("TB_LIMIT") or 0)
    return ids[:lim] if lim else ids


def load_gt(s, tid): return json.load(open(os.path.join(SETS, s, tid + ".json"), encoding="utf-8"))


def mine(key, shard, n): return zlib.crc32(key.encode()) % n == shard


def doc_of(s, gt):
    d = gt.get("doc") or gt.get("filename") or gt["id"]
    return s.replace("rotpdf", "rot") + ":" + str(d)


def split_of(doc):
    """dev or test, fixed by the document (the rotated copy of a page shares its source document's split)"""
    base = doc.split(":", 1)[1] if ":" in doc else doc
    return "dev" if zlib.crc32(("r8split|" + base).encode()) % 10 < DEV_SHARE else "test"


def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def strip_text(cells):
    return [{k: v for k, v in c.items() if k != "text"} for c in (cells or [])]
