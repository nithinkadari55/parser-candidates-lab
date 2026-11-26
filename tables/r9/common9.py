"""Round 9 shared definitions. Extends the round-8 registry (common8.SET_INFO, updated in place so tf8/tatr8/prep8
helpers see the new sets) with the round-9 sets, and gives every table a document type and a language.

Answer-key json is the round-8 format (common8.py); round-9 sets add optional keys: r9type (document type of this
table, overrides the set's), lang (ISO-ish language code), src_meta (small dict of dataset attributes, no text)."""
import os, sys, json, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
R8 = os.path.join(TABLES, "r8")
for p in (R8, HERE):
    if p not in sys.path: sys.path.insert(0, p)
import common8 as C8  # noqa: E402
from common8 import *  # noqa: E402,F401,F403

R9OUT = os.path.join(OUT, "r9")

NEW = {
    "pulse": ("multilingual filings", "scan", "PulseBench-Tab (9 languages; filings, government and regulatory reports)"),
    "rdtb": ("hard real-world", "scan", "RD-TableBench (scans, handwriting, dense and multilingual tables)"),
    "odb": ("scientific", "scan", "OmniDocBench v1.5 table blocks (full page images; type per page source)"),
    "tabreczh": ("scans/photos", "scan", "TabRecSet Chinese (scans and photos)"),
    "mustard": ("magazines/newspapers", "scan", "MUSTARD (13 languages, Indic scripts + Chinese; printed, scanned, scene)"),
    "iflytab": ("scans/photos", "scan", "iFLYTAB val (camera photos and scans, wired and wireless)"),
    "cisol": ("industrial forms", "scan", "CISOL validation (German steel order lists)"),
}
C8.SET_INFO.update(NEW)
SET_INFO = C8.SET_INFO
R8_SETS = list(C8.ALL_SETS)
NEW_SETS9 = list(NEW)
ALL_SETS9 = R8_SETS + NEW_SETS9

LANG_OF_SET = {"tabreczh": "zh", "cisol": "de", "iflytab": "zh/en"}
ODB_TYPE = {
    "academic_literature": "scientific", "book": "education", "colorful_textbook": "education", "exam_paper": "education",
    "magazine": "magazines/newspapers", "newspaper": "magazines/newspapers", "PPT2PDF": "slides",
    "research_report": "financial/research reports", "note": "notes/handwriting",
}
TYPES9 = ["scientific", "financial", "financial/research reports", "government/business", "historical", "scans/photos",
          "rotated", "multilingual filings", "hard real-world", "magazines/newspapers", "education", "slides",
          "notes/handwriting", "industrial forms"]
# sets whose tables trained the milestone-14 selector (licences14.TRAIN members actually used by variant w0)
SEL_TRAIN = ["r4pt", "r4scitsr", "pt2", "ftnc", "r4brgm", "tabrec"]


def table_type(s, gt):
    return gt.get("r9type") or C8.SET_INFO[s][0]


def table_lang(s, gt):
    return gt.get("lang") or LANG_OF_SET.get(s) or ("fr/en" if s == "r4brgm" else "en")


def set_kind9(s): return C8.SET_INFO[s][1]


def core(tid, share):
    """fixed subset for slow models: share in [0, 1] of the table ids, chosen by a hash of the id"""
    return zlib.crc32(("r9core|" + tid).encode()) % 10000 < int(round(share * 10000))
