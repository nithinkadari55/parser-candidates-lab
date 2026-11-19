import os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(SRC)
WORK = os.path.abspath(os.environ.get("TB_WORK", os.path.join(TABLES, "work")))
RAW = os.path.join(WORK, "data", "raw")
SETS = os.path.join(WORK, "data", "sets")
MODELS = os.path.join(WORK, "models")
TATR_SRC = os.path.join(WORK, "third", "tatr_src")
OUT = os.path.join(WORK, "out")
THIRD = os.path.join(TABLES, "third_party")
for p in (os.path.join(SRC, "shims"), THIRD, TATR_SRC, SRC):
    if p not in sys.path: sys.path.insert(0, p)
