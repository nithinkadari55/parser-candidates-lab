"""Milestone 14: keep only licence-allowed tables in the harness's TATR fixtures.

tatr_fixtures.json was written by tables/r5/r5_tatr.py (shard 0, the first saved tables of each round-4 set), which
included three ICDAR-2013 tables (evaluation-only, licences14.EVAL_ONLY). This drops every case whose set is not in
licences14.TRAIN and writes the file back in the same json.dump layout, so the committed fixture is r5_tatr.py's output
filtered by this script.

usage: fixclean14.py IN_JSON OUT_JSON"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from licences14 import TRAIN, source_set


def allowed(case):
    s = case.get("set") or case["id"].split("_")[0]
    return source_set(case["id"], s) in TRAIN


def main():
    src, dst = sys.argv[1:3]
    d = json.load(open(src, encoding="utf-8"))
    for key in ("post", "raw", "e2e"):
        before = len(d[key])
        d[key] = [c for c in d[key] if allowed(c)]
        print(key, before, "->", len(d[key]))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
