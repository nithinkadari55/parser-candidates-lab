"""Round 9: rebuild a shard's candidate bundle (crop PNGs + answer keys) from the set PDFs, with the same r5words
render base9.py used, and check every crop against base9's image hash (identical input for every arm).
usage: mkbundle9.py SHARD BUNDLE_DIR WORDS_BIN   (BUNDLE_DIR holds base9's rows.jsonl.gz)"""
import os, sys, json, gzip, hashlib, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common9 import *
from PIL import Image


def main():
    shard, bdir, words_bin = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    rows = []
    with gzip.open(os.path.join(bdir, "rows.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    jobs = os.path.join(bdir, "jobs.jsonl"); wdir = os.path.join(WORK, "r9cwords", f"s{shard}")
    with open(jobs, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(dict(set=r["set"], id=r["id"], pdf=os.path.join(SETS, r["set"], r["pdf"]), page=0, crop_pt=r["crop_pt"])) + "\n")
    subprocess.run([words_bin, jobs, wdir], check=True)
    os.makedirs(os.path.join(bdir, "img"), exist_ok=True); os.makedirs(os.path.join(bdir, "gt"), exist_ok=True)
    bad = 0
    for r in rows:
        png = os.path.join(wdir, r["set"], r["id"] + ".png")
        im = Image.open(png).convert("RGB")
        if hashlib.sha256(im.tobytes()).hexdigest() != r["img_sha"]: bad += 1
        shutil.move(png, os.path.join(bdir, "img", r["id"] + ".png"))
        shutil.copy(os.path.join(SETS, r["set"], r["id"] + ".json"), os.path.join(bdir, "gt", r["id"] + ".json"))
    print("bundle", shard, "tables", len(rows), "crop hash mismatches", bad, flush=True)
    if bad: sys.exit(f"{bad} crops differ from base9's")


if __name__ == "__main__":
    main()
