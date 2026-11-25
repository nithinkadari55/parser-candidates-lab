"""Run heron fp32 or heron int8 (docling.rs model files) over a shard of a round-7 page set.
usage: r7_run.py MODEL PAGES.jsonl SHARD NSHARDS OUT.jsonl"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import models  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    name, pages, k, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    m = models.load(name)
    rows = [json.loads(line) for line in open(pages, encoding="utf-8")]
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if i % n != k:
                continue
            img = Image.open(r["img"]).convert("RGB")
            t = time.perf_counter()
            dets = m(img)
            sec = time.perf_counter() - t
            f.write(json.dumps(dict(id=r["id"], src=r["src"], W=r["W"], H=r["H"], sec=round(sec, 4),
                                    dets=[[d[0], round(d[1], 4)] + [round(v, 1) for v in d[2:6]] for d in dets])) + "\n")


if __name__ == "__main__":
    main()
