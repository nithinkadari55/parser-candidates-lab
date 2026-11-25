"""heron fp32 vs int8 on DocLayNet test per document category (financial reports, manuals, scientific
articles, laws and regulations, patents, government tenders), from the saved detections of order run
44477997398. usage: r7_dln.py ART_DIR OUT.md"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r7_eval import ap_stats, boot, page_f1, wil  # noqa: E402

DLN = ["Table", "Picture", "Text", "Title", "List-item", "Formula", "Caption", "Footnote", "Page-header",
       "Page-footer", "Section-header"]
MAP = {"Document Index": "Table", "Code": "Text"}


def load(pattern):
    out = {}
    for p in glob.glob(pattern, recursive=True):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def preds(r):
    out = collections.defaultdict(list)
    for d in sorted(r["dets"], key=lambda d: -d[1])[:100]:
        lab = d[0].replace(" - ", "-")
        out[MAP.get(lab, lab)].append((d[1], d[2:6]))
    return out


def main():
    art, outp = sys.argv[1], sys.argv[2]
    f32 = load(os.path.join(art, "**", "layout-heron-*.jsonl"))
    i8 = load(os.path.join(art, "**", "layout-heron_int8-*.jsonl"))
    gtr = load(os.path.join(art, "**", "dln-gt-*.jsonl"))
    by = collections.defaultdict(list)
    for pid, r in gtr.items():
        if pid in f32 and pid in i8:
            by[r.get("doc_category") or "unknown"].append(pid)
    L = ["## DocLayNet test by document category: heron fp32 vs int8\n",
         "| category | pages | fp32 mAP | int8 mAP | page F1 fp32 | page F1 int8 | int8 - fp32 [95% CI] | Wilcoxon p | "
         "int8 better / worse pages |", "|---|---|---|---|---|---|---|---|---|"]
    for cat in sorted(by) + ["all"]:
        ids = [i for v in by.values() for i in v] if cat == "all" else by[cat]

        def gts(pid):
            g = collections.defaultdict(list)
            for x in gtr[pid]["gt"]:
                g[x[0]].append(x[1:5])
            return g

        pa = [(gts(i), preds(f32[i])) for i in ids]
        pb = [(gts(i), preds(i8[i])) for i in ids]
        ma = [s["map"] for s in (ap_stats(pa, c) for c in DLN) if s]
        mb = [s["map"] for s in (ap_stats(pb, c) for c in DLN) if s]
        fa = [page_f1(g, p, DLN) for g, p in pa]
        fb = [page_f1(g, p, DLN) for g, p in pb]
        d = [y - x for x, y in zip(fa, fb)]
        m, lo, hi = boot(d)
        L.append(f"| {cat} | {len(ids)} | {np.mean(ma):.3f} | {np.mean(mb):.3f} | {np.mean(fa):.4f} | {np.mean(fb):.4f} | "
                 f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {wil(fb, fa):.2g} | {sum(x > 1e-9 for x in d)} / "
                 f"{sum(x < -1e-9 for x in d)} |")
    text = "\n".join(L) + "\n"
    open(outp, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
