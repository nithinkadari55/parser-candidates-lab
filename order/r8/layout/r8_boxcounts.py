"""Round 8, test B: direction of the fp32 / int8 differences on pages without ground truth. Per source and
coarse class: boxes at score >= 0.5 from each model, and boxes found by only one model (no same-class match
at IoU >= 0.5 in the other model's output).
usage: r8_boxcounts.py PAGES.jsonl DETS_DIR SET OUT.md"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "layout", "r7"))
from r7_eval import CLASSES, iou_mat, load, preds  # noqa: E402


def only(a, b):
    """boxes of A with no same-class box of B at IoU >= 0.5 (greedy one-to-one)"""
    if not a:
        return 0
    if not b:
        return len(a)
    M = iou_mat(a, b)
    used = set()
    n = 0
    for k in range(len(a)):
        order = sorted(range(len(b)), key=lambda j: -M[k, j])
        j = next((j for j in order if j not in used and M[k, j] >= 0.5), None)
        if j is None:
            n += 1
        else:
            used.add(j)
    return n


def main():
    pages_f, dets, ds, outp = sys.argv[1:5]
    pages = [json.loads(x) for x in open(pages_f, encoding="utf-8")]
    f32 = load(os.path.join(dets, f"{ds}-heron-*.jsonl"))
    i8 = load(os.path.join(dets, f"{ds}-heron_int8-*.jsonl"))
    T = collections.defaultdict(lambda: [0, 0, 0, 0])
    for p in pages:
        if p["id"] not in f32 or p["id"] not in i8:
            continue
        pa, pb = preds(f32[p["id"]], False), preds(i8[p["id"]], False)
        for c in CLASSES:
            a = [x for s, x in pa.get(c, []) if s >= 0.5]
            b = [x for s, x in pb.get(c, []) if s >= 0.5]
            for key in (p["src"], "all"):
                t = T[(key, c)]
                t[0] += len(a)
                t[1] += len(b)
                t[2] += only(a, b)
                t[3] += only(b, a)
    srcs = sorted({k[0] for k in T} - {"all"}) + ["all"]
    L = [f"### Box counts on pages without ground truth ({ds})\n",
         "Boxes at score >= 0.5; 'fp32 only' / 'int8 only' = boxes with no same-class box of the other model at IoU >= 0.5.\n",
         "| source | class | fp32 boxes | int8 boxes | fp32 only | int8 only |", "|---|---|---|---|---|---|"]
    for s in srcs:
        for c in CLASSES:
            t = T.get((s, c))
            if t and (t[0] or t[1]):
                L.append(f"| {s} | {c} | {t[0]} | {t[1]} | {t[2]} | {t[3]} |")
    open(outp, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
