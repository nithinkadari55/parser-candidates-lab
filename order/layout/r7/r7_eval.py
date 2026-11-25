"""Round 7: heron fp32 vs heron int8 on more layout data.
Sets with ground truth (D4LA test, M6Doc test, PubLayNet validation): COCO-style mAP per coarse class
and class-agnostic, paired per-page F1 with a bootstrap CI and Wilcoxon p.
Real PDF pages (no ground truth): agreement between the two models per page.
usage: r7_eval.py DATA_DIR DETS_DIR OUT.md"""
import collections
import glob
import json
import os
import random
import sys

import numpy as np
from scipy.stats import wilcoxon

THR = [0.5 + 0.05 * i for i in range(10)]
COARSE = {"Table": "Table", "Document Index": "Table", "Picture": "Picture", "Caption": "Caption",
          "Formula": "Formula", "Page-header": "Page-header", "Page-footer": "Page-footer"}
CLASSES = ["Table", "Picture", "Text", "Caption", "Formula", "Page-header", "Page-footer"]


def coarse(label, agnostic):
    if agnostic:
        return "Any"
    return COARSE.get(label.replace(" - ", "-"), "Text")


def preds(r, agnostic):
    out = collections.defaultdict(list)
    for d in sorted(r["dets"], key=lambda d: -d[1])[:100]:
        out[coarse(d[0], agnostic)].append((d[1], d[2:6]))
    return out


def gts(r, agnostic):
    g = collections.defaultdict(list)
    for x in r["gt"]:
        g["Any" if agnostic else x[0]].append(x[1:5])
    return g


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    i = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / np.maximum(aa[:, None] + bb[None, :] - i, 1e-9)


def ap_stats(pages, cls):
    ngt = sum(len(g.get(cls, [])) for g, _ in pages)
    if ngt == 0:
        return None
    aps = []
    for t in THR:
        recs = []
        for g, p in pages:
            gb = g.get(cls, [])
            pb = p.get(cls, [])
            M = iou_mat([b for _, b in pb], gb)
            used = set()
            for k, (s, _) in enumerate(pb):
                j, best = -1, t
                for jj in range(len(gb)):
                    if jj not in used and M[k, jj] >= best:
                        best, j = M[k, jj], jj
                if j >= 0:
                    used.add(j)
                recs.append((s, j >= 0))
        recs.sort(key=lambda x: -x[0])
        if not recs:
            aps.append(0.0)
            continue
        tp = np.cumsum([r[1] for r in recs])
        fp = np.cumsum([not r[1] for r in recs])
        rec = tp / ngt
        prec = tp / np.maximum(tp + fp, 1e-9)
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        pts = []
        for x in np.linspace(0, 1, 101):
            k = np.searchsorted(rec, x, side="left")
            pts.append(prec[k] if k < len(prec) else 0.0)
        aps.append(float(np.mean(pts)))
    return dict(map=float(np.mean(aps)), map50=aps[0], ngt=ngt)


def page_f1(g, p, classes):
    tp = npred = ngt = 0
    for c in classes:
        gb = g.get(c, [])
        pb = [b for s, b in p.get(c, []) if s >= 0.5]
        ngt += len(gb)
        npred += len(pb)
        M = iou_mat(pb, gb)
        used = set()
        for k in range(len(pb)):
            j = int(np.argmax(M[k])) if len(gb) else -1
            if j >= 0 and M[k, j] >= 0.5 and j not in used:
                used.add(j)
                tp += 1
    return 1.0 if ngt + npred == 0 else 2 * tp / (ngt + npred)


def agree(a, b):
    """Per-page agreement of two detection sets (score >= 0.5, same coarse class, IoU >= 0.5)."""
    pa = preds(a, False)
    pb = preds(b, False)
    tp = na = nb = 0
    for c in set(pa) | set(pb):
        xa = [x for s, x in pa.get(c, []) if s >= 0.5]
        xb = [x for s, x in pb.get(c, []) if s >= 0.5]
        na += len(xa)
        nb += len(xb)
        M = iou_mat(xa, xb)
        used = set()
        for k in range(len(xa)):
            if len(xb):
                j = int(np.argmax(M[k]))
                if M[k, j] >= 0.5 and j not in used:
                    used.add(j)
                    tp += 1
    return 1.0 if na + nb == 0 else 2 * tp / (na + nb)


def boot(d, n=2000, seed=0):
    rng = random.Random(seed)
    m = sum(d) / len(d)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def wil(x, y):
    d = [a - b for a, b in zip(x, y)]
    if not any(abs(v) > 1e-12 for v in d):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def load(pattern):
    out = {}
    for p in glob.glob(pattern):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main():
    data, dets, outp = sys.argv[1:4]
    L = []
    P = L.append
    P("# Round 7: heron fp32 vs heron int8 on more layout data\n")
    P("Same pages for both models (docling.rs layout_heron.onnx / layout_heron_int8.onnx, same preprocessing). "
      "Sets with ground truth: COCO-style mAP@[.5:.95] (101-point, max 100 detections per page) on coarse classes "
      "(Text merges text, titles, section headers, lists, footnotes; Table includes Document Index) and "
      "class-agnostic; paired per-page F1 (score >= 0.5, IoU 0.5) with a 95% bootstrap CI and Wilcoxon p.\n")
    summary = {}
    for ds in ("d4la", "m6doc", "publaynet", "real"):
        pages = [json.loads(x) for x in open(os.path.join(data, ds, "pages.jsonl"), encoding="utf-8")] \
            if os.path.exists(os.path.join(data, ds, "pages.jsonl")) else []
        f32 = load(os.path.join(dets, f"{ds}-heron-*.jsonl"))
        i8 = load(os.path.join(dets, f"{ds}-heron_int8-*.jsonl"))
        ids = [p for p in pages if p["id"] in f32 and p["id"] in i8]
        if not ids:
            P(f"\n## {ds}: no pages\n")
            continue
        sec32 = sorted(f32[p["id"]]["sec"] for p in ids)
        sec8 = sorted(i8[p["id"]]["sec"] for p in ids)
        speed = (f"CPU ms per page: fp32 p50 {1000 * sec32[len(sec32) // 2]:.0f}, "
                 f"int8 p50 {1000 * sec8[len(sec8) // 2]:.0f}.")
        if ds == "real":
            P(f"\n## Real PDF pages without ground truth: {len(ids)} pages\n")
            P("Agreement = F1 between the two models' detections (score >= 0.5, same coarse class, IoU >= 0.5).\n")
            P("| source | pages | mean agreement | pages identical (1.0) | pages below 0.9 | pages below 0.75 |")
            P("|---|---|---|---|---|---|")
            by = collections.defaultdict(list)
            for p in ids:
                by[p["src"]].append(agree(f32[p["id"]], i8[p["id"]]))
            allv = [v for vs in by.values() for v in vs]
            for src in sorted(by) + ["all"]:
                vs = allv if src == "all" else by[src]
                P(f"| {src} | {len(vs)} | {np.mean(vs):.4f} | {np.mean([v == 1.0 for v in vs]):.1%} | "
                  f"{np.mean([v < 0.9 for v in vs]):.1%} | {np.mean([v < 0.75 for v in vs]):.1%} |")
            n32 = sum(sum(1 for d in f32[p["id"]]["dets"] if d[1] >= 0.5) for p in ids)
            n8 = sum(sum(1 for d in i8[p["id"]]["dets"] if d[1] >= 0.5) for p in ids)
            P(f"\nBoxes at score >= 0.5: fp32 {n32}, int8 {n8}. {speed}")
            summary["real"] = dict(pages=len(ids), mean_agreement=float(np.mean(allv)),
                                   identical=float(np.mean([v == 1.0 for v in allv])))
            continue
        agn = ds == "m6doc"
        P(f"\n## {ds}: {len(ids)} pages with ground truth\n")
        P("| class | GT boxes | fp32 mAP | int8 mAP | fp32 mAP50 | int8 mAP50 |")
        P("|---|---|---|---|---|---|")
        classes = ["Any"] if agn else CLASSES
        res = {}
        for cls in classes + ([] if agn else ["Any"]):
            ag = agn or cls == "Any"
            pa = [(gts(p, ag), preds(f32[p["id"]], ag)) for p in ids]
            pb = [(gts(p, ag), preds(i8[p["id"]], ag)) for p in ids]
            a = ap_stats(pa, cls)
            b = ap_stats(pb, cls)
            if not a:
                continue
            res[cls] = (a, b)
            P(f"| {cls} | {a['ngt']} | {a['map']:.3f} | {b['map']:.3f} | {a['map50']:.3f} | {b['map50']:.3f} |")
        cl = [c for c in res if c != "Any"] or ["Any"]
        if cl != ["Any"]:
            m32 = float(np.mean([res[c][0]["map"] for c in cl]))
            m8 = float(np.mean([res[c][1]["map"] for c in cl]))
            P(f"| **mean over coarse classes** | | **{m32:.3f}** | **{m8:.3f}** | | |")
        f_a = [page_f1(gts(p, agn), preds(f32[p["id"]], agn), cl) for p in ids]
        f_b = [page_f1(gts(p, agn), preds(i8[p["id"]], agn), cl) for p in ids]
        d = [x - y for x, y in zip(f_b, f_a)]
        m, lo, hi = boot(d)
        P(f"\nPaired per-page F1: fp32 {np.mean(f_a):.4f}, int8 {np.mean(f_b):.4f}; int8 - fp32 {m:+.4f} "
          f"[95% CI {lo:+.4f}, {hi:+.4f}], Wilcoxon p {wil(f_b, f_a):.2g}; int8 better on "
          f"{sum(x > 1e-9 for x in d)} pages, worse on {sum(x < -1e-9 for x in d)}. {speed}")
        ag_vals = [agree(f32[p["id"]], i8[p["id"]]) for p in ids]
        P(f"Agreement between the two models: mean {np.mean(ag_vals):.4f}, identical pages "
          f"{np.mean([v == 1.0 for v in ag_vals]):.1%}.")
        summary[ds] = dict(pages=len(ids), f1_diff=[m, lo, hi], agnostic_map=[res["Any"][0]["map"], res["Any"][1]["map"]])
    P("\n```json\n" + json.dumps(summary, indent=1) + "\n```")
    text = "\n".join(L) + "\n"
    open(outp, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
