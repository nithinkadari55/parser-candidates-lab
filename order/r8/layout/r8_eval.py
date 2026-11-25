"""Round 8, test B: heron fp32 vs heron int8 on the sets not covered in round 7 (same metrics as round 7:
COCO-style mAP per coarse class and class-agnostic, paired per-page F1 with a bootstrap CI and Wilcoxon p,
agreement on real pages). Sets whose ground truth covers only some classes (eval_classes on the page) are
scored on those classes only, and without the class-agnostic score.
usage: r8_eval.py DATA_DIR DETS_DIR OUT.md [OUT.json]"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "layout", "r7"))
from r7_eval import CLASSES, agree, ap_stats, boot, gts, load, page_f1, preds, wil  # noqa: E402

SETS = ("pod2017", "docbank", "iiitar13k", "real2")
TITLES = {"pod2017": "ICDAR-2017 POD (scientific papers; table subset, Table boxes only)",
          "docbank": "DocBank test sample (arXiv papers, weakly supervised boxes)",
          "iiitar13k": "IIIT-AR-13K (annual reports; Table and Picture boxes only)",
          "real2": "Real PDFs and scans without ground truth"}


def main():
    data, dets, outp = sys.argv[1:4]
    outj = sys.argv[4] if len(sys.argv) > 4 else None
    L = []
    P = L.append
    P("# Round 8, test B: heron fp32 vs heron int8 on more layout sets\n")
    P("Same pages for both models (docling.rs layout_heron.onnx / layout_heron_int8.onnx, same preprocessing, "
      "code of round 7). COCO-style mAP@[.5:.95] (101-point, max 100 detections per page) on coarse classes "
      "(Text merges text, titles, section headers, lists, footnotes; Table includes Document Index) and "
      "class-agnostic where the ground truth is complete; paired per-page F1 (score >= 0.5, IoU 0.5) with a 95% "
      "bootstrap CI and Wilcoxon p. Real pages: agreement F1 between the two models.\n")
    summary = {}
    for ds in SETS:
        pf = os.path.join(data, ds, "pages.jsonl")
        pages = [json.loads(x) for x in open(pf, encoding="utf-8")] if os.path.exists(pf) else []
        f32 = load(os.path.join(dets, f"{ds}-heron-*.jsonl"))
        i8 = load(os.path.join(dets, f"{ds}-heron_int8-*.jsonl"))
        ids = [p for p in pages if p["id"] in f32 and p["id"] in i8]
        if not ids:
            P(f"\n## {TITLES[ds]}: no pages\n")
            summary[ds] = None
            continue
        sec32 = sorted(f32[p["id"]]["sec"] for p in ids)
        sec8 = sorted(i8[p["id"]]["sec"] for p in ids)
        speed = (f"CPU ms per page: fp32 p50 {1000 * sec32[len(sec32) // 2]:.0f}, "
                 f"int8 p50 {1000 * sec8[len(sec8) // 2]:.0f}.")
        if ds == "real2":
            P(f"\n## {TITLES[ds]}: {len(ids)} pages\n")
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
            summary[ds] = dict(pages=len(ids), mean_agreement=float(np.mean(allv)),
                               identical=float(np.mean([v == 1.0 for v in allv])),
                               by_source={s: [len(v), float(np.mean(v))] for s, v in by.items()})
            continue
        partial = ids[0].get("eval_classes")
        P(f"\n## {TITLES[ds]}: {len(ids)} pages with ground truth\n")
        P("| class | GT boxes | fp32 mAP | int8 mAP | fp32 mAP50 | int8 mAP50 |")
        P("|---|---|---|---|---|---|")
        classes = partial or CLASSES
        res = {}
        for cls in classes + ([] if partial else ["Any"]):
            ag = cls == "Any"
            pa = [(gts(p, ag), preds(f32[p["id"]], ag)) for p in ids]
            pb = [(gts(p, ag), preds(i8[p["id"]], ag)) for p in ids]
            a = ap_stats(pa, cls)
            b = ap_stats(pb, cls)
            if not a:
                continue
            res[cls] = (a, b)
            P(f"| {cls} | {a['ngt']} | {a['map']:.3f} | {b['map']:.3f} | {a['map50']:.3f} | {b['map50']:.3f} |")
        cl = [c for c in res if c != "Any"]
        m32 = float(np.mean([res[c][0]["map"] for c in cl]))
        m8 = float(np.mean([res[c][1]["map"] for c in cl]))
        if len(cl) > 1:
            P(f"| **mean over scored classes** | | **{m32:.3f}** | **{m8:.3f}** | | |")
        f_a = [page_f1(gts(p, False), preds(f32[p["id"]], False), cl) for p in ids]
        f_b = [page_f1(gts(p, False), preds(i8[p["id"]], False), cl) for p in ids]
        d = [x - y for x, y in zip(f_b, f_a)]
        m, lo, hi = boot(d)
        P(f"\nPaired per-page F1 ({', '.join(cl)}): fp32 {np.mean(f_a):.4f}, int8 {np.mean(f_b):.4f}; int8 - fp32 "
          f"{m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}], Wilcoxon p {wil(f_b, f_a):.2g}; int8 better on "
          f"{sum(x > 1e-9 for x in d)} pages, worse on {sum(x < -1e-9 for x in d)}. {speed}")
        ag_vals = [agree(f32[p["id"]], i8[p["id"]]) for p in ids]
        P(f"Agreement between the two models: mean {np.mean(ag_vals):.4f}, identical pages "
          f"{np.mean([v == 1.0 for v in ag_vals]):.1%}.")
        summary[ds] = dict(pages=len(ids), classes=cl, f1=[float(np.mean(f_a)), float(np.mean(f_b))],
                           f1_diff=[m, lo, hi], map_mean=[m32, m8],
                           per_class={c: [res[c][0]["map"], res[c][1]["map"]] for c in res},
                           agnostic_map=[res["Any"][0]["map"], res["Any"][1]["map"]] if "Any" in res else None)
    P("\n```json\n" + json.dumps(summary, indent=1) + "\n```")
    text = "\n".join(L) + "\n"
    open(outp, "w", encoding="utf-8").write(text)
    if outj:
        json.dump(summary, open(outj, "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
