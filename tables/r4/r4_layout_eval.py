"""RF-DETR-Large vs heron INT8 layout comparison + missed-table arm.
usage: r4_layout_eval.py RF_ROOT HERON_ROOT R3DET_ROOT ODB_JSON"""
import os, sys, json, glob, random, collections
from r4_common import *
import numpy as np

DLN_C = ["Table", "Picture", "Text", "Title", "List-item", "Formula", "Caption", "Footnote", "Page-header", "Page-footer", "Section-header"]
ODB_C = ["Table", "Picture", "Text", "Title", "Formula", "Caption", "Footnote", "Page-header", "Page-footer"]
PRED_MAP = {"Document Index": "Table", "Code": "Text", "List - item": "List-item", "Page - footer": "Page-footer", "Page - header": "Page-header", "Section - header": "Section-header"}
ODB_GT = {"title": "Title", "text_block": "Text", "figure": "Picture", "figure_caption": "Caption", "figure_footnote": "Footnote", "table": "Table",
          "table_caption": "Caption", "table_footnote": "Footnote", "equation_isolated": "Formula", "equation_caption": "Caption", "header": "Page-header",
          "footer": "Page-footer", "page_footnote": "Footnote", "code_txt": "Text", "code_txt_caption": "Caption", "reference": "Text"}
THR = [0.5 + 0.05 * i for i in range(10)]


def load(pattern):
    d = {}
    for p in glob.glob(pattern, recursive=True):
        for r in jl(p): d[r["id"]] = r
    return d


def preds(r, ds):
    out = collections.defaultdict(list)
    for d in sorted(r["dets"], key=lambda d: -d[1])[:100] if r else []:
        lab = PRED_MAP.get(d[0], d[0])
        if ds == "odb":
            lab = {"Section-header": "Title", "List-item": "Text"}.get(lab, lab)
        out[lab].append((d[1], d[2:6]))
    return out


def iou_mat(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    i = ix * iy; aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / np.maximum(aa[:, None] + bb[None, :] - i, 1e-9)


def ap_stats(pages, cls, P):
    """pages: list of (gt dict cls->boxes, pred dict cls->[(s, box)]); COCO-style 101-point AP"""
    ngt = sum(len(g.get(cls, [])) for g, _ in pages)
    if ngt == 0: return None
    res = {}
    for t in THR:
        recs = []
        for g, p in pages:
            gb = g.get(cls, []); pb = p.get(cls, [])
            M = iou_mat([b for _, b in pb], gb); used = set()
            for k, (s, _) in enumerate(pb):
                j = -1; best = t
                for jj in range(len(gb)):
                    if jj not in used and M[k, jj] >= best: best = M[k, jj]; j = jj
                if j >= 0: used.add(j)
                recs.append((s, j >= 0))
        recs.sort(key=lambda x: -x[0])
        tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
        rec = tp / ngt; prec = tp / np.maximum(tp + fp, 1e-9)
        for i in range(len(prec) - 2, -1, -1): prec[i] = max(prec[i], prec[i + 1])
        ap = np.mean([prec[np.searchsorted(rec, x, side="left")] if np.searchsorted(rec, x, side="left") < len(prec) else 0.0 for x in np.linspace(0, 1, 101)]) if len(recs) else 0.0
        res[round(t, 2)] = ap
        if abs(t - 0.5) < 1e-9:
            sel = [r for r in recs if r[0] >= 0.5]; tp5 = sum(1 for r in sel if r[1])
            res["R"] = tp5 / ngt; res["P"] = tp5 / max(1, len(sel))
    res["map"] = float(np.mean([res[round(t, 2)] for t in THR])); res["map50"] = res[0.5]; res["ngt"] = ngt
    return res


def page_f1(g, p, classes):
    tp = npred = ngt = 0
    for c in classes:
        gb = g.get(c, []); pb = [b for s, b in p.get(c, []) if s >= 0.5]
        ngt += len(gb); npred += len(pb)
        M = iou_mat(pb, gb); used = set()
        for k in range(len(pb)):
            j = int(np.argmax(M[k])) if len(gb) else -1
            if j >= 0 and M[k, j] >= 0.5 and j not in used: used.add(j); tp += 1
    if ngt + npred == 0: return 1.0
    return 2 * tp / (ngt + npred)


def boot(d, n=2000, seed=0):
    rng = random.Random(seed); m = sum(d) / len(d)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main():
    rfroot, hroot, yroot, odbj = sys.argv[1:5]
    rf = load(os.path.join(rfroot, "**", "rfdetr-*.jsonl")); he = load(os.path.join(hroot, "**", "layout-heron_int8-*.jsonl"))
    yx = load(os.path.join(yroot, "**", "det-yolox_l-*.jsonl"))
    gt = {}
    for r in load(os.path.join(hroot, "**", "dln-gt-*.jsonl")).values():
        g = collections.defaultdict(list)
        for x in r["gt"]: g[x[0]].append(x[1:5])
        gt[r["id"]] = g
    unk = collections.Counter()
    if os.path.exists(odbj):
        for rec in json.load(open(odbj, encoding="utf-8")):
            pi = rec["page_info"]
            if pi["page_attribute"].get("language") != "english": continue
            g = collections.defaultdict(list)
            for e in rec["layout_dets"]:
                if e.get("ignore"): continue
                c = "Table" if e["category_type"] == "table_mask" else ODB_GT.get(e["category_type"])
                if c is None: unk[e["category_type"]] += 1; continue
                pl = e["poly"]; xs, ys = pl[0::2], pl[1::2]; g[c].append([min(xs), min(ys), max(xs), max(ys)])
            gt["odb:" + os.path.basename(pi["image_path"])] = g
    out = []; P = out.append
    P("## Layout: RF-DETR-Large (DocLayNet fine-tune) vs heron INT8")
    P("Model nie3e/rf-detr-large-doclay @fecf537 (card: Apache-2.0, base Roboflow/rf-detr-large Apache-2.0, trained on DocLayNet v1.2 train; no LICENSE file in the model repo), "
      "transformers 5.22.2 + torch 2.15.0 CPU fp32; heron INT8 = docling layout_heron_int8.onnx detections from order run 44477997398. Same pages, same GT; "
      "COCO-style AP (101-point, max 100 detections per page), recall/precision at score >= 0.5 and IoU 0.5. "
      f"OmniDocBench: Section-header -> Title and List-item -> Text (ODB has no such split); ODB categories not mapped: {dict(unk)}.")
    summ = {}
    for ds, classes in (("dln", DLN_C), ("odb", ODB_C)):
        ids = sorted(i for i in rf if i.startswith(ds + ":") and i in he and i in gt)
        if not ids: continue
        pages = {m: [(gt[i], preds(src[i], ds)) for i in ids] for m, src in (("rf", rf), ("heron", he))}
        P(f"\n### {'DocLayNet test' if ds == 'dln' else 'OmniDocBench English'}: {len(ids)} pages")
        P("| class | GT | RF-DETR mAP50-95 | heron mAP50-95 | RF-DETR mAP50 | heron mAP50 | RF-DETR R / P | heron R / P |\n|---|---|---|---|---|---|---|---|")
        tot = {"rf": [], "heron": []}
        for c in classes:
            a, b = ap_stats(pages["rf"], c, None), ap_stats(pages["heron"], c, None)
            if not a: continue
            tot["rf"].append(a); tot["heron"].append(b)
            P(f"| {c} | {a['ngt']} | {a['map']:.3f} | {b['map']:.3f} | {a['map50']:.3f} | {b['map50']:.3f} | {a['R']:.3f} / {a['P']:.3f} | {b['R']:.3f} / {b['P']:.3f} |")
        mm = lambda m, k: float(np.mean([x[k] for x in tot[m]]))
        P(f"| **mean over classes** | | **{mm('rf', 'map'):.3f}** | **{mm('heron', 'map'):.3f}** | **{mm('rf', 'map50'):.3f}** | **{mm('heron', 'map50'):.3f}** | {mm('rf', 'R'):.3f} / {mm('rf', 'P'):.3f} | {mm('heron', 'R'):.3f} / {mm('heron', 'P'):.3f} |")
        f_rf = [page_f1(gt[i], preds(rf[i], ds), classes) for i in ids]; f_h = [page_f1(gt[i], preds(he[i], ds), classes) for i in ids]
        d = [a - b for a, b in zip(f_rf, f_h)]; m, lo, hi = boot(d)
        P(f"\nPaired per-page F1 (all classes, score >= 0.5, IoU 0.5): RF-DETR {np.mean(f_rf):.4f} vs heron {np.mean(f_h):.4f}; difference {m:+.4f} [95% bootstrap CI {lo:+.4f}, {hi:+.4f}], "
          f"Wilcoxon p {wilcoxon_p(f_rf, f_h):.2g}; RF-DETR better on {sum(x > 1e-9 for x in d)} pages, worse on {sum(x < -1e-9 for x in d)}.")
        sr = sorted(rf[i]["sec"] for i in ids); sh = sorted(he[i]["sec"] for i in ids)
        P(f"CPU ms per page (4-vCPU runner): RF-DETR p50 {1000 * sr[len(sr) // 2]:.0f} / mean {1000 * np.mean(sr):.0f}; heron INT8 p50 {1000 * sh[len(sh) // 2]:.0f} / mean {1000 * np.mean(sh):.0f}.")
        summ[ds] = dict(pages=len(ids), rf_map=mm("rf", "map"), heron_map=mm("heron", "map"), rf_map50=mm("rf", "map50"), heron_map50=mm("heron", "map50"), f1_diff=[m, lo, hi],
                        rf_ms=1000 * sr[len(sr) // 2], heron_ms=1000 * sh[len(sh) // 2])
        # missed-table arm
        ids2 = [i for i in ids if (not yx) or i in yx]
        ht = {i: [d[2:6] for d in he[i]["dets"] if PRED_MAP.get(d[0], d[0]) == "Table" and d[1] >= 0.5] for i in ids2}
        miss = {}; tp = 0; ng = 0
        for i in ids2:
            g = gt[i].get("Table", []); ng += len(g)
            M = iou_mat(ht[i], g); used = set()
            for k in np.argsort(-M.max(1)) if len(ht[i]) and len(g) else []:
                j = int(np.argmax(M[k]))
                if M[k, j] >= 0.5 and j not in used: used.add(j)
            tp += len(used); miss[i] = [g[j] for j in range(len(g)) if j not in used]
        nm = sum(len(v) for v in miss.values())
        P(f"\n**Missed-table arm ({ds})**: {len(ids2)} pages with heron, RF-DETR{' and YOLOX-L (round 3, run 44689850800)' if yx else ''}; heron recall@.5 {tp / max(1, ng):.3f}, misses {nm}")
        P("| second detector | where | added / 100 pages | misses recovered @.5 | inc. recall | precision of added | false tables / 100 pages | heron+added recall |\n|---|---|---|---|---|---|---|---|")
        arm = {}
        for name, src, lab in (("RF-DETR-Large", rf, "Table"), ("YOLOX-L", yx, "Table")):
            if not src: continue
            for tau in (0.5, 0.7):
                for where in ("uncovered regions", "pages without a heron table"):
                    na = rec = good = false = 0
                    for i in ids2:
                        tb = [d[2:6] for d in src[i]["dets"] if PRED_MAP.get(d[0], d[0]) == lab and d[1] >= tau]
                        a = [b for b in tb if cover(b, ht[i]) < 0.3] if where == "uncovered regions" else (tb if not ht[i] else [])
                        na += len(a); g = gt[i].get("Table", [])
                        M = iou_mat(a, miss[i]); used = set()
                        for k in range(len(a)):
                            if len(miss[i]):
                                j = int(np.argmax(M[k]))
                                if M[k, j] >= 0.5 and j not in used: used.add(j); good += 1
                        rec += len(used)
                        false += sum(1 for b in a if all(iou(b, x) < 0.5 for x in g))
                    arm[f"{name}|{tau}|{where}"] = dict(rec=rec, prec=good / max(1, na), false100=100 * false / len(ids2))
                    P(f"| {name} >= {tau} | {where} | {100 * na / len(ids2):.1f} | {rec} of {nm} | {rec / max(1, nm):.3f} | {good / max(1, na):.3f} | {100 * false / len(ids2):.1f} | {(tp + rec) / max(1, ng):.3f} |")
        summ[ds]["arm"] = arm; summ[ds]["misses"] = nm
    txt = "\n".join(out); print(txt)
    os.makedirs(R4OUT, exist_ok=True)
    open(os.path.join(R4OUT, "layout.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(summ, open(os.path.join(R4OUT, "layout_summary.json"), "w"))


if __name__ == "__main__":
    main()
