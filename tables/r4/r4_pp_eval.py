"""PP-DocLayout_plus-L vs heron INT8 (and RF-DETR-Large, YOLOX-L for reference) with the RF-DETR harness.
usage: r4_pp_eval.py PP_ROOT HERON_ROOT R3DET_ROOT RF_ROOT ODB_JSON [PARITY_JSON]"""
import os, sys, json, glob, collections
from r4_common import *
import numpy as np
from r4_layout_eval import load, ap_stats, page_f1, iou_mat, boot, ODB_GT, PRED_MAP

PPMAP = {"doc_title": "Title", "paragraph_title": "Section-header", "image": "Picture", "chart": "Picture", "formula": "Formula", "table": "Table",
         "header": "Page-header", "footer": "Page-footer", "footnote": "Footnote", "figure_title": "Caption", "text": "Text"}
EXT_TEXT = {"abstract", "content", "reference", "reference_content", "aside_text", "algorithm"}
DLN_S = ["Table", "Picture", "Text", "Title", "Section-header", "Formula", "Caption", "Footnote", "Page-header", "Page-footer"]
ODB_S = ["Table", "Picture", "Text", "Title", "Formula", "Caption", "Footnote", "Page-header", "Page-footer"]
GENRE = {"academic_literature": "paper", "book": "book", "exam_paper": "exam", "PPT2PDF": "slides", "newspaper": "newspaper", "magazine": "magazine", "research_report": "report"}


def preds(r, model, ds, merge_list=False, ext=False):
    out = collections.defaultdict(list)
    if not r: return out
    for d in sorted(r["dets"], key=lambda d: -d[1])[:100]:
        if model == "pp":
            lab = PPMAP.get(d[0]) or ("Text" if ext and d[0] in EXT_TEXT else None)
            if lab is None: continue
        else:
            lab = PRED_MAP.get(d[0], d[0])
        if ds == "odb": lab = {"Section-header": "Title", "List-item": "Text"}.get(lab, lab)
        if merge_list and lab == "List-item": lab = "Text"
        out[lab].append((d[1], d[2:6]))
    return out


def gt_merge(g):
    h = collections.defaultdict(list, {k: list(v) for k, v in g.items()})
    h["Text"] = list(g.get("Text", [])) + list(g.get("List-item", [])); h.pop("List-item", None)
    return h


def main():
    pproot, hroot, yroot, rfroot, odbj = sys.argv[1:6]
    parity = json.load(open(sys.argv[6])) if len(sys.argv) > 6 and os.path.exists(sys.argv[6]) else None
    src = {"pp": load(os.path.join(pproot, "**", "ppplus-*.jsonl")), "heron": load(os.path.join(hroot, "**", "layout-heron_int8-*.jsonl")),
           "rf": load(os.path.join(rfroot, "**", "rfdetr-*.jsonl"))}
    yx = load(os.path.join(yroot, "**", "det-yolox_l-*.jsonl"))
    gt = {}
    for r in load(os.path.join(hroot, "**", "dln-gt-*.jsonl")).values():
        g = collections.defaultdict(list)
        for x in r["gt"]: g[x[0]].append(x[1:5])
        gt[r["id"]] = g
    genre = {}
    for rec in json.load(open(odbj, encoding="utf-8")):
        pi = rec["page_info"]
        if pi["page_attribute"].get("language") != "english": continue
        g = collections.defaultdict(list)
        for e in rec["layout_dets"]:
            if e.get("ignore"): continue
            c = "Table" if e["category_type"] == "table_mask" else ODB_GT.get(e["category_type"])
            if c is None: continue
            pl = e["poly"]; xs, ys = pl[0::2], pl[1::2]; g[c].append([min(xs), min(ys), max(xs), max(ys)])
        pid = "odb:" + os.path.basename(pi["image_path"]); gt[pid] = g
        genre[pid] = pi["page_attribute"].get("data_source", "unknown")
    names = {"pp": "PP-DocLayout_plus-L", "heron": "heron INT8", "rf": "RF-DETR-Large"}
    out = []; P = out.append
    P("## Layout: PP-DocLayout_plus-L vs heron INT8")
    P("Model PaddlePaddle/PP-DocLayout_plus-L_onnx @feb7461 (inference.onnx, 130 MB; model card tag apache-2.0, no LICENSE file in either HF repo; PaddleOCR / PaddleX code Apache-2.0), "
      "ONNX Runtime CPU with the same default thread settings as heron INT8; pre/post-processing per inference.yml and PaddleX 3.7.2 DetPostProcess "
      "(800x800 cubic resize, /255, threshold, no layout NMS, large-image filter). Same pages, GT and scoring as the RF-DETR-Large section. "
      "Fixed label map: doc_title->Title, paragraph_title->Section-header, image+chart->Picture, formula->Formula, table->Table, header->Page-header, "
      "footer->Page-footer, footnote->Footnote, figure_title->Caption, text->Text; all other PP labels dropped (abstract, content, reference, reference_content, "
      "aside_text, algorithm, number, formula_number, seal). PP has no list class: List-item is not scored for it; 'Text+List-item' merges both classes for every model and GT.")
    if parity:
        s = parity["summary"]
        P(f"\n**ONNX vs native Paddle parity** (paddlepaddle 3.2.2, paddleocr 3.7.0, paddlex 3.7.2, same weights, threshold {s['threshold']}, {s['pages']} pages): "
          f"{s['matched_iou95_same_label']} of {s['native_boxes']} native boxes matched by an ONNX box of the same label at IoU >= 0.95 ({100 * s['matched_of_native']:.1f}%; "
          f"{100 * s['matched_of_onnx']:.1f}% of {s['onnx_boxes']} ONNX boxes), mean |score difference| {s['mean_score_diff']}, max {s['max_score_diff']}, mean IoU {s['mean_iou']}; "
          f"native {s['native_ms']:.0f} ms/page vs ONNX {s['onnx_ms']:.0f} ms/page.")
    summ = {}
    for ds, classes in (("dln", DLN_S), ("odb", ODB_S)):
        ids = sorted(i for i in src["pp"] if i.startswith(ds + ":") and i in src["heron"] and i in gt)
        if not ids: continue
        models = [m for m in ("pp", "heron", "rf") if all(i in src[m] for i in ids)]
        P(f"\n### {'DocLayNet test' if ds == 'dln' else 'OmniDocBench English'}: {len(ids)} pages")
        P("| class | GT | " + " | ".join(f"{names[m]} mAP50-95" for m in models) + " | " + " | ".join(f"{names[m]} mAP50" for m in models) + " | " + " | ".join(f"{names[m]} R / P" for m in models) + " |")
        P("|" + "---|" * (2 + 3 * len(models)))
        res = {m: {} for m in models}
        rows = list(classes) + (["Text+List-item"] if ds == "dln" else []) + ["Text (PP extended map)"]
        for c in rows:
            st = {}
            for m in models:
                if c == "Text+List-item":
                    pages = [(gt_merge(gt[i]), preds(src[m][i], m, ds, merge_list=True)) for i in ids]; cc = "Text"
                elif c == "Text (PP extended map)":
                    if m != "pp": continue
                    pages = [(gt[i], preds(src[m][i], m, ds, ext=True)) for i in ids]; cc = "Text"
                else:
                    pages = [(gt[i], preds(src[m][i], m, ds)) for i in ids]; cc = c
                st[m] = ap_stats(pages, cc, None)
            if not st or not any(st.values()): continue
            ref = next(v for v in st.values() if v)
            f = lambda m, k: (f"{st[m][k]:.3f}" if st.get(m) else "-")
            P(f"| {c} | {ref['ngt']} | " + " | ".join(f(m, "map") for m in models) + " | " + " | ".join(f(m, "map50") for m in models) + " | "
              + " | ".join((f"{st[m]['R']:.3f} / {st[m]['P']:.3f}" if st.get(m) else "-") for m in models) + " |")
            if c in classes:
                for m in models: res[m][c] = st[m]
        mm = {m: (float(np.mean([v["map"] for v in res[m].values()])), float(np.mean([v["map50"] for v in res[m].values()]))) for m in models}
        P("| **shared-class mean** | | " + " | ".join(f"**{mm[m][0]:.3f}**" for m in models) + " | " + " | ".join(f"**{mm[m][1]:.3f}**" for m in models) + " | " + " | ".join("" for m in models) + " |")
        if ds == "dln":
            nl = sum(len(gt[i].get("List-item", [])) for i in ids)
            P(f"\nCoverage: PP-DocLayout_plus-L covers 10 of the 11 DocLayNet classes; List-item ({nl} GT boxes) has no PP class and is excluded from its shared-class mean "
              "(PP predicts list items as text, which counts against its Text precision; see the Text+List-item row).")
        drop = collections.Counter(d[0] for i in ids for d in src["pp"][i]["dets"] if d[1] >= 0.5 and d[0] not in PPMAP)
        P(f"PP detections (score >= 0.5) under dropped labels: {dict(drop.most_common())}.")
        fcls = [c for c in classes if c != "List-item"]
        F = {m: [page_f1(gt_merge(gt[i]) if ds == "dln" else gt[i], preds(src[m][i], m, ds, merge_list=True), fcls) for i in ids] for m in models}
        d = [a - b for a, b in zip(F["pp"], F["heron"])]; b_ = boot(d)
        P(f"\nPaired per-page F1 (shared classes, Text+List-item merged, score >= 0.5, IoU 0.5): PP {np.mean(F['pp']):.4f} vs heron {np.mean(F['heron']):.4f}; "
          f"difference {b_[0]:+.4f} [95% bootstrap CI {b_[1]:+.4f}, {b_[2]:+.4f}], Wilcoxon p {wilcoxon_p(F['pp'], F['heron']):.2g}; PP better on {sum(x > 1e-9 for x in d)} pages, worse on {sum(x < -1e-9 for x in d)}."
          + (f" RF-DETR-Large on the same basis: {np.mean(F['rf']):.4f}." if "rf" in F else ""))
        ms = {m: sorted(src[m][i]["sec"] for i in ids) for m in models}
        P("CPU ms per page (4-vCPU runner): " + "; ".join(f"{names[m]} p50 {1000 * ms[m][len(ms[m]) // 2]:.0f} / mean {1000 * np.mean(ms[m]):.0f}" for m in models) + ".")
        summ[ds] = dict(pages=len(ids), map={m: mm[m][0] for m in models}, map50={m: mm[m][1] for m in models}, f1_diff=list(b_), ms={m: 1000 * ms[m][len(ms[m]) // 2] for m in models})
        if ds == "odb":
            P("\n**Per genre (OmniDocBench data_source)** - shared-class mean mAP50-95 and paired page F1 (PP - heron)")
            P("| genre | pages | " + " | ".join(f"{names[m]} mAP50-95" for m in models) + " | PP - heron page F1 [95% CI] |\n|" + "---|" * (3 + len(models)))
            gs = {}
            for g_, lab in list(GENRE.items()) + [(k, k) for k in sorted(set(genre.values()) - set(GENRE))]:
                gi = [i for i in ids if genre.get(i) == g_]
                if len(gi) < 5: continue
                vals = {}
                for m in models:
                    aps = [ap_stats([(gt[i], preds(src[m][i], m, ds)) for i in gi], c, None) for c in classes]
                    aps = [a["map"] for a in aps if a]; vals[m] = float(np.mean(aps)) if aps else float("nan")
                ii = [ids.index(i) for i in gi]; dd = [F["pp"][k] - F["heron"][k] for k in ii]; bb = boot(dd)
                gs[lab] = dict(pages=len(gi), **vals, f1_diff=list(bb))
                P(f"| {lab} | {len(gi)} | " + " | ".join(f"{vals[m]:.3f}" for m in models) + f" | {bb[0]:+.4f} [{bb[1]:+.4f}, {bb[2]:+.4f}] |")
            summ[ds]["genre"] = gs
        # missed-table arm
        ids2 = [i for i in ids if (not yx) or i in yx]
        ht = {i: [d[2:6] for d in src["heron"][i]["dets"] if PRED_MAP.get(d[0], d[0]) == "Table" and d[1] >= 0.5] for i in ids2}
        miss = {}; tp = 0; ng = 0
        for i in ids2:
            g = gt[i].get("Table", []); ng += len(g)
            M = iou_mat(ht[i], g); used = set()
            for k in (np.argsort(-M.max(1)) if len(ht[i]) and len(g) else []):
                j = int(np.argmax(M[k]))
                if M[k, j] >= 0.5 and j not in used: used.add(j)
            tp += len(used); miss[i] = [g[j] for j in range(len(g)) if j not in used]
        nm = sum(len(v) for v in miss.values())
        P(f"\n**Missed-table arm ({ds})**: {len(ids2)} pages; heron recall@.5 {tp / max(1, ng):.3f}, misses {nm}")
        P("| second detector | where | added / 100 pages | misses recovered @.5 | inc. recall | precision of added | false tables / 100 pages | heron+added recall |\n|---|---|---|---|---|---|---|---|")
        arm = {}
        for name, s_, tbl in (("PP-DocLayout_plus-L", src["pp"], ("table",)), ("RF-DETR-Large", src["rf"], ("Table",)), ("YOLOX-L", yx, ("Table",))):
            if not s_ or not all(i in s_ for i in ids2): continue
            for tau in (0.5, 0.7):
                for where in ("uncovered regions", "pages without a heron table"):
                    na = rec = good = false = 0
                    for i in ids2:
                        tb = [d[2:6] for d in s_[i]["dets"] if d[0] in tbl and d[1] >= tau]
                        a = [b for b in tb if cover(b, ht[i]) < 0.3] if where == "uncovered regions" else (tb if not ht[i] else [])
                        na += len(a); g = gt[i].get("Table", [])
                        M = iou_mat(a, miss[i]); used = set()
                        for k in range(len(a)):
                            if len(miss[i]):
                                j = int(np.argmax(M[k]))
                                if M[k, j] >= 0.5 and j not in used: used.add(j); good += 1
                        rec += len(used); false += sum(1 for b in a if all(iou(b, x) < 0.5 for x in g))
                    arm[f"{name}|{tau}|{where}"] = dict(rec=rec, prec=good / max(1, na), false100=100 * false / len(ids2))
                    P(f"| {name} >= {tau} | {where} | {100 * na / len(ids2):.1f} | {rec} of {nm} | {rec / max(1, nm):.3f} | {good / max(1, na):.3f} | {100 * false / len(ids2):.1f} | {(tp + rec) / max(1, ng):.3f} |")
        summ[ds]["arm"] = arm; summ[ds]["misses"] = nm
    txt = "\n".join(out); print(txt)
    os.makedirs(R4OUT, exist_ok=True)
    open(os.path.join(R4OUT, "pp.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(summ, open(os.path.join(R4OUT, "pp_summary.json"), "w"), default=str)


if __name__ == "__main__":
    main()
