"""Layout evaluation: COCO AP per unified class (pycocotools), table recall, heading confusion, speed/RAM.
usage: eval_layout.py ARTIFACT_DIR OUT.md OUT.json"""
import collections, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import models

A, OUT, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = ["heron", "heron_int8", "egret", "ppl", "ppv3"]
THR = 0.5
lines, res = [], {}


def P(s=""):
    lines.append(s)
    print(s)


def jl(pattern):
    for f in sorted(glob.glob(os.path.join(A, "**", pattern), recursive=True)):
        for l in open(f, encoding="utf-8"):
            try:
                yield json.loads(l)
            except ValueError:
                pass


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def ioa(d, g):
    ix = max(0, min(d[2], g[2]) - max(d[0], g[0]))
    iy = max(0, min(d[3], g[3]) - max(d[1], g[1]))
    a = (d[2] - d[0]) * (d[3] - d[1])
    return ix * iy / a if a > 0 else 0.0


# ------------------------------------------------------------------ ground truth
ODB_MAP = {"title": "Section-header", "text_block": "Text", "reference": "Text", "code_txt": "Text", "equation_explanation": "Text",
           "figure": "Picture", "table": "Table", "figure_caption": "Caption", "table_caption": "Caption", "code_txt_caption": "Caption",
           "figure_footnote": "Footnote", "table_footnote": "Footnote", "page_footnote": "Footnote", "equation_isolated": "Formula",
           "header": "Page-header", "footer": "Page-footer", "page_number": "Page-footer"}


def gt_dln():
    g = {}
    for r in jl("dln-gt-*.jsonl"):
        g[r["id"]] = dict(W=r["W"], H=r["H"], cat=r.get("doc_category"), gt=[("Text" if c == "List-item" else c, x0, y0, x1, y1) for c, x0, y0, x1, y1 in r["gt"]], ign=[])
    return g


def gt_odb():
    import datasets
    d = json.load(open(os.path.join(datasets.ODB, "OmniDocBench.json"), encoding="utf-8"))
    g = {}
    for rec in d:
        info = rec["page_info"]
        if info["page_attribute"].get("language") != "english":
            continue
        pid = "odb:" + os.path.basename(info["image_path"])
        gt, ign = [], []
        for e in rec["layout_dets"]:
            b = datasets.box(e)
            c = ODB_MAP.get(e["category_type"])
            if c and not e.get("ignore"):
                gt.append((c,) + tuple(b))
            elif e["category_type"] not in ("list_group", "equation_semantic"):
                ign.append(b)
        g[pid] = dict(W=info["width"], H=info["height"], cat=info["page_attribute"].get("data_source"), gt=gt, ign=ign)
    return g


def load_dets(model, ds):
    out = {}
    if model == "dlrs":
        for r in jl("dlrs-*.jsonl"):
            if r.get("ds") != ds or "items" not in r:
                continue
            ds_ = []
            for it in r["items"]:
                u = models.unify("dlrs", it["label"], it["box"][1], it["box"][3], r["H"])
                if u:
                    ds_.append((u, 1.0) + tuple(it["box"]))
            out[r["id"]] = (ds_, r.get("sec"))
        return out
    for r in jl("layout-%s-*.jsonl" % model):
        if r.get("ds") != ds:
            continue
        ds_ = []
        for d in r["dets"]:
            u = models.unify(model, d[0], d[3], d[5], r["H"])
            if u:
                ds_.append((u, d[1]) + tuple(d[2:6]))
        out[r["id"]] = (ds_, r.get("sec"))
    return out


def coco_eval(G, D, merge_title):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    cats = [c for c in models.UNI if not (merge_title and c == "Title")]
    cid = {c: i + 1 for i, c in enumerate(cats)}
    fix = (lambda c: "Section-header" if merge_title and c == "Title" else c)
    ids = sorted(set(G) & set(D))
    if not ids:
        return None
    imgs, anns, dts = [], [], []
    aid = 1
    for n, pid in enumerate(ids, 1):
        g = G[pid]
        imgs.append(dict(id=n, width=g["W"], height=g["H"]))
        for c, x0, y0, x1, y1 in g["gt"]:
            c = fix(c)
            anns.append(dict(id=aid, image_id=n, category_id=cid[c], bbox=[x0, y0, x1 - x0, y1 - y0], area=(x1 - x0) * (y1 - y0), iscrowd=0))
            aid += 1
        for c, s, x0, y0, x1, y1 in D[pid][0]:
            c = fix(c)
            b = (x0, y0, x1, y1)
            if any(ioa(b, ig) > 0.5 for ig in g["ign"]):
                continue
            dts.append(dict(image_id=n, category_id=cid[c], bbox=[x0, y0, x1 - x0, y1 - y0], score=s))
    gt = COCO()
    gt.dataset = dict(info={}, licenses=[], images=imgs, annotations=anns, categories=[dict(id=i, name=c) for c, i in cid.items()])
    gt.createIndex()
    if not dts:
        return None
    dt = gt.loadRes(dts)
    E = COCOeval(gt, dt, "bbox")
    E.params.maxDets = [1, 10, 100]
    E.evaluate()
    E.accumulate()
    pr = E.eval["precision"]
    out = {}
    for c, i in cid.items():
        p = pr[:, :, i - 1, 0, 2]
        ap = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
        p50 = pr[0, :, i - 1, 0, 2]
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan")
        out[c] = (ap, ap50)
    valid = [v for v in out.values() if v[0] == v[0]]
    out["mAP"] = (sum(v[0] for v in valid) / len(valid), sum(v[1] for v in valid) / len(valid))
    out["pages"] = len(ids)
    return out


def table_recall(G, D, thr):
    tot = {0.5: 0, 0.7: 0}
    hit = {0.5: 0, 0.7: 0}
    npred = mpred = 0
    pages = 0
    for pid, g in G.items():
        if pid not in D:
            continue
        gts = [b[1:] for b in g["gt"] if b[0] == "Table"]
        preds = [d[2:] for d in D[pid][0] if d[0] == "Table" and d[1] >= thr]
        if gts:
            pages += 1
        for t in gts:
            best = max((iou(t, p) for p in preds), default=0)
            for k in tot:
                tot[k] += 1
                hit[k] += best >= k
        npred += len(preds)
        mpred += sum(1 for p in preds if max((iou(p, t) for t in gts), default=0) >= 0.5)
    return dict(pages=pages, tables=tot[0.5], recall50=hit[0.5] / tot[0.5] if tot[0.5] else float("nan"),
                recall70=hit[0.7] / tot[0.7] if tot[0.7] else float("nan"), precision50=mpred / npred if npred else float("nan"))


def confusion(G, D, thr):
    src = collections.Counter()
    as_head = collections.Counter()
    heads = 0
    heads_wrong = collections.Counter()
    for pid, g in G.items():
        if pid not in D:
            continue
        preds = [d for d in D[pid][0] if d[1] >= thr]
        for c, *b in g["gt"]:
            if c in ("Page-header", "Page-footer", "Caption", "Footnote"):
                src[c] += 1
                best, bl = 0, None
                for d in preds:
                    v = iou(b, d[2:])
                    if v > best:
                        best, bl = v, d[0]
                if best >= 0.5 and bl in ("Section-header", "Title"):
                    as_head[c] += 1
        for d in preds:
            if d[0] in ("Section-header", "Title"):
                heads += 1
                best, bl = 0, None
                for c, *b in g["gt"]:
                    v = iou(b, d[2:])
                    if v > best:
                        best, bl = v, c
                if best >= 0.5 and bl in ("Page-header", "Page-footer", "Caption", "Footnote"):
                    heads_wrong[bl] += 1
    return dict(gt=dict(src), labelled_heading=dict(as_head), rate={c: as_head[c] / src[c] for c in src if src[c]},
                pred_headings=heads, wrong_heading_share=sum(heads_wrong.values()) / heads if heads else float("nan"), wrong_by=dict(heads_wrong))


def fmt(v):
    return "-" if v is None or v != v else f"{100 * v:.1f}"


def main():
    G = {"dln": gt_dln(), "odb": gt_odb()}
    P("## Part 1: layout models")
    P("Unified classes: DocLayNet's 11 with List-item folded into Text. On OmniDocBench Title is folded into Section-header "
      "(its `title` covers both) and masked/abandoned regions are ignore zones. Mapping: see order/layout/models.py.")
    for ds in ("dln", "odb"):
        g = G[ds]
        P(f"\n### {'DocLayNet test' if ds == 'dln' else 'OmniDocBench English'} ({len(g)} pages with GT)")
        rows = {}
        for m in MODELS:
            D = load_dets(m, ds)
            if not D:
                continue
            ev = coco_eval(g, D, merge_title=(ds == "odb"))
            if not ev:
                continue
            tr = table_recall(g, D, THR)
            cf = confusion(g, D, THR)
            rows[m] = (ev, tr, cf)
            res.setdefault(ds, {})[m] = dict(ap=ev, table=tr, confusion=cf)
        if not rows:
            P("no results")
            continue
        cl = [c for c in models.UNI if not (ds == "odb" and c == "Title")]
        P("\nAP@[.5:.95] per class (COCO, maxDets 100), in %; last column AP50 mean")
        P("| model | pages | mAP | " + " | ".join(cl) + " | mAP50 |")
        P("|---|---|---|" + "---|" * len(cl) + "---|")
        for m, (ev, tr, cf) in rows.items():
            P(f"| {m} | {ev['pages']} | {fmt(ev['mAP'][0])} | " + " | ".join(fmt(ev[c][0]) for c in cl) + f" | {fmt(ev['mAP'][1])} |")
        P(f"\nTables (pages with a GT table; predictions with score >= {THR})")
        P("| model | pages | GT tables | recall IoU.5 | recall IoU.7 | precision IoU.5 |")
        P("|---|---|---|---|---|---|")
        for m, (ev, tr, cf) in rows.items():
            P(f"| {m} | {tr['pages']} | {tr['tables']} | {fmt(tr['recall50'])} | {fmt(tr['recall70'])} | {fmt(tr['precision50'])} |")
        P("\nHeading confusion: % of GT page-header / page-footer / caption / footnote boxes whose best prediction (IoU>=.5) is a "
          "heading; and % of predicted headings that are really one of those")
        P("| model | page-header->heading | page-footer->heading | caption->heading | footnote->heading | predicted headings | wrong-heading share |")
        P("|---|---|---|---|---|---|---|")
        for m, (ev, tr, cf) in rows.items():
            r = cf["rate"]
            P(f"| {m} | {fmt(r.get('Page-header'))} | {fmt(r.get('Page-footer'))} | {fmt(r.get('Caption'))} | {fmt(r.get('Footnote'))} | {cf['pred_headings']} | {fmt(cf['wrong_heading_share'])} |")
        if ds == "dln":
            P("\nmAP by DocLayNet document category")
            cats = sorted({v["cat"] for v in g.values() if v["cat"]})
            P("| model | " + " | ".join(cats) + " |")
            P("|---|" + "---|" * len(cats))
            for m in rows:
                D = load_dets(m, ds)
                cells = []
                for c in cats:
                    ev = coco_eval({k: v for k, v in g.items() if v["cat"] == c}, D, False)
                    cells.append(fmt(ev["mAP"][0]) if ev else "-")
                P(f"| {m} | " + " | ".join(cells) + " |")
        Dd = load_dets("dlrs", ds)
        if Dd:
            sub = {k: v for k, v in g.items() if k in Dd}
            P(f"\ndocling-rs 2.84.0 end to end (heron + docling postprocessing, no scores) vs raw models on the same {len(sub)} pages")
            P("| model | table recall .5 | table recall .7 | table precision .5 | page-header->heading | caption->heading | wrong-heading share |")
            P("|---|---|---|---|---|---|---|")
            for m in ["dlrs"] + MODELS:
                D = Dd if m == "dlrs" else load_dets(m, ds)
                if not D:
                    continue
                tr = table_recall(sub, D, THR)
                cf = confusion(sub, D, THR)
                res.setdefault(ds + "_dlrs_subset", {})[m] = dict(table=tr, confusion=cf)
                P(f"| {m} | {fmt(tr['recall50'])} | {fmt(tr['recall70'])} | {fmt(tr['precision50'])} | {fmt(cf['rate'].get('Page-header'))} | {fmt(cf['rate'].get('Caption'))} | {fmt(cf['wrong_heading_share'])} |")
    P("\n### Speed and memory (dedicated timing job, one process per model, ubuntu-latest 4 vCPU)")
    P("| model | file MB | load s | median s/page | p95 s/page | peak RSS MB |")
    P("|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(A, "**", "bench-*.json"), recursive=True)):
        b = json.load(open(f))
        res.setdefault("bench", {})[b["model"]] = b
        P(f"| {b['model']} | {b.get('file_mb', 0):.1f} | {b.get('load_s', 0):.2f} | {b.get('median_s', 0):.3f} | {b.get('p95_s', 0):.3f} | {b.get('peak_rss_mb', 0):.0f} |")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(res, open(OUTJ, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
