import os, sys, json, glob
import numpy as np

ODB = os.environ.get("ODB_DIR", os.path.expanduser("~/odb"))
SECOND = ["pico", "tatrdet", "yolox_tiny", "yolox_l"]
TAU_H = 0.5


def load_dets(pattern):
    d = {}
    for p in glob.glob(pattern, recursive=True):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); d[r["id"]] = r
    return d


def tables(r, tau, label="Table"):
    return [(t[1], t[2:6]) for t in r["dets"] if t[0] == label and t[1] >= tau]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def match(gt, pred, thr):
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)), reverse=True)
    ug, up = set(), set()
    for v, gi, pi in pairs:
        if v < thr: break
        if gi in ug or pi in up: continue
        ug.add(gi); up.add(pi)
    return ug, up


def dedupe(primary, extra, thr=0.5):
    out = list(primary)
    for b in extra:
        if all(iou(b, p) < thr for p in out): out.append(b)
    return out


def wbf(h, s, W, H):
    from ensemble_boxes import weighted_boxes_fusion
    bl, sl, ll = [], [], []
    for lst in (h, s):
        bl.append([[max(0, min(1, b[0] / W)), max(0, min(1, b[1] / H)), max(0, min(1, b[2] / W)), max(0, min(1, b[3] / H))] for _, b in lst] or np.zeros((0, 4)))
        sl.append([sc for sc, _ in lst] or []); ll.append([0] * len(lst))
    if not any(len(x) for x in sl): return []
    b, sc, _ = weighted_boxes_fusion(bl, sl, ll, iou_thr=0.55, skip_box_thr=0.3, conf_type="max")
    return [[x[0] * W, x[1] * H, x[2] * W, x[3] * H] for x, v in zip(b, sc) if v >= 0.5]


def odb_gt():
    d = json.load(open(os.path.join(ODB, "OmniDocBench.json"), encoding="utf-8"))
    out = {}
    for rec in d:
        pi = rec["page_info"]
        if pi["page_attribute"].get("language") != "english": continue
        bs = []
        for e in rec["layout_dets"]:
            if e.get("ignore") or e["category_type"] not in ("table", "table_mask"): continue
            p = e["poly"]; xs, ys = p[0::2], p[1::2]
            bs.append([min(xs), min(ys), max(xs), max(ys)])
        out["odb:" + os.path.basename(pi["image_path"])] = bs
    return out


def main():
    root = sys.argv[1]
    heron = load_dets(os.path.join(root, "order", "**", "layout-heron_int8-*.jsonl"))
    gt = {}
    for p in glob.glob(os.path.join(root, "order", "**", "dln-gt-*.jsonl"), recursive=True):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); gt[r["id"]] = [g[1:5] for g in r["gt"] if g[0] == "Table"]
    if os.path.exists(os.path.join(ODB, "OmniDocBench.json")): gt.update(odb_gt())
    sec = {m: load_dets(os.path.join(root, "r3det", "**", f"det-{m}-*.jsonl")) for m in SECOND}
    out = []; P = out.append
    P("## Test 4: tables missed at layout detection (heron INT8 + a second detector)")
    for ds in ("dln", "odb"):
        ids = [i for i in heron if heron[i]["ds"] == ds and i in gt and all(i in sec[m] for m in SECOND if sec[m])]
        ngt = sum(len(gt[i]) for i in ids)
        P(f"\n### {ds}: {len(ids)} pages, {ngt} ground-truth tables")
        P("| method | recall@.5 | precision@.5 | recall@.7 | precision@.7 | heron misses recovered @.5 | false tables / page @.5 | time / page ms |\n|---|---|---|---|---|---|---|---|")
        hmiss = {}
        for i in ids:
            ug, _ = match(gt[i], [b for _, b in tables(heron[i], TAU_H)], 0.5)
            hmiss[i] = set(range(len(gt[i]))) - ug
        nmiss = sum(len(v) for v in hmiss.values())

        def evaluate(name, fn, tsec):
            res = {}
            for thr in (0.5, 0.7):
                tp = fp = rec = 0; npred = 0
                for i in ids:
                    pred = fn(i)
                    ug, up = match(gt[i], pred, thr)
                    tp += len(ug); fp += len(pred) - len(up); npred += len(pred)
                    if thr == 0.5: rec += len(ug & hmiss[i])
                res[thr] = (tp / max(1, ngt), tp / max(1, npred), fp, rec)
            ms = 1000 * sum(tsec(i) for i in ids) / max(1, len(ids))
            P(f"| {name} | {res[0.5][0]:.3f} | {res[0.5][1]:.3f} | {res[0.7][0]:.3f} | {res[0.7][1]:.3f} | {res[0.5][3]} of {nmiss} | {res[0.5][2] / max(1, len(ids)):.3f} | {ms:.0f} |")
        hb = lambda i: [b for _, b in tables(heron[i], TAU_H)]
        evaluate("heron INT8 alone (score >= 0.5)", hb, lambda i: heron[i]["sec"])
        for m in SECOND:
            if not sec[m]: continue
            lab = "Table"
            for tau in (0.5, 0.7):
                sb = lambda i, tau=tau: [b for _, b in tables(sec[m][i], tau, lab)]
                evaluate(f"{m} alone (>= {tau})", sb, lambda i: sec[m][i]["sec"])
                evaluate(f"heron + {m} union (>= {tau})", lambda i, sb=sb: dedupe(hb(i), sb(i)), lambda i: heron[i]["sec"] + sec[m][i]["sec"])
                evaluate(f"heron + {m} only where heron found no table (>= {tau})", lambda i, sb=sb: hb(i) if hb(i) else sb(i),
                         lambda i: heron[i]["sec"] + (0 if hb(i) else sec[m][i]["sec"]))
            evaluate(f"heron + {m} WBF (skip<0.3, fused>=0.5, max conf)", lambda i: wbf(tables(heron[i], 0.3), tables(sec[m][i], 0.3, lab), heron[i]["W"], heron[i]["H"]),
                     lambda i: heron[i]["sec"] + sec[m][i]["sec"])
    txt = "\n".join(out); print(txt)
    os.makedirs(os.path.join(root, "r3"), exist_ok=True)
    open(os.path.join(root, "r3", "detect.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
