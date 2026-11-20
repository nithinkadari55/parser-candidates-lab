import os, sys, json, glob, re, random, statistics
import numpy as np
import paths
from paths import SETS, OUT
from prep_sets import parse_html, place
from score import gt_chunks, pred_html

SETS_ = ["ptn", "ftn", "pt1m", "pt2", "icdar", "scitsr", "tabrec", "rot"]
A, B = "tableformer", "tatr16"
BUCKETS = ["simple", "medium", "complex", "edge"]


def load(root, m, s, scored):
    d = {}
    for p in glob.glob(os.path.join(root, "**", f"{m}_{s}_s*.jsonl"), recursive=True):
        if p.endswith(".scored.jsonl") != scored: continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); d[r["id"]] = r
    return d


def doc_key(s, gt):
    f = gt.get("filename") or gt["id"]
    if s == "rot": f = gt.get("filename") or gt.get("source", "")
    m = re.search(r"(PMC\d+)", f)
    if m: return m.group(1)
    if s == "ftn" or ".page_" in f: return f.split(".page_")[0]
    if s == "scitsr": return re.split(r"v\d", f)[0]
    if s == "icdar": return f
    return f


def val(r):
    if r is None: return None
    return r["teds_s"] if r.get("teds_s") is not None else r.get("grits_top")


def area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b): return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def cv(v):
    v = [x for x in v if x > 0]
    return float(np.std(v) / np.mean(v)) if len(v) > 1 else 0.0


def feats(cells, words):
    f = {}
    cells = [c for c in cells if c.get("bbox") and area(c["bbox"]) > 0]
    f["no_output"] = float(not cells)
    if not cells:
        for k in ("n_cells", "rows", "cols", "spans", "empty", "orphan", "split", "cam_acc", "cam_score", "tatr_fit", "row_align", "col_align", "multiline", "cell_cv", "rowh_cv", "colw_cv", "wpc"):
            f[k] = 0.0
        f["orphan"] = 1.0
        return f
    R = max(c["r1"] for c in cells); C = max(c["c1"] for c in cells)
    f["n_cells"], f["rows"], f["cols"] = len(cells), R, C
    f["spans"] = sum(1 for c in cells if c["r1"] - c["r0"] > 1 or c["c1"] - c["c0"] > 1) / len(cells)
    assign = {}; errs = []; best = []; orphan = split = 0
    for wi, w in enumerate(words):
        wb = w["bbox"]; wa = area(wb)
        if wa <= 0: continue
        ov = [(inter(wb, c["bbox"]) / wa, k) for k, c in enumerate(cells)]
        ov.sort(reverse=True)
        best.append(ov[0][0])
        if ov[0][0] <= 0: orphan += 1; errs.append(1.0); continue
        if len(ov) > 1 and ov[1][0] >= 0.2: split += 1
        assign.setdefault(ov[0][1], []).append(wi)
        errs.append(1.0 - ov[0][0])
    n = max(1, len(best))
    f["orphan"] = orphan / n; f["split"] = split / n
    f["cam_acc"] = 100.0 * (1 - (sum(errs) / len(errs) if errs else 1.0))
    f["empty"] = 1.0 - len(assign) / len(cells)
    f["cam_score"] = f["cam_acc"] / 100.0 * (1 - f["empty"])
    f["tatr_fit"] = (sum(best) / len(best) + min(best)) / 2 if best else 0.0
    hs = [words[i]["bbox"][3] - words[i]["bbox"][1] for i in range(len(words)) if area(words[i]["bbox"]) > 0]
    mh = float(np.median(hs)) if hs else 1.0
    rows = {}; cols = {}; ml = 0
    for k, wis in assign.items():
        c = cells[k]
        if c["r1"] - c["r0"] == 1: rows.setdefault(c["r0"], []).extend(wis)
        if c["c1"] - c["c0"] == 1: cols.setdefault(c["c0"], []).extend(wis)
        ys = sorted((words[i]["bbox"][1] + words[i]["bbox"][3]) / 2 for i in wis)
        if ys and ys[-1] - ys[0] > 0.8 * mh: ml += 1
    ra = [np.std([words[i]["bbox"][3] for i in wis]) / mh for wis in rows.values() if len(wis) > 1]
    ca = []
    for wis in cols.values():
        if len(wis) < 2: continue
        l = np.std([words[i]["bbox"][0] for i in wis]); r = np.std([words[i]["bbox"][2] for i in wis]); m = np.std([(words[i]["bbox"][0] + words[i]["bbox"][2]) / 2 for i in wis])
        ca.append(min(l, r, m) / mh)
    f["row_align"] = float(np.median(ra)) if ra else 0.0
    f["col_align"] = float(np.median(ca)) if ca else 0.0
    f["multiline"] = ml / len(cells)
    f["cell_cv"] = cv([area(c["bbox"]) for c in cells])
    rh = {}; cw = {}
    for c in cells:
        if c["r1"] - c["r0"] == 1: rh.setdefault(c["r0"], []).append(c["bbox"][3] - c["bbox"][1])
        if c["c1"] - c["c0"] == 1: cw.setdefault(c["c0"], []).append(c["bbox"][2] - c["bbox"][0])
    f["rowh_cv"] = cv([np.median(v) for v in rh.values()]); f["colw_cv"] = cv([np.median(v) for v in cw.values()])
    f["wpc"] = len(best) / len(cells)
    return f


def agreement(ca, cb):
    import grits
    if not ca or not cb: return 0.0
    Ra, Ca = max(c["r1"] for c in ca), max(c["c1"] for c in ca); Rb, Cb = max(c["r1"] for c in cb), max(c["c1"] for c in cb)
    if Ra * Ca > 1200 or Rb * Cb > 1200: return float(Ra == Rb and Ca == Cb)
    try:
        g = grits.grits_from_html(pred_html(Ra, Ca, ca, {}, True)[12:-14], pred_html(Rb, Cb, cb, {}, True)[12:-14])
        return float(g["grits_top"])
    except Exception:
        return 0.0


def paired(x, y):
    from scipy.stats import wilcoxon
    d = [a - b for a, b in zip(x, y)]
    if len(d) < 10: return None
    rng = random.Random(0); bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    try: p = wilcoxon(x, y, zero_method="zsplit").pvalue
    except ValueError: p = 1.0
    return dict(diff=sum(d) / len(d), lo=bs[50], hi=bs[1949], p=p, wins=sum(v > 1e-9 for v in d), losses=sum(v < -1e-9 for v in d))


def main():
    root = sys.argv[1]
    rows = []
    for s in SETS_:
        pa, pb = load(root, A, s, False), load(root, B, s, False)
        sa, sb = load(root, A, s, True), load(root, B, s, True)
        lim = int(os.environ.get("TB_LIMIT") or 0)
        for tid in sorted(set(pa) & set(pb) & set(sa) & set(sb))[: (lim * 20 if lim else None)]:
            gt = json.load(open(os.path.join(SETS, s, tid + ".json"), encoding="utf-8"))
            words = [w for w in gt_chunks(gt)]
            fa, fb = feats(pa[tid].get("cells") or [], words), feats(pb[tid].get("cells") or [], words)
            x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
            x.update({"d_" + k: fb[k] - fa[k] for k in fa})
            x["agree"] = agreement(pa[tid].get("cells") or [], pb[tid].get("cells") or [])
            x["n_words"] = len(words)
            rows.append(dict(id=tid, set=s, bucket=sa[tid]["bucket"], doc=s + ":" + doc_key(s, gt), va=val(sa[tid]), vb=val(sb[tid]), x=x))
    rows = [r for r in rows if r["va"] is not None and r["vb"] is not None]
    print("tables", len(rows), flush=True)
    keys = sorted(rows[0]["x"])
    X = np.array([[r["x"][k] for k in keys] for r in rows], float)
    y = np.array([1 if r["vb"] > r["va"] + 1e-9 else 0 for r in rows])
    groups = np.array([r["doc"] for r in rows])
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import xgboost as xgb
    p_lr = np.zeros(len(rows)); p_xgb = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], y[tr]); p_lr[te] = lr.predict_proba(X[te])[:, 1]
        xg = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=0).fit(X[tr], y[tr])
        p_xgb[te] = xg.predict_proba(X[te])[:, 1]
    imp = sorted(zip(xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=4, random_state=0).fit(X, y).feature_importances_, keys), reverse=True)[:15]
    for i, r in enumerate(rows):
        x = r["x"]
        r["sel_rule_cam"] = r["vb"] if x["b_cam_score"] > x["a_cam_score"] + 1e-9 else r["va"]
        r["sel_rule_fit"] = r["vb"] if x["b_tatr_fit"] > x["a_tatr_fit"] + 1e-9 else r["va"]
        r["sel_lr"] = r["vb"] if p_lr[i] > 0.5 else r["va"]
        r["sel_xgb"] = r["vb"] if p_xgb[i] > 0.5 else r["va"]
        r["oracle"] = max(r["va"], r["vb"])
        r["pick_xgb"] = int(p_xgb[i] > 0.5); r["pick_lr"] = int(p_lr[i] > 0.5)
        r["pick_cam"] = int(x["b_cam_score"] > x["a_cam_score"] + 1e-9); r["pick_fit"] = int(x["b_tatr_fit"] > x["a_tatr_fit"] + 1e-9)
    out = []; P = out.append
    cols = [("always TableFormer", "va"), ("always TATR fp16", "vb"), ("rule: Camelot fit", "sel_rule_cam"), ("rule: TATR token fit", "sel_rule_fit"),
            ("logistic regression", "sel_lr"), ("XGBoost", "sel_xgb"), ("oracle", "oracle")]
    P("## Test 3: per-table selector, TableFormer vs TATR fp16 (TEDS-Struct per table; GriTS-Top where TEDS is capped)")
    P(f"Tables: {len(rows)}; documents (CV groups): {len(set(groups))}; 5-fold GroupKFold by document. TATR better on {int(y.sum())} tables, TableFormer better or tied on {len(y) - int(y.sum())}.")
    P("| set/bucket | n | " + " | ".join(c for c, _ in cols) + " | XGB picks TATR % | XGB pick accuracy % |\n|" + "---|" * (len(cols) + 4))
    groups_ = [("all", None, None)] + [(s, s, None) for s in SETS_] + [(f"main/{b}", "main", b) for b in BUCKETS] + [(f"{s}/{b}", s, b) for s in SETS_ for b in BUCKETS]
    for name, s, b in groups_:
        sub = [r for r in rows if (s is None or (r["set"] in ("ptn", "ftn", "pt1m") if s == "main" else r["set"] == s)) and (b is None or r["bucket"] == b)]
        if not sub: continue
        acc = 100 * sum(1 for r in sub if (r["pick_xgb"] == 1) == (r["vb"] > r["va"] + 1e-9) or abs(r["vb"] - r["va"]) <= 1e-9) / len(sub)
        P(f"| {name} | {len(sub)} | " + " | ".join(f"{sum(r[k] for r in sub) / len(sub):.4f}" for _, k in cols) + f" | {100 * sum(r['pick_xgb'] for r in sub) / len(sub):.1f} | {acc:.1f} |")
    P("\n### Paired tests (Wilcoxon signed-rank, bootstrap 95% CI of the mean difference)")
    for name, s in [("all", None), ("main", "main")] + [(x, x) for x in SETS_]:
        sub = [r for r in rows if s is None or (r["set"] in ("ptn", "ftn", "pt1m") if s == "main" else r["set"] == s)]
        for sel in ("sel_xgb", "sel_lr", "sel_rule_cam", "sel_rule_fit"):
            for base, bn in (("va", "TableFormer"), ("vb", "TATR fp16")):
                t = paired([r[sel] for r in sub], [r[base] for r in sub])
                if t: P(f"- {sel} vs {bn} [{name}]: n={len(sub)} diff={t['diff']:+.4f} CI [{t['lo']:+.4f}, {t['hi']:+.4f}] p={t['p']:.2g} wins/losses {t['wins']}/{t['losses']}")
    P("\n### XGBoost feature importance (top 15, model fitted on all tables)")
    P(", ".join(f"{k} {v:.3f}" for v, k in imp))
    txt = "\n".join(out); print(txt)
    os.makedirs(os.path.join(OUT, "r3"), exist_ok=True)
    open(os.path.join(OUT, "r3", "select.md"), "w", encoding="utf-8").write(txt + "\n")
    with open(os.path.join(OUT, "r3", "select_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if k != "x"}) + "\n")


if __name__ == "__main__":
    main()
