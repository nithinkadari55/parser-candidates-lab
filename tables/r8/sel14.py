"""Milestone 14: the harness's table selector retrained on the round-8 tables (run 45676430603).

Same features as the committed selector (round-3 feats() + agreement + n_words), same XGBoost settings, the words the
harness reads at run time for each source: docling-pdf word cells for text-layer pages, PP-OCRv6 small words for
scans. Rotated tables are left out (the harness keeps TATR there). 5-fold GroupKFold by document, out-of-fold picks.
A second candidate adds general table-level features (numbers per cell, right-edge alignment of numbers, empty rows,
empty cells inside number columns, rows with one filled cell), never a source or type flag; it is kept only if it
beats the base features on the dev split. The gate is checked on the test split:
  - at least as good overall as the committed selector (TEDS-Struct and exact shape),
  - exact shape not more than 2 points under TableFormer alone on any document type.
The final model is fitted on all training tables and exported for the Rust evaluator with an equality check.

usage: sel14.py PUB_ROOT PRIV_ROOT COMMITTED_SELECTOR OUT_DIR"""
import os, sys, json, gzip, glob, math, random, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import numpy as np
import score8 as S8
import r3_select as R3
from r5_sel import leaf_sum, PARAMS, export
from multiprocessing import Pool

SELECTOR_SHA = "c27d7295112ab514e46cf0a1595782468c932b06084d10ad3247d093a2ed34f1"
NUM = __import__("re").compile(r"^[\s\d.,:;%()$€£¥+\-−–—/*xX]+$")
EXTRA = ["num_share", "num_align", "empty_rows", "num_col_empty", "one_cell_rows"]
MIN_GAIN = 0.0003
META, PRIV = {}, {}


def base_doc(doc): return doc.split(":", 1)[1] if ":" in doc else doc


def area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b): return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def extra_feats(cells, words):
    """table-level features from one model's grid and the words (crop pixels); every value is a ratio"""
    cells = [c for c in cells if c.get("bbox") and area(c["bbox"]) > 0]
    f = {k: 0.0 for k in EXTRA}
    if not cells: return f
    R = max(c["r1"] for c in cells)
    hs = [w["bbox"][3] - w["bbox"][1] for w in words if area(w["bbox"]) > 0]
    mh = float(np.median(hs)) if hs else 1.0
    texts = collections.defaultdict(list); rights = collections.defaultdict(list)
    for w in words:
        wb = w["bbox"]; wa = area(wb)
        if wa <= 0: continue
        ov = [(inter(wb, c["bbox"]) / wa, k) for k, c in enumerate(cells)]
        best, k = max(ov)
        if best <= 0: continue
        texts[k].append(w["text"]); rights[k].append(wb[2])
    filled = [k for k in range(len(cells)) if texts.get(k)]
    num = {k for k in filled if all(NUM.match(t) for t in texts[k])}
    f["num_share"] = len(num) / max(1, len(filled))
    bycol = collections.defaultdict(list)
    for k, c in enumerate(cells):
        if c["c1"] - c["c0"] == 1: bycol[c["c0"]].append(k)
    al, nce, ncn = [], 0, 0
    for q, ks in bycol.items():
        nk = [k for k in ks if k in num]
        if len(nk) >= 2: al.append(float(np.std([max(rights[k]) for k in nk])) / mh)
        fk = [k for k in ks if k in filled]
        if fk and len(nk) >= 0.5 * len(fk):
            ncn += len(ks); nce += sum(1 for k in ks if k not in filled)
    f["num_align"] = float(np.median(al)) if al else 0.0
    f["num_col_empty"] = nce / ncn if ncn else 0.0
    rows_filled = collections.Counter()
    for k in filled:
        c = cells[k]
        if c["r1"] - c["r0"] == 1: rows_filled[c["r0"]] += 1
    f["empty_rows"] = sum(1 for r in range(R) if rows_filled[r] == 0) / R
    f["one_cell_rows"] = sum(1 for r in range(R) if rows_filled[r] == 1) / R
    return f


def _feat(tid):
    p = PRIV[tid]; words = [dict(text=w["text"], bbox=w["bbox"]) for w in p["words"]]
    ca, cb = p["cells"]["tf"], p["cells"]["tatr"]
    fa, fb = R3.feats(ca, words), R3.feats(cb, words)
    x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
    x.update({"d_" + k: fb[k] - fa[k] for k in fa})
    x["agree"] = R3.agreement(ca, cb); x["n_words"] = len(words)
    ea, eb = extra_feats(ca, words), extra_feats(cb, words)
    e = {"a_" + k: v for k, v in ea.items()}; e.update({"b_" + k: v for k, v in eb.items()})
    e.update({"d_" + k: eb[k] - ea[k] for k in ea})
    return tid, x, e


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    pub_root, priv_root, committed, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    for p in glob.glob(os.path.join(pub_root, "**", "meta.jsonl"), recursive=True):
        for r in jl(p): META[r["id"]] = r
    for p in glob.glob(os.path.join(priv_root, "**", "priv.jsonl.gz"), recursive=True):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line); PRIV[r["id"]] = r
    assert hashlib.sha256(open(committed, "rb").read()).hexdigest() == SELECTOR_SHA
    forest = json.load(open(committed)); keys = forest["features"]
    rows = []
    for tid, m in META.items():
        sc = m.get("scores") or {}
        if not m["found"] or m.get("err") or tid not in PRIV or "tf" not in sc or "tatr" not in sc: continue
        if S8.value(sc["tf"]) is None or S8.value(sc["tatr"]) is None: continue
        rows.append(dict(id=tid, set=m["set"], type=m["type"], split=m["split"], bdoc=base_doc(m["doc"]), turn=m.get("turn") or 0,
                         empty=not PRIV[tid]["cells"]["tatr"],
                         v=dict(tf=S8.value(sc["tf"]), tatr=S8.value(sc["tatr"])),
                         s=dict(tf=float(bool(sc["tf"].get("rc_exact"))), tatr=float(bool(sc["tatr"].get("rc_exact"))))))
    print("rows", len(rows), flush=True)
    with Pool(4) as pool:
        res = {tid: (x, e) for tid, x, e in pool.imap_unordered(_feat, [r["id"] for r in rows], chunksize=8)}
    for r in rows:
        r["x"], r["e"] = res[r["id"]]
        m = float(np.float32(forest["base_margin"]) + leaf_sum(forest["trees"], np.array([r["x"][k] for k in keys], np.float32)))
        r["p_committed"] = 1 / (1 + math.exp(-m))
    tr = [r for r in rows if r["turn"] == 0 and not r["empty"]]
    ekeys = sorted(keys + ["a_" + k for k in EXTRA] + ["b_" + k for k in EXTRA] + ["d_" + k for k in EXTRA])
    import xgboost as XG
    from sklearn.model_selection import GroupKFold
    y = np.array([1 if r["v"]["tatr"] > r["v"]["tf"] + 1e-9 else 0 for r in tr]); g = np.array([r["bdoc"] for r in tr])
    mats = {"base": (keys, np.array([[r["x"][k] for k in keys] for r in tr], float)),
            "extended": (ekeys, np.array([[{**r["x"], **r["e"]}[k] for k in ekeys] for r in tr], float))}
    folds = list(GroupKFold(n_splits=5).split(mats["base"][1], y, g))
    for name, (ks, X) in mats.items():
        p = np.zeros(len(tr))
        for a, b in folds: p[b] = XG.XGBClassifier(**PARAMS).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        for r, q in zip(tr, p): r["p_" + name] = float(q)
    for r in rows:
        for name, key in (("committed", "p_committed"), ("base", "p_base"), ("extended", "p_extended")):
            arm = "tf" if r["empty"] else ("tatr" if r["turn"] else ("tatr" if r.get(key, 0.0) > 0.5 else "tf"))
            r["v"][name] = r["v"][arm]; r["s"][name] = r["s"][arm]
    arms = ["tf", "tatr", "committed", "base", "extended"]

    def table(sub):
        return {a: dict(teds_s=mean([r["v"][a] for r in sub]), shape=mean([r["s"][a] for r in sub])) for a in arms}

    def by_type(split):
        sub = [r for r in rows if r["split"] == split]
        out = {"overall": dict(n=len(sub), **table(sub))}
        for t in TYPES:
            st = [r for r in sub if r["type"] == t]
            if st: out[t] = dict(n=len(st), **table(st))
        return out
    dev, test = by_type("dev"), by_type("test")

    def gate(res, name, ref="committed"):
        ok_overall = res["overall"][name]["teds_s"] >= res["overall"][ref]["teds_s"] and res["overall"][name]["shape"] >= res["overall"][ref]["shape"]
        worst = {t: 100 * (res[t][name]["shape"] - res[t]["tf"]["shape"]) for t in TYPES if t in res}
        return ok_overall and all(v >= -2.0 for v in worst.values()), worst
    # the feature set, on the dev split only
    gb, _ = gate(dev, "base"); ge, _ = gate(dev, "extended")
    choice = "extended" if (ge and dev["overall"]["extended"]["teds_s"] >= dev["overall"]["base"]["teds_s"] + MIN_GAIN) or (ge and not gb) else "base"
    passed, worst = gate(test, choice)
    test_rows = [r for r in rows if r["split"] == "test"]
    paired = {}
    for a, b in ((choice, "committed"), (choice, "tf"), ("extended", "base")):
        for key, k2 in (("v", "teds_s"), ("s", "shape")):
            rr = [dict(doc=r["bdoc"], x=r[key][a], y=r[key][b]) for r in test_rows]
            c = cluster_boot(rr, "x", "y", group="doc"); p = wilcoxon_p([r["x"] for r in rr], [r["y"] for r in rr])
            paired[f"{a} - {b} {k2}"] = dict(diff=c[0], lo=c[1], hi=c[2], p=p)
    # final model on every training table, exported for the Rust evaluator
    ks, X = mats[choice]
    final = XG.XGBClassifier(**PARAMS).fit(X, y)
    model_json, margins, err = export(final, ks, X)
    model_json.update(words="docling-pdf 2.84.0 PdfPage::word_cells (text-layer pages); PP-OCRv6 small det + rec on the crop (scans)",
                      training="parser-candidates-lab tables/r8/sel14.py on tables-r8 run 45676430603: %d tables, %d documents" % (len(tr), len(set(g))),
                      tatr_rows_joined="none", split_row_features=False, extra_features=EXTRA if choice == "extended" else [])
    json.dump(model_json, open(os.path.join(out, "table_selector.json"), "w"), separators=(",", ":"))
    probs = final.predict_proba(X)[:, 1]
    rng = random.Random(14); idx = sorted(rng.sample(range(len(tr)), min(80, len(tr))))
    json.dump(dict(features=ks, rows=[dict(id=tr[i]["id"], x=[float(v) for v in X[i]], margin=float(margins[i]), prob=float(probs[i])) for i in idx]),
              open(os.path.join(out, "selector_fixture.json"), "w"))
    # feature fixture: tables of the CDLA-Permissive sets only (their words may be published)
    small = [r for r in tr if r["set"] in ("r4pt", "pt2", "ftnc") and 3 <= len(PRIV[r["id"]]["words"]) <= 60
             and 0 < len(PRIV[r["id"]]["cells"]["tf"]) <= 80 and 0 < len(PRIV[r["id"]]["cells"]["tatr"]) <= 80]
    rng.shuffle(small); pick = []; per = collections.Counter()
    for r in small:
        if per[r["set"]] < 4: pick.append(r); per[r["set"]] += 1
        if len(pick) >= 12: break
    strip = lambda cs: [dict(r0=c["r0"], r1=c["r1"], c0=c["c0"], c1=c["c1"], bbox=c["bbox"]) for c in cs]
    json.dump(dict(tables=[dict(id=r["id"], tableformer=strip(PRIV[r["id"]]["cells"]["tf"]), tatr=strip(PRIV[r["id"]]["cells"]["tatr"]),
                                words=[dict(text=w["text"], bbox=w["bbox"]) for w in PRIV[r["id"]]["words"]], features={**r["x"], **r["e"]}) for r in pick]),
              open(os.path.join(out, "features_fixture.json"), "w"))
    summary = dict(rows=len(rows), training=len(tr), docs=len(set(g)), choice=choice, gate_passed=passed, worst_type_shape_vs_tf_pts=worst,
                   dev=dev, test=test, paired=paired, export_max_margin_diff=err, extra=EXTRA,
                   picks_tatr_test={a: sum(1 for r in test_rows if not r["empty"] and (r["turn"] or r.get(k, 0) > 0.5)) / len(test_rows)
                                    for a, k in (("committed", "p_committed"), ("base", "p_base"), ("extended", "p_extended"))})
    json.dump(summary, open(os.path.join(out, "sel14_summary.json"), "w"), indent=1)
    names = dict(tf="TableFormer", tatr="TATR", committed="committed selector", base="retrained (same features)", extended="retrained + table features")
    md = ["# Milestone 14: table selector retrained on the round-8 tables", "",
          f"Training tables: {len(tr)} (unrotated, TATR output present), documents {len(set(g))}; 5-fold GroupKFold by document, out-of-fold picks.",
          f"Feature set chosen on the dev split: **{choice}**. Gate on the test split (overall >= committed on TEDS-Struct and shape; no type's shape > 2 points under TableFormer): **{'pass' if passed else 'FAIL'}**.", ""]
    for split, res in (("test", test), ("dev", dev)):
        md += [f"## {split} split: TEDS-Struct / exact shape %", "", "| type | n | " + " | ".join(names[a] for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
        for t, v in res.items():
            md.append(f"| {t} | {v['n']} | " + " | ".join(f"{v[a]['teds_s']:.4f} / {100 * v[a]['shape']:.1f}" for a in arms) + " |")
        md.append("")
    md += ["## Paired (test split, 95% CI by document, Wilcoxon p)", ""] + [f"- {k}: {100 * v['diff']:+.2f} [{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] p={v['p']:.2g}" for k, v in paired.items()]
    md += ["", f"Type shape vs TableFormer (points, chosen selector, test): {json.dumps({k: round(v, 2) for k, v in worst.items()})}",
           f"Exported {len(model_json['trees'])} trees over {len(ks)} features; export check max |margin difference| {err:.2e}."]
    open(os.path.join(out, "sel14.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
