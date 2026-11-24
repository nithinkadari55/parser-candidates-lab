"""Round 5 (harness port): per-table selector retrained on docling.rs word cells, exported for the Rust evaluator.

usage: r5_sel.py TF_PRED_ROOT TATR_PRED_ROOT WORDS_ROOT
  TF_PRED_ROOT    tableformer_<set>_s<k>(.scored).jsonl of the round-4 run 44879805191 (docling-rs 2.84.0, unchanged:
                  TableFormer does not read our words)
  TATR_PRED_ROOT  tatr16r5_<set>_s<k>(.scored).jsonl from r5_tatr.py (docling.rs crops and words)
  WORDS_ROOT      <set>/<id>.json from tables/r5/words (docling-pdf PdfPage::word_cells, crop pixels)
Features: round-3 feats() + agreement() unchanged (r3_select.py), words = docling.rs word cells, the source the
harness reads at run time. 5-fold GroupKFold by document, XGBoost with the round-3/4 settings.
Writes $TB_WORK/out/r5/: sel.md, sel_summary.json, sel_rows.jsonl, table_selector.json (trees for the Rust evaluator),
selector_fixture.json (feature rows + xgboost predictions), features_fixture.json (cells + words + features)."""
import os, sys, json, glob, math, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r4"))
from r4_common import *
import numpy as np
import r3_select as R3
import r5_gaps
GAPS = os.environ.get("R5_GAPS") == "1"

R5OUT = os.path.join(OUT, os.environ.get("R5_OUT", "r5"))
A, B = "tableformer", os.environ.get("R5_B", "tatr16r5")
R4 = dict(tf=0.916, tatr=0.951, sel=0.961, oracle=0.968)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=0)


def xgb():
    import xgboost as X
    return X.XGBClassifier(**PARAMS)


def words_of(root, s, tid):
    p = os.path.join(root, s, tid + ".json")
    if not os.path.exists(p): return None
    w = json.load(open(p, encoding="utf-8"))
    if "error" in w: return None
    return [dict(text=x["text"], bbox=x["bbox"], gi=None) for x in w["words"]]


def rows_for(tf_root, tatr_root, words_root):
    rows = []; missing = 0
    for s in SEL_SETS:
        pa, pb = R3.load(tf_root, A, s, False), R3.load(tatr_root, B, s, False)
        sa, sb = R3.load(tf_root, A, s, True), R3.load(tatr_root, B, s, True)
        for tid in sorted(set(pa) & set(pb) & set(sa) & set(sb)):
            words = words_of(words_root, s, tid)
            gp = os.path.join(SETS, s, tid + ".json")
            if words is None or not os.path.exists(gp): missing += 1; continue
            gt = json.load(open(gp, encoding="utf-8"))
            ca, cb = pa[tid].get("cells") or [], pb[tid].get("cells") or []
            t0 = time.perf_counter()
            fa, fb = R3.feats(ca, words), R3.feats(cb, words)
            x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
            x.update({"d_" + k: fb[k] - fa[k] for k in fa})
            x["agree"] = R3.agreement(ca, cb)
            x["n_words"] = len(words)
            if GAPS: r5_gaps.add_to_row(x, ca, cb, words)
            ft = (time.perf_counter() - t0) * 1000
            va, vb = R3.val(sa[tid]), R3.val(sb[tid])
            if va is None or vb is None: continue
            rows.append(dict(id=tid, set=s, bucket=sa[tid]["bucket"], doc=s + ":" + str(gt.get("doc", tid)), va=va, vb=vb, x=x, feat_ms=ft,
                             comp=bool(gt.get("tags", {}).get("scitsr_comp")), ca=ca, cb=cb, words=words, ms_b=pb[tid].get("ms_total")))
    print("tables", len(rows), "skipped without words", missing, flush=True)
    return rows


def leaf_sum(trees, x):
    m = np.float32(0.0)
    for t in trees:
        i = 0
        while t["left"][i] != -1:
            v = np.float32(x[t["feature"][i]])
            if math.isnan(v): i = t["left"][i] if t["default_left"][i] else t["right"][i]
            else: i = t["left"][i] if v < np.float32(t["threshold"][i]) else t["right"][i]
        m = np.float32(m + np.float32(t["threshold"][i]))
    return m


def export(model, keys, X):
    bst = model.get_booster()
    raw = json.loads(bytes(bst.save_raw(raw_format="json")).decode())
    learner = raw["learner"]
    assert learner["objective"]["name"] == "binary:logistic", learner["objective"]
    bs = float(str(learner["learner_model_param"]["base_score"]).strip("[]").split(",")[0])
    base_margin = float(np.float32(math.log(bs / (1.0 - bs))))
    trees = []
    for t in learner["gradient_booster"]["model"]["trees"]:
        trees.append(dict(left=[int(v) for v in t["left_children"]], right=[int(v) for v in t["right_children"]],
                          feature=[int(v) for v in t["split_indices"]], threshold=[float(np.float32(v)) for v in t["split_conditions"]],
                          default_left=[bool(v) for v in t["default_left"]]))
    import xgboost as XG
    want = bst.predict(XG.DMatrix(X.astype(np.float32), feature_names=None), output_margin=True)
    got = np.array([np.float32(np.float32(base_margin) + leaf_sum(trees, row)) for row in X.astype(np.float32)])
    err = float(np.max(np.abs(want - got)))
    print("export check: max |margin difference| over", len(X), "rows =", err, flush=True)
    assert err < 1e-4, err
    model_json = dict(format="contextual-table-selector-v1", objective="binary:logistic", features=keys, base_margin=base_margin,
                      threshold=0.5, xgboost=XG.__version__, params=PARAMS, words="docling-pdf 2.84.0 PdfPage::word_cells", tatr_rows_joined=os.environ.get("R5_JOIN", "none"), split_row_features=GAPS,
                      positive="tatr", trees=trees)
    return model_json, want, err


def main():
    tf_root, tatr_root, words_root = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = rows_for(tf_root, tatr_root, words_root)
    keys = sorted(rows[0]["x"])
    X = np.array([[r["x"][k] for k in keys] for r in rows], float)
    y = np.array([1 if r["vb"] > r["va"] + 1e-9 else 0 for r in rows])
    groups = np.array([r["doc"] for r in rows])
    from sklearn.model_selection import GroupKFold
    p_x = np.zeros(len(rows)); p_base = np.zeros(len(rows))
    gap_keys = set("abd"[j] + "_" + n for j in range(3) for n in r5_gaps.NAMES)
    base_cols = [j for j, k in enumerate(keys) if k not in gap_keys]
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        p_x[te] = xgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if GAPS:
            Xb = X[:, base_cols]
            p_base[te] = xgb().fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
    for i, r in enumerate(rows):
        r["xg"] = r["vb"] if p_x[i] > 0.5 else r["va"]
        r["xg_base"] = r["vb"] if p_base[i] > 0.5 else r["va"]
        r["oracle"] = max(r["va"], r["vb"]); r["p_cv"] = float(p_x[i])
    out = []; P = out.append
    P("## Round 5: table selector retrained on docling.rs word cells (the harness's run-time word source)")
    P(f"Value per table: TEDS-Struct (GriTS-Top where the grid exceeds 1200 cells). Tables: {len(rows)}, documents (CV groups): {len(set(groups))}. "
      f"TableFormer = round-4 predictions (docling-rs 2.84.0, unchanged). TATR = v1.1 fp16 on docling.rs crops with docling.rs words. "
      f"Selector = round-3 features on docling.rs words + XGBoost, 5-fold GroupKFold by document. CI = 95% bootstrap over documents.")
    P("\n| dataset | tables | docs | TableFormer | TATR fp16 (docling words) | selector (CV) | oracle | gain vs best single [95% CI] | Wilcoxon p |")
    P("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for name, sub in [("all", rows)] + [(s, [r for r in rows if r["set"] == s]) for s in SEL_SETS]:
        if not sub: continue
        m = {k: sum(r[k] for r in sub) / len(sub) for k in ("va", "vb", "xg", "oracle")}
        best = "va" if m["va"] >= m["vb"] else "vb"
        for r in sub: r["_best"] = r[best]
        g = cluster_boot(sub, "xg", "_best"); p = wilcoxon_p([r["xg"] for r in sub], [r["_best"] for r in sub])
        P(f"| {SEL_NAMES.get(name, name)} | {len(sub)} | {len(set(r['doc'] for r in sub))} | {m['va']:.4f} | {m['vb']:.4f} | {m['xg']:.4f} | {m['oracle']:.4f} | "
          f"{g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] | {p:.2g} |")
        summary[name] = dict(n=len(sub), tf=m["va"], tatr=m["vb"], sel=m["xg"], oracle=m["oracle"], gain=g, p=p)
    a = summary["all"]
    P(f"\n**Round 4 (pdfplumber words): TableFormer {R4['tf']:.3f}, TATR {R4['tatr']:.3f}, selector {R4['sel']:.3f}, oracle {R4['oracle']:.3f}. "
      f"Round 5 (docling.rs words): TableFormer {a['tf']:.4f}, TATR {a['tatr']:.4f}, selector {a['sel']:.4f}, oracle {a['oracle']:.4f}. "
      f"Selector vs round 4: {a['sel'] - R4['sel']:+.4f}.**")
    if GAPS:
        gb = cluster_boot(rows, "xg", "xg_base"); pw = wilcoxon_p([r["xg"] for r in rows], [r["xg_base"] for r in rows])
        P("")
        P(f"**Split-row features: selector {a['sel']:.4f} vs {sum(r['xg_base'] for r in rows) / len(rows):.4f} without them (same folds); "
          f"difference {gb[0]:+.4f} [95% CI {gb[1]:+.4f}, {gb[2]:+.4f}], Wilcoxon p {pw:.2g}; {len(keys)} features.**")
        for s in SEL_SETS:
            sub = [r for r in rows if r["set"] == s]
            if sub:
                g2 = cluster_boot(sub, "xg", "xg_base")
                P(f"- {SEL_NAMES.get(s, s)}: {g2[0]:+.4f} [{g2[1]:+.4f}, {g2[2]:+.4f}]")
    picks = int(sum(p_x > 0.5)); P(f"\nSelector picks TATR on {picks} of {len(rows)} tables ({100 * picks / len(rows):.1f}%).")
    ft = sorted(r["feat_ms"] for r in rows); P(f"Feature time (Python) p50 {ft[len(ft) // 2]:.1f} ms, p95 {ft[int(0.95 * (len(ft) - 1))]:.1f} ms.")
    final = xgb().fit(X, y)
    imp = sorted(zip(final.feature_importances_, keys), reverse=True)[:12]
    P("XGBoost feature importance (final model, all tables): " + ", ".join(f"{k} {v:.3f}" for v, k in imp))
    model_json, margins, err = export(final, keys, X)
    P(f"Exported {len(model_json['trees'])} trees; export check max |margin difference| {err:.2e}.")
    txt = "\n".join(out); print(txt)
    os.makedirs(R5OUT, exist_ok=True)
    open(os.path.join(R5OUT, "sel.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(dict(summary=summary, round4=R4, tables=len(rows), docs=len(set(groups))), open(os.path.join(R5OUT, "sel_summary.json"), "w"), default=str)
    json.dump(model_json, open(os.path.join(R5OUT, "table_selector.json"), "w"), separators=(",", ":"))
    probs = final.predict_proba(X)[:, 1]
    rng = random.Random(5); idx = sorted(rng.sample(range(len(rows)), min(60, len(rows))))
    json.dump(dict(features=keys, rows=[dict(id=rows[i]["id"], x=[float(v) for v in X[i]], margin=float(margins[i]), prob=float(probs[i])) for i in idx]),
              open(os.path.join(R5OUT, "selector_fixture.json"), "w"))
    small = [i for i, r in enumerate(rows) if 3 <= len(r["words"]) <= 60 and 0 < len(r["ca"]) <= 80 and 0 < len(r["cb"]) <= 80]
    rng.shuffle(small); pick = []; per = {}
    for i in small:
        s = rows[i]["set"]
        if per.get(s, 0) < 3: pick.append(i); per[s] = per.get(s, 0) + 1
        if len(pick) >= 12: break
    strip = lambda cs: [dict(r0=c["r0"], r1=c["r1"], c0=c["c0"], c1=c["c1"], bbox=c["bbox"]) for c in cs]
    json.dump(dict(tables=[dict(id=rows[i]["id"], tableformer=strip(rows[i]["ca"]), tatr=strip(rows[i]["cb"]),
                                words=[dict(text=w["text"], bbox=w["bbox"]) for w in rows[i]["words"]], features=rows[i]["x"]) for i in pick]),
              open(os.path.join(R5OUT, "features_fixture.json"), "w"))
    with open(os.path.join(R5OUT, "sel_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if k not in ("x", "ca", "cb", "words", "_best")}) + "\n")


if __name__ == "__main__":
    main()
