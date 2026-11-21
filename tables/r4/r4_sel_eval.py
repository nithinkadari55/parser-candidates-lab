"""Gap 15 evaluation: per-table selector (round-3 features) on real PDF words.
usage: r4_sel_eval.py PRED_ROOT [R3_PRED_ROOT]"""
import os, sys, json, glob, time, statistics
from r4_common import *
import numpy as np
import r3_select as R3
from score import gt_chunks

A, B = "tableformer", "tatr16"


def rows_for(root, sets, a=A, b=B, sets_dir=SETS, doc_fn=None):
    rows = []
    for s in sets:
        pa, pb = R3.load(root, a, s, False), R3.load(root, b, s, False)
        sa, sb = R3.load(root, a, s, True), R3.load(root, b, s, True)
        for tid in sorted(set(pa) & set(pb) & set(sa) & set(sb)):
            p = os.path.join(sets_dir, s, tid + ".json")
            if not os.path.exists(p): continue
            gt = json.load(open(p, encoding="utf-8"))
            words = gt_chunks(gt)
            t0 = time.perf_counter()
            fa, fb = R3.feats(pa[tid].get("cells") or [], words), R3.feats(pb[tid].get("cells") or [], words)
            x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
            x.update({"d_" + k: fb[k] - fa[k] for k in fa})
            x["agree"] = R3.agreement(pa[tid].get("cells") or [], pb[tid].get("cells") or [])
            x["n_words"] = len(words)
            ft = (time.perf_counter() - t0) * 1000
            va, vb = R3.val(sa[tid]), R3.val(sb[tid])
            if va is None or vb is None: continue
            doc = doc_fn(s, gt) if doc_fn else gt.get("doc", tid)
            rows.append(dict(id=tid, set=s, bucket=sa[tid]["bucket"], doc=s + ":" + str(doc), va=va, vb=vb,
                             ta=sa[tid].get("teds"), tb=sb[tid].get("teds"), x=x, feat_ms=ft,
                             tf_mode=pa[tid].get("tf_mode"), ms_a=pa[tid].get("ms_total"), ms_b=pb[tid].get("ms_total"),
                             comp=bool(gt.get("tags", {}).get("scitsr_comp")), a_empty=not (pa[tid].get("cells")), b_empty=not (pb[tid].get("cells"))))
    return rows


def xgb():
    import xgboost as X
    return X.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=0)


def main():
    root = sys.argv[1]; r3root = sys.argv[2] if len(sys.argv) > 2 else None
    rows = rows_for(root, SEL_SETS)
    print("r4 tables", len(rows), flush=True)
    keys = sorted(rows[0]["x"])
    X = np.array([[r["x"][k] for k in keys] for r in rows], float)
    y = np.array([1 if r["vb"] > r["va"] + 1e-9 else 0 for r in rows])
    groups = np.array([r["doc"] for r in rows])
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    p_lr = np.zeros(len(rows)); p_x = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        p_lr[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        p_x[te] = xgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    p_fz = None; n_r3 = 0
    if r3root and os.path.isdir(r3root) and os.path.exists(os.path.join(SETS, "ptn", "index.json")):
        r3 = rows_for(r3root, R3.SETS_, a="tableformer", b="tatr16", doc_fn=lambda s, gt: R3.doc_key(s, gt))
        n_r3 = len(r3)
        if r3:
            X3 = np.array([[r["x"][k] for k in keys] for r in r3], float); y3 = np.array([1 if r["vb"] > r["va"] + 1e-9 else 0 for r in r3])
            p_fz = xgb().fit(X3, y3).predict_proba(X)[:, 1]
    print("round-3 training tables for the frozen selector", n_r3, flush=True)
    imp = sorted(zip(xgb().fit(X, y).feature_importances_, keys), reverse=True)[:12]
    for i, r in enumerate(rows):
        x = r["x"]
        r["fit"] = r["vb"] if x["b_tatr_fit"] > x["a_tatr_fit"] + 1e-9 else r["va"]
        r["lr"] = r["vb"] if p_lr[i] > 0.5 else r["va"]
        r["xg"] = r["vb"] if p_x[i] > 0.5 else r["va"]
        r["fz"] = (r["vb"] if p_fz[i] > 0.5 else r["va"]) if p_fz is not None else None
        r["oracle"] = max(r["va"], r["vb"])
        r["pick_xg"] = int(p_x[i] > 0.5)
        if r["ta"] is not None and r["tb"] is not None: r["t_xg"] = r["tb"] if p_x[i] > 0.5 else r["ta"]
    out = []; P = out.append
    P("## Gap 15: table selector on real PDF words (pdfplumber words re-extracted from the source PDFs)")
    P(f"Value per table: TEDS-Struct (GriTS-Top where the grid exceeds 1200 cells). Selector = round-3 features + XGBoost; "
      f"'retrained' = 5-fold GroupKFold by document on these {len(rows)} tables only; 'frozen r3' = the round-3 selector trained on the "
      f"{n_r3} round-3 tables (answer-key words for PubTabNet/FinTabNet/PT1M) and applied unchanged. CI = 95% bootstrap over documents.")
    cols = [("TableFormer", "va"), ("TATR fp16", "vb"), ("rule: token fit", "fit"), ("LR retrained", "lr"), ("XGB retrained", "xg"), ("XGB frozen r3", "fz"), ("oracle", "oracle")]
    P("\n| dataset | tables | docs | " + " | ".join(c for c, _ in cols) + " | best single | gain XGB-retrained vs best [95% CI] | Wilcoxon p | gain frozen-r3 vs best [95% CI] | regret vs oracle |")
    P("|" + "---|" * (len(cols) + 8))
    summary = {}
    for name, sub in [("all", rows)] + [(s, [r for r in rows if r["set"] == s]) for s in SEL_SETS]:
        if not sub: continue
        m = {k: (sum(r[k] for r in sub) / len(sub) if sub[0][k] is not None else None) for _, k in cols}
        best = "va" if m["va"] >= m["vb"] else "vb"
        for r in sub: r["_best"] = r[best]
        g = cluster_boot(sub, "xg", "_best"); p = wilcoxon_p([r["xg"] for r in sub], [r["_best"] for r in sub])
        gf = cluster_boot(sub, "fz", "_best") if m["fz"] is not None else None
        label = SEL_NAMES.get(name, name)
        P(f"| {label} | {len(sub)} | {len(set(r['doc'] for r in sub))} | " + " | ".join("-" if m[k] is None else f"{m[k]:.4f}" for _, k in cols)
          + f" | {'TableFormer' if best == 'va' else 'TATR'} | {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] | {p:.2g} | "
          + (f"{gf[0]:+.4f} [{gf[1]:+.4f}, {gf[2]:+.4f}]" if gf else "-") + f" | {m['oracle'] - m['xg']:.4f} |")
        summary[name] = dict(n=len(sub), docs=len(set(r['doc'] for r in sub)), **{k: m[k] for _, k in cols}, best=best, gain=g, p=p, gain_frozen=gf)
    P("\n### By complexity bucket (TEDS-Struct means)")
    P("| dataset/bucket | n | TableFormer | TATR fp16 | XGB retrained | XGB frozen r3 | oracle | gain vs best [95% CI] |\n|---|---|---|---|---|---|---|---|")
    for s in SEL_SETS:
        for b in ["simple", "medium", "complex", "edge"] + (["SciTSR-COMP"] if s == "r4scitsr" else []):
            sub = [r for r in rows if r["set"] == s and (r["comp"] if b == "SciTSR-COMP" else r["bucket"] == b)]
            if len(sub) < 5: continue
            ma, mb = sum(r["va"] for r in sub) / len(sub), sum(r["vb"] for r in sub) / len(sub)
            for r in sub: r["_best"] = r["va"] if ma >= mb else r["vb"]
            g = cluster_boot(sub, "xg", "_best")
            fz = (sum(r["fz"] for r in sub) / len(sub)) if sub[0]["fz"] is not None else None
            P(f"| {s}/{b} | {len(sub)} | {ma:.4f} | {mb:.4f} | {sum(r['xg'] for r in sub) / len(sub):.4f} | {'-' if fz is None else f'{fz:.4f}'} | {sum(r['oracle'] for r in sub) / len(sub):.4f} | {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] |")
    P("\n### TEDS with content (cells filled with our PDF words)")
    P("| dataset | n | TableFormer | TATR fp16 | XGB retrained pick |\n|---|---|---|---|---|")
    for s in SEL_SETS:
        sub = [r for r in rows if r["set"] == s and r.get("t_xg") is not None]
        if sub: P(f"| {s} | {len(sub)} | {sum(r['ta'] for r in sub) / len(sub):.4f} | {sum(r['tb'] for r in sub) / len(sub):.4f} | {sum(r['t_xg'] for r in sub) / len(sub):.4f} |")
    P("\n### Failures and TableFormer table detection on the real page")
    P("| dataset | TF found table on full page | TF needed crop fallback | TF no table | TF empty | TATR empty | TF fail (<0.5) | TATR fail | XGB-retrained fail |\n|---|---|---|---|---|---|---|---|---|")
    for s in SEL_SETS:
        sub = [r for r in rows if r["set"] == s]
        if not sub: continue
        c = lambda f: sum(1 for r in sub if f(r))
        P(f"| {s} | {c(lambda r: r['tf_mode'] == 'page')} | {c(lambda r: r['tf_mode'] == 'crop')} | {c(lambda r: r['tf_mode'] == 'none')} | {c(lambda r: r['a_empty'])} | {c(lambda r: r['b_empty'])} | "
          f"{c(lambda r: r['va'] < 0.5)} | {c(lambda r: r['vb'] < 0.5)} | {c(lambda r: r['xg'] < 0.5)} |")
    P("\n### Speed (ms per table on a 4-vCPU GitHub runner, default ONNX threads)")
    P("| dataset | TableFormer page pipeline p50 / p95 | TATR fp16 p50 / p95 | selector features p50 / p95 |\n|---|---|---|---|")
    q = lambda v, f: (sorted(v)[int(f * (len(v) - 1))] if v else float("nan"))
    for s in SEL_SETS:
        sub = [r for r in rows if r["set"] == s]
        if not sub: continue
        a = [r["ms_a"] for r in sub if r["ms_a"]]; b = [r["ms_b"] for r in sub if r["ms_b"]]; f = [r["feat_ms"] for r in sub]
        P(f"| {s} | {q(a, .5):.0f} / {q(a, .95):.0f} | {q(b, .5):.0f} / {q(b, .95):.0f} | {q(f, .5):.1f} / {q(f, .95):.1f} |")
    P("\nXGBoost feature importance (fitted on all r4 tables): " + ", ".join(f"{k} {v:.3f}" for v, k in imp))
    txt = "\n".join(out); print(txt)
    os.makedirs(R4OUT, exist_ok=True)
    open(os.path.join(R4OUT, "sel.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(summary, open(os.path.join(R4OUT, "sel_summary.json"), "w"), default=str)
    with open(os.path.join(R4OUT, "sel_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if k != "x"}) + "\n")


if __name__ == "__main__":
    main()
