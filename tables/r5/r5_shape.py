"""Round 5d: exact table shape (rows and columns both equal to the answer key, score.py `rc_exact`), the metric of the
harness corpus gate, on the 6,476 round-5 tables, from saved outputs only.

usage: r5_shape.py TF_PRED_ROOT TATR_PRED_ROOT SEL_ROWS_JSONL
  TableFormer: round-4 scored predictions (run 44879805191); TATR: round-5 scored predictions (run 45317894543);
  selector: the committed round-5 selector's 5-fold CV probabilities (`p_cv` in sel_rows.jsonl of run 45317894543).
CIs: 95% bootstrap over documents (r4_common.cluster_boot)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r4"))
from r4_common import *
import r3_select as R3


def main():
    tf_root, tatr_root, sel_rows = sys.argv[1:4]
    sel = {}
    for line in open(sel_rows, encoding="utf-8"):
        r = json.loads(line); sel[(r["set"], r["id"])] = r
    rows = []
    for s in SEL_SETS:
        sa, sb = R3.load(tf_root, os.environ.get("R5_A", "tableformer"), s, True), R3.load(tatr_root, os.environ.get("R5_B", "tatr16r5"), s, True)
        for tid in sorted(set(sa) & set(sb)):
            r = sel.get((s, tid))
            if r is None: continue
            a, b = float(bool(sa[tid].get("rc_exact"))), float(bool(sb[tid].get("rc_exact")))
            pick_tatr = r["p_cv"] > 0.5
            frozen = (b if r["p_frozen"] > 0.5 else a) if "p_frozen" in r else None
            rows.append(dict(set=s, doc=r["doc"], tf=a, tatr=b, sel=b if pick_tatr else a, oracle=max(a, b), zero=0.0, frozen=frozen))
    out = []; P = out.append
    P("## Round 5d: exact table shape (rows and columns both right), committed selector vs single models")
    P(f"Tables: {len(rows)} (the round-5 selector rows). Selector = committed round-5 trees' 5-fold CV picks. 95% CI: bootstrap over documents.\n")
    P("| dataset | tables | TableFormer [CI] | TATR [CI] | selector [CI] | selector - TableFormer [CI] | selector - TATR [CI] | oracle |")
    P("|---|---|---|---|---|---|---|---|")
    summary = {}
    for name, sub in [("all", rows)] + [(s, [r for r in rows if r["set"] == s]) for s in SEL_SETS]:
        if not sub: continue
        ci = {k: cluster_boot(sub, k, "zero") for k in ("tf", "tatr", "sel")}
        d_tf = cluster_boot(sub, "sel", "tf"); d_tatr = cluster_boot(sub, "sel", "tatr")
        orc = sum(r["oracle"] for r in sub) / len(sub)
        f = lambda c: f"{c[0]:.4f} [{c[1]:.4f}, {c[2]:.4f}]"
        g = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]"
        P(f"| {SEL_NAMES.get(name, name)} | {len(sub)} | {f(ci['tf'])} | {f(ci['tatr'])} | {f(ci['sel'])} | {g(d_tf)} | {g(d_tatr)} | {orc:.4f} |")
        if sub[0]["frozen"] is not None:
            fz = cluster_boot(sub, "frozen", "zero"); dfz = cluster_boot(sub, "frozen", "tf")
            P(f"| ... committed selector | | | | {f(fz)} | {g(dfz)} | | |")
        summary[name] = dict(n=len(sub), tf=ci["tf"], tatr=ci["tatr"], sel=ci["sel"], sel_minus_tf=d_tf, sel_minus_tatr=d_tatr, oracle=orc)
    txt = "\n".join(out); print(txt)
    os.makedirs(os.path.join(OUT, "r5d"), exist_ok=True)
    open(os.path.join(OUT, "r5d", "shape.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(summary, open(os.path.join(OUT, "r5d", "shape.json"), "w"))


if __name__ == "__main__":
    main()
