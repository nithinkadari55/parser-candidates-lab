"""Gap 14 evaluation. usage: r4_miss_eval.py ROOT (searches ROOT/**/miss-*.jsonl)"""
import os, sys, glob, json, collections
from r4_common import *

SP = {"spartan-boundary", "spartan-text"}
VEC = {"vec-grid", "vec-band", "vec-rect"}
VARIANTS = [("SPARTAN (boundary + text)", SP), ("SPARTAN boundary only", {"spartan-boundary"}), ("SPARTAN text (borderless) only", {"spartan-text"}),
            ("vector rules (pdf-inspector idea)", VEC), ("union SPARTAN + vector", SP | VEC), ("agreement SPARTAN and vector", "agree")]
DSN = {"dln": "DocLayNet test (CDLA-Permissive-1.0)", "odb": "OmniDocBench EN pages with original PDFs (research-only)", "sci": "SCI-3000 sample (CC BY 4.0 / BY-SA 4.0)"}


def match(gt, pred, thr):
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)), reverse=True)
    ug, up = {}, set()
    for v, gi, pi in pairs:
        if v < thr: break
        if gi in ug or pi in up: continue
        ug[gi] = pi; up.add(pi)
    return ug, up


def added(r, srcs, with_fig):
    ps = [p for p in r["props"] if p["cov_heron"] < 0.3 and (with_fig or not p["in_fig"])]
    if srcs == "agree":
        vec = [p["box"] for p in ps if p["src"] in VEC]
        ps = [p for p in ps if p["src"] in SP and any(iou(p["box"], v) >= 0.3 for v in vec)]
    else:
        ps = [p for p in ps if p["src"] in srcs]
    out = []
    for p in ps:
        if all(iou(p["box"], q) <= 0.5 for q in out): out.append(p["box"])
    return out


def tags(t, r):
    T = []
    hr, vr = t.get("h_rules", 0), t.get("v_rules", 0)
    T.append("ruled grid" if (vr >= 2 and hr >= 2) else "horizontal rules only" if hr >= 2 else "borderless")
    if t.get("rows", 0) <= 3 or (t.get("med_cols", 0) <= 2 and t.get("rows", 0) <= 8): T.append("tiny or key-value")
    figs = r.get("gt_figures") or []
    if t.get("in_fig") or (figs and cover(t["box"], figs) >= 0.5): T.append("in-figure")
    if t.get("vertical_frac", 0) >= 0.5: T.append("rotated")
    if t.get("ink_p1", 0) > 110 or (t.get("bg_med", 255) - t.get("ink_p1", 0)) < 80: T.append("low-contrast")
    if t.get("n_words", 1) == 0: T.append("no text layer")
    if t.get("heron_other"): T.append("heron label: " + t["heron_other"])
    elif 0.1 <= t.get("heron_iou", 0) < 0.5: T.append("heron box partial (IoU .1-.5)")
    return T


def main():
    root = sys.argv[1]
    recs = []
    for p in sorted(glob.glob(os.path.join(root, "**", "miss-*.jsonl"), recursive=True)): recs += jl(p)
    seen = set(); R = []
    for r in recs:
        if r["id"] in seen: continue
        seen.add(r["id"]); R.append(r)
    out = []; P = out.append
    P("## Gap 14: missed tables - train-free proposals where heron INT8 found no table")
    P("Added region = proposal with < 30% of its area covered by heron INT8 tables (score >= 0.5). 'with figures' keeps proposals inside heron Picture boxes "
      "(they always carry aligned words or ruling lines by construction); 'no figures' drops them. A heron miss = GT table without a heron table at IoU >= 0.5. "
      "False table = added region with IoU < 0.5 to every GT table.")
    summary = {}
    for ds in ["dln", "odb", "sci", "all"]:
        sub = [r for r in R if (ds == "all" or r["ds"] == ds) and "error" not in r]
        err = sum(1 for r in R if (ds == "all" or r["ds"] == ds) and "error" in r)
        if not sub: continue
        ng = sum(len(r["gt"]) for r in sub); npg = len(sub)
        miss = {}; htp = 0; nh = 0
        for r in sub:
            gb = [t["box"] for t in r["gt"]]; ug, _ = match(gb, r["heron_tables"], 0.5)
            miss[r["id"]] = set(range(len(gb))) - set(ug); htp += len(ug); nh += len(r["heron_tables"])
        nm = sum(len(v) for v in miss.values())
        P(f"\n### {DSN.get(ds, 'all datasets')}: {npg} pages ({err} errors), {ng} GT tables, heron INT8 recall@.5 {htp / max(1, ng):.3f}, precision {htp / max(1, nh):.3f}, misses {nm}")
        P("| proposer | figures | added / 100 pages | heron misses recovered @.5 | inc. recall @.5 | inc. recall @.3 | precision of added @.5 | precision @.3 | false tables / 100 pages | heron+added recall | heron+added precision |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        res = {}
        for name, srcs in VARIANTS:
            for wf in (True, False):
                na = rec5 = rec3 = good5 = good3 = false = 0
                for r in sub:
                    gb = [t["box"] for t in r["gt"]]; a = added(r, srcs, wf); na += len(a)
                    mg = [gb[i] for i in sorted(miss[r["id"]])]
                    for thr in (0.5, 0.3):
                        ug, up = match(mg, a, thr)
                        if thr == 0.5: rec5 += len(ug); good5 += len(up)
                        else: rec3 += len(ug); good3 += len(up)
                    false += sum(1 for b in a if all(iou(b, g) < 0.5 for g in gb))
                fr = htp + rec5; fp_ = nh + na
                res[(name, wf)] = dict(added=na, rec5=rec5, rec3=rec3, prec5=good5 / max(1, na), false100=100 * false / npg)
                P(f"| {name} | {'with' if wf else 'no'} | {100 * na / npg:.1f} | {rec5} of {nm} | {rec5 / max(1, nm):.3f} | {rec3 / max(1, nm):.3f} | {good5 / max(1, na):.3f} | {good3 / max(1, na):.3f} | {100 * false / npg:.1f} | {fr / max(1, ng):.3f} | {(htp + good5) / max(1, fp_):.3f} |")
        summary[ds] = dict(pages=npg, gt=ng, heron_recall=htp / max(1, ng), misses=nm, variants={f"{k[0]}|{'fig' if k[1] else 'nofig'}": v for k, v in res.items()})
        # taxonomy
        P(f"\n**Miss taxonomy ({ds})** - tags overlap; recovered = union SPARTAN + vector, figures kept")
        cnt = collections.Counter(); mt = collections.Counter(); r5 = collections.Counter(); r3 = collections.Counter()
        for r in sub:
            gb = [t["box"] for t in r["gt"]]; a = added(r, SP | VEC, True)
            ids = sorted(miss[r["id"]]); mg = [gb[i] for i in ids]
            u5, _ = match(mg, a, 0.5); u3, _ = match(mg, a, 0.3)
            for i, t in enumerate(r["gt"]):
                for tg in tags(t, r):
                    cnt[tg] += 1
                    if i in miss[r["id"]]:
                        mt[tg] += 1; j = ids.index(i)
                        if j in u5: r5[tg] += 1
                        if j in u3: r3[tg] += 1
        P("| tag | GT tables | heron miss rate | misses | recovered @.5 | recovered @.3 |\n|---|---|---|---|---|---|")
        for tg, c in sorted(cnt.items(), key=lambda kv: -mt[kv[0]]):
            P(f"| {tg} | {c} | {mt[tg] / max(1, c):.3f} | {mt[tg]} | {r5[tg]} | {r3[tg]} |")
        sp = sorted(r.get("sec_spartan", 0) for r in sub); vc = sorted(r.get("sec_vector", 0) for r in sub); hs = sorted((r.get("heron_sec") or 0) for r in sub)
        q = lambda v, f_: v[int(f_ * (len(v) - 1))] * 1000 if v else float("nan")
        P(f"\nSpeed per page: SPARTAN 300 DPI (render + words + detect) p50 {q(sp, .5):.0f} ms / p95 {q(sp, .95):.0f} ms; vector proposer p50 {q(vc, .5):.0f} / p95 {q(vc, .95):.0f} ms; heron INT8 p50 {q(hs, .5):.0f} ms. "
          f"Mappings used: {dict(collections.Counter(r.get('mapping') for r in sub))}; vector notes: {dict(collections.Counter(r.get('vec_note') for r in sub if r.get('vec_note')))}")
    txt = "\n".join(out); print(txt)
    os.makedirs(os.path.join(root, "r4"), exist_ok=True)
    open(os.path.join(root, "r4", "miss.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(summary, open(os.path.join(root, "r4", "miss_summary.json"), "w"))


if __name__ == "__main__":
    main()
