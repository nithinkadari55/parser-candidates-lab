"""Test-6 report. usage: agg6.py ARTIFACT_DIR OUT.md"""
import collections, glob, json, os, statistics, sys

A, OUT = sys.argv[1], sys.argv[2]
lines = []
EDGE = ("fullwidth_interrupts_columns", "footnotes", "caption_beside_figure", "exam_numbered_items", "landscape_or_spread")


def P(s=""):
    lines.append(s)
    print(s)


def jl(pat):
    for f in sorted(glob.glob(os.path.join(A, "**", pat), recursive=True)):
        for l in open(f, encoding="utf-8"):
            try:
                yield json.loads(l)
            except ValueError:
                pass


R = list(jl("eval6-*.jsonl"))
TEST = [r for r in R if r["split"] == "test"]


def cell(v):
    v = [x for x in v if "tau" in x]
    if not v:
        return "-"
    s = f"{statistics.mean(x['tau'] for x in v):.3f} / {statistics.mean(x['ed'] for x in v):.3f} / {100 * statistics.mean(x['exact'] for x in v):.1f}%"
    if "cov" in v[0]:
        s += f" / {100 * statistics.mean(x['cov'] for x in v):.0f}%"
    return s


def groups(recs):
    g = [("ALL", recs)]
    for b in ("complex", "medium", "simple"):
        g.append((f"bucket={b}", [r for r in recs if r["bucket"] == b]))
    g.append(("edge (any edge tag)", [r for r in recs if any(t in r["tags"] for t in EDGE)]))
    for t in ("newspaper_magazine", "exam_numbered_items", "boxed_or_shaded_region", "captions", "caption_beside_figure", "slides",
              "fullwidth_interrupts_columns", "footnotes", "landscape_or_spread"):
        g.append((f"tag={t}", [r for r in recs if t in r["tags"]]))
    for ds in ("odb", "drdoc"):
        g.append((f"dataset={ds}", [r for r in recs if r["ds"] == ds]))
    d = collections.defaultdict(list)
    for r in recs:
        d[str(r["layout"])].append(r)
    g += [(f"layout={k}", d[k]) for k in sorted(d)]
    return [(n, x) for n, x in g if x]


def table(recs, meths, title, legend):
    meths = [m for m in meths if any(m in r for r in recs)]
    P(f"\n### {title}")
    P(legend)
    P("")
    P("| group | pages | " + " | ".join(meths) + " |")
    P("|---|---|" + "---|" * len(meths))
    for name, g in groups(recs):
        P(f"| {name} | {len(g)} | " + " | ".join(cell([r[m] for r in g if m in r]) for m in meths) + " |")


def sign(recs, a, b):
    from scipy.stats import binomtest, wilcoxon
    rs = [r for r in recs if a in r and b in r and "exact" in r[a] and "exact" in r[b]]
    if not rs:
        return None
    w = sum(1 for r in rs if r[a]["exact"] and not r[b]["exact"])
    l = sum(1 for r in rs if r[b]["exact"] and not r[a]["exact"])
    pv = binomtest(w, w + l).pvalue if w + l else 1.0
    d = [r[a]["tau"] - r[b]["tau"] for r in rs]
    try:
        pw = wilcoxon(d).pvalue if any(x != 0 for x in d) else 1.0
    except ValueError:
        pw = 1.0
    ex = 100 * (statistics.mean(r[a]["exact"] for r in rs) - statistics.mean(r[b]["exact"] for r in rs))
    return len(rs), w, l, pv, statistics.mean(d), pw, ex


def main():
    P("## Test 6: learned pairwise re-ranker (kraken ROMLP idea) and PdfPig unsupervised order on top of XY-Cut++")
    P(f"Pages: {len(R)} (OmniDocBench English + DrDocBench dev); test split {len(TEST)}; models trained on the dev split only.")
    mism = sum(1 for r in R if r.get("tree_matches_xycutpp") is False)
    P(f"Sanity: instrumented XY-Cut++ tree differs from ro.xycutpp on {mism} pages (must be 0).")
    gt_body = ["xycutpp_gtorder_input", "xycutpp", "pdfpig", "romlp_stock", "romlp", "xycutpp_rr"]
    gt_full = ["full:xycutpp+deco", "full:xycutpp", "full:pdfpig", "full:pdfpig+deco", "full:romlp", "full:romlp+deco", "full:xycutpp_rr", "full:xycutpp+deco+rr"]
    h8 = ["h8:xycutpp", "h8:xycutpp_all", "h8:pdfpig", "h8:romlp_stock", "h8:romlp", "h8:romlp_all", "h8:xycutpp_rr_all", "h8:xycutpp_rr"]
    for recs, nm in ((TEST, "TEST split"), (R, "all pages (dev + test; dev was used for training)")):
        table(recs, gt_full, f"GT boxes incl. headers/footers/page numbers, {nm}",
              "current pick = full:xycutpp+deco (XY-Cut++ + repeated-decoration removal, no labels). (a) full:romlp (+deco), (b) full:xycutpp_rr, (c) full:xycutpp+deco+rr. cells: tau / ed / % exact")
        table(recs, gt_body, f"GT body boxes (headers/footers removed by GT labels), {nm}",
              "xycutpp_gtorder_input = XY-Cut++ fed in ground-truth order (earlier runs); all others get boxes in top-left order")
        table(recs, h8, f"Real pipeline: heron int8 boxes (score>=0.5), {nm}",
              "current pick = h8:xycutpp (page header/footer removed by the model label). (a) h8:romlp_all / h8:romlp, (b) h8:xycutpp_rr_all, (c) h8:xycutpp_rr. cells: tau / ed / % exact / % GT matched")
    P("\n### Sign tests against the current pick (pages only A fully correct vs only B; binomial; tau by Wilcoxon)")
    P("| A vs B | subset | pages | A only | B only | p (exact) | mean d-tau | p (tau) | d-exact pts |")
    P("|---|---|---|---|---|---|---|---|---|")
    pairs = [(a, "full:xycutpp+deco") for a in ("full:romlp", "full:romlp+deco", "full:xycutpp_rr", "full:xycutpp+deco+rr", "full:pdfpig+deco", "full:xycutpp")]
    pairs += [(a, "xycutpp") for a in ("romlp", "romlp_stock", "xycutpp_rr", "pdfpig")]
    pairs += [(a, "h8:xycutpp") for a in ("h8:romlp", "h8:romlp_all", "h8:romlp_stock", "h8:xycutpp_rr_all", "h8:xycutpp_rr", "h8:pdfpig")]
    pairs += [("xycutpp", "xycutpp_gtorder_input")]
    subsets = [("test", TEST), ("test complex+edge", [r for r in TEST if r["bucket"] == "complex" or any(t in r["tags"] for t in EDGE)]), ("all", R)]
    for a, b in pairs:
        for sn, rs in subsets:
            s = sign(rs, a, b)
            if s:
                n, w, l, pv, dt, pw, ex = s
                P(f"| {a} vs {b} | {sn} | {n} | {w} | {l} | {pv:.2g} | {dt:+.4f} | {pw:.2g} | {ex:+.1f} |")
    P("\n### Models")
    for f in sorted(glob.glob(os.path.join(A, "**", "train-*.json"), recursive=True)):
        t = json.load(open(f))
        P(f"- {t['kind']}: {t['features']} pair features, hidden {t['hidden']}, {t['params']} parameters = {t['fp32_kb']:.1f} KB fp32 "
          f"(npz {t['npz_bytes'] / 1024:.1f} KB); trained on {t['train_pages']} dev pages / {t['train_pairs']} pairs, validation {t['val_pages']} dev pages "
          f"(doc-separated), val pair accuracy {t['val_pair_acc']:.4f}, {t['epochs']} epochs, {t['seconds']:.0f} s")
    P("\n### Cost per page (shared runner; median / p95 ms, includes features + MLP + decoding)")
    for m in ["xycutpp", "pdfpig", "romlp", "xycutpp_rr", "full:xycutpp+deco", "full:xycutpp+deco+rr", "h8:xycutpp", "h8:romlp", "h8:xycutpp_rr"]:
        v = sorted(r[m]["ms"] for r in R if m in r and "ms" in r[m])
        if v:
            P(f"- {m}: {statistics.median(v):.2f} / {v[int(0.95 * (len(v) - 1))]:.2f} ms")
    un = [r.get("uncertain_nodes", 0) for r in R if "uncertain_nodes" in r]
    if un:
        P(f"- pages with at least one uncertain XY-Cut++ node (re-ranked in (b)/(c)): {sum(1 for u in un if u)} of {len(un)}")
    P("\n### Failure examples (test split): (c) full:xycutpp+deco+rr wrong where the current pick is right, and the reverse")
    worse = [r for r in TEST if "full:xycutpp+deco+rr" in r and "full:xycutpp+deco" in r and "exact" in r["full:xycutpp+deco+rr"]
             and r["full:xycutpp+deco"].get("exact") and not r["full:xycutpp+deco+rr"]["exact"]]
    better = [r for r in TEST if "full:xycutpp+deco+rr" in r and "full:xycutpp+deco" in r and "exact" in r["full:xycutpp+deco+rr"]
              and not r["full:xycutpp+deco"].get("exact") and r["full:xycutpp+deco+rr"]["exact"]]
    for name, rs in (("re-ranker breaks the pick", worse), ("re-ranker fixes the pick", better)):
        P(f"\n{name}: {len(rs)} pages")
        for r in rs[:10]:
            pa, pb = r["full:xycutpp+deco"].get("pred"), r["full:xycutpp+deco+rr"].get("pred")
            k = next((i for i in range(min(len(pa or []), len(pb or []))) if pa[i] != pb[i]), None)
            P(f"- {r['id'][:80]} | {r['ds']} {r['layout']} {r['source']} | tags {','.join(r['tags'])} | n={r['n']} uncertain_nodes={r.get('uncertain_nodes')}"
              + (f" | first difference at position {k}: pick reads GT region {pa[k]}, re-ranker reads {pb[k]}" if k is not None else ""))
    rw = [r for r in TEST if "romlp" in r and "xycutpp" in r and "exact" in r["romlp"] and r["xycutpp"].get("exact") and not r["romlp"]["exact"]]
    P(f"\n(a) ROMLP alone wrong where XY-Cut++ is right (body boxes, test): {len(rw)} pages, e.g.")
    for r in rw[:6]:
        P(f"- {r['id'][:80]} | {r['layout']} {r['source']} | n={r['n']} | ROMLP tau {r['romlp']['tau']:.3f}")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
