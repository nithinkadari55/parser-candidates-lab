import collections, glob, json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ro, odb

A = sys.argv[1]
OUT = sys.argv[2]
PP_SKIP = {"header", "footer", "number", "header_image", "footer_image"}
lines = []


def P(s=""):
    lines.append(s)
    print(s)


def jl(pattern):
    recs = []
    for f in sorted(glob.glob(os.path.join(A, "**", pattern), recursive=True)):
        for l in open(f, encoding="utf-8"):
            try:
                recs.append(json.loads(l))
            except ValueError:
                pass
    return recs


def cell(v, keys=("tau", "ed", "exact")):
    if not v:
        return "-"
    tau = statistics.mean(x["tau"] for x in v)
    ed = statistics.mean(x["ed"] for x in v)
    ex = 100 * statistics.mean(x["exact"] for x in v)
    s = f"{tau:.3f} / {ed:.3f} / {ex:.0f}%"
    if "cov" in v[0]:
        s += f" / {100 * statistics.mean(x['cov'] for x in v):.0f}%"
    return s


def groups(recs):
    g = [("ALL", recs)]
    for key in ("split", "bucket"):
        d = collections.defaultdict(list)
        for r in recs:
            d[r[key]].append(r)
        g += [(f"{key}={k}", d[k]) for k in sorted(d)]
    g.append(("bucket=edge (any edge tag)", [r for r in recs if odb.edge(r["tags"])]))
    d = collections.defaultdict(list)
    for r in recs:
        d[r["layout"]].append(r)
    g += [(f"layout={k}", d[k]) for k in sorted(d)]
    d = collections.defaultdict(list)
    for r in recs:
        for t in r["tags"]:
            d[t].append(r)
    g += [(f"tag={k}", d[k]) for k in sorted(d)]
    d = collections.defaultdict(list)
    for r in recs:
        d[r["source"]].append(r)
    g += [(f"source={k}", d[k]) for k in sorted(d)]
    return g


def table(recs, meths, title, legend):
    P(f"### {title}")
    P(legend)
    P("")
    P("| group | pages | " + " | ".join(meths) + " |")
    P("|---|---|" + "---|" * len(meths))
    for name, g in groups(recs):
        P(f"| {name} | {len(g)} | " + " | ".join(cell([r[m] for r in g if m in r]) for m in meths) + " |")
    P("")


def merge_gt():
    by = {}
    for r in jl("gt-*.jsonl"):
        by.setdefault(r["img"], {}).update(r)
    return list(by.values())


def match(gtb, regs):
    m = {}
    for g, b in enumerate(gtb):
        ga = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        best, bi = 0, None
        for k, r in enumerate(regs):
            inter = max(0, min(b[2], r[2]) - max(b[0], r[0])) * max(0, min(b[3], r[3]) - max(b[1], r[1]))
            ra = max(1e-6, (r[2] - r[0]) * (r[3] - r[1]))
            if (inter / ga >= 0.5 or inter / ra >= 0.8) and inter > best:
                best, bi = inter, k
        if bi is not None:
            m[g] = bi
    return m


def evaluate(p, regs, order):
    m = match(p["boxes"], regs)
    if len(m) < 2:
        return None
    rank = {k: i for i, k in enumerate(order)}
    gts = sorted(m)
    pred = sorted(gts, key=lambda g: (rank[m[g]], p["boxes"][g][1], p["boxes"][g][0]))
    s = ro.score(pred, gts)
    s["cov"] = len(m) / len(p["boxes"])
    return s


def det():
    pages = {p["img"]: p for p in odb.pages() if len(p["boxes"]) >= 2}
    dl = {r["img"]: r for r in jl("heron-*.jsonl") if "items" in r}
    dlerr = [r for r in jl("heron-*.jsonl") if "error" in r]
    pp = {r["img"]: r for r in jl("pp-*.jsonl") if "items" in r}
    out = []
    for img, p in pages.items():
        if img not in dl and img not in pp:
            continue
        W, H = p["W"], p["H"]
        rec = {k: p[k] for k in ("img", "split", "layout", "source", "bucket", "tags")}
        rec["n"] = len(p["boxes"])
        rec["in_heron"] = img in dl
        if img in dl:
            its = [i for i in dl[img]["items"] if not i["furniture"] and i["label"] not in ("page_header", "page_footer")]
            regs = [i["box"] for i in its]
            labs = [i["label"] for i in its]
            rec["heron_sec"] = dl[img]["sec"]
            if regs:
                for k, o in {"heron:docling_rs": list(range(len(regs))),
                             "heron:docling_rb_py": ro.docling_rb(regs, labs, W, H),
                             "heron:xycut_x": ro.xycut(regs, W, H, "x"),
                             "heron:xycutpp": ro.xycutpp(regs, W, H),
                             "heron:xycutpp_lab": ro.xycutpp_lab(regs, labs, W, H)}.items():
                    s = evaluate(p, regs, o)
                    if s:
                        rec[k] = s
        if img in pp:
            its = [i for i in pp[img]["items"] if i["label"] not in PP_SKIP]
            regs = [i["box"] for i in its]
            rec["pp_sec"] = pp[img]["sec"]
            if regs:
                for k, o in {"pp:ppdlv3_order": list(range(len(regs))),
                             "pp:docling_rb_py": ro.docling_rb(regs, ["text"] * len(regs), W, H),
                             "pp:xycut_x": ro.xycut(regs, W, H, "x"),
                             "pp:xycutpp": ro.xycutpp(regs, W, H)}.items():
                    s = evaluate(p, regs, o)
                    if s:
                        rec[k] = s
        out.append(rec)
    return out, len(dl), len(dlerr), len(pp)


def main():
    P("## Reading-order benchmark (OmniDocBench English pages)")
    gt = merge_gt()
    gm = ["docling_rb", "xycut_yfirst", "xycut_xfirst", "xycutpp", "xycutpp_lab", "xycutpp_row", "xycutpp_nostack"]
    table(gt, gm, f"A. Ground-truth boxes ({len(gt)} pages)", "cells: mean Kendall tau / mean normalised edit distance / % pages fully correct")
    for a, b in [("xycutpp", "docling_rb"), ("xycutpp", "xycut_xfirst"), ("xycutpp_lab", "xycutpp")]:
        w = sum(1 for r in gt if a in r and b in r and r[a]["exact"] and not r[b]["exact"])
        l = sum(1 for r in gt if a in r and b in r and r[b]["exact"] and not r[a]["exact"])
        try:
            from scipy.stats import binomtest
            pv = binomtest(w, w + l).pvalue if w + l else 1.0
        except Exception:
            pv = float("nan")
        P(f"- {a} vs {b}: pages only {a} fully correct {w}, only {b} {l}, sign-test p={pv:.2g}")
    P("")
    dt, ndl, nerr, npp = det()
    json.dump(dt, open(os.path.join(os.path.dirname(OUT), "det_results.json"), "w"))
    both = [r for r in dt if r["in_heron"] and "heron:docling_rs" in r and "pp:ppdlv3_order" in r]
    table(both, ["heron:docling_rs", "heron:docling_rb_py", "heron:xycut_x", "heron:xycutpp", "heron:xycutpp_lab", "pp:ppdlv3_order", "pp:xycutpp", "pp:docling_rb_py"],
          f"B. Real pipeline: heron boxes from docling-rs 2.84.0 vs PP-DocLayoutV3 boxes ({len(both)} pages; heron ok {ndl}, heron errors {nerr})",
          "cells: tau / edit distance / % pages fully correct / % GT regions matched to a detected box (order scored on matched regions)")
    allpp = [r for r in dt if "pp:ppdlv3_order" in r and "pp:xycutpp" in r]
    table(allpp, ["pp:ppdlv3_order", "pp:xycutpp", "pp:xycut_x", "pp:docling_rb_py"],
          f"C. Same PP-DocLayoutV3 boxes, model order vs logic ({len(allpp)} pages)", "cells: tau / edit distance / % exact / % GT matched")
    for f in glob.glob(os.path.join(A, "**", "own.json"), recursive=True):
        R = json.load(open(f))
        meths = [k for k in R[0] if k != "case"]
        P("### D. Our cases (hand-checked snippet order)")
        P("cells: OK = all snippets found in order; else pair accuracy; found/total")
        P("")
        P("| method | " + " | ".join(r["case"] for r in R) + " | fully correct | mean pair acc |")
        P("|---|" + "---|" * (len(R) + 2))
        for m in meths:
            cs, ex, pa = [], 0, []
            for r in R:
                v = r.get(m, {})
                if "exact" in v:
                    cs.append(("OK" if v["exact"] else f"{v['pair_acc']:.2f}") + f" {v['found']}/{v['total']}")
                    ex += v["exact"]
                    pa.append(v["pair_acc"])
                else:
                    cs.append("n/a: " + str(v.get("error", ""))[:40])
            P(f"| {m} | " + " | ".join(cs) + f" | {ex}/{len(R)} | {statistics.mean(pa) if pa else 0:.3f} |")
        P("")
    for f in glob.glob(os.path.join(A, "**", "timing.json"), recursive=True):
        P("### E. Timing (dedicated clean job)")
        P("```")
        P(json.dumps(json.load(open(f)), indent=1))
        P("```")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
