"""Reading-order report: GT-box methods and add-ons, and the real pipeline per layout model.
usage: aggregate_ro.py ARTIFACT_DIR OUT.md OUT.json"""
import collections, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "layout"))
import ro, addons, datasets, pdx_order

A, OUT, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3]
lines, RES = [], {}


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


def cell(v):
    v = [x for x in v if "tau" in x]
    if not v:
        return "-"
    s = f"{statistics.mean(x['tau'] for x in v):.3f} / {statistics.mean(x['ed'] for x in v):.3f} / {100 * statistics.mean(x['exact'] for x in v):.0f}%"
    if "cov" in v[0]:
        s += f" / {100 * statistics.mean(x['cov'] for x in v):.0f}%"
    return s


def groups(recs):
    g = [("ALL", recs), ("split=test", [r for r in recs if r["split"] == "test"]), ("split=dev", [r for r in recs if r["split"] == "dev"])]
    for b in ("complex", "medium", "simple"):
        g.append((f"bucket={b}", [r for r in recs if r["bucket"] == b]))
    g.append(("bucket=edge (any edge tag)", [r for r in recs if datasets.edge(r["tags"])]))
    g.append(("complex+edge, test split", [r for r in recs if r["split"] == "test" and (r["bucket"] == "complex" or datasets.edge(r["tags"]))]))
    d = collections.defaultdict(list)
    for r in recs:
        for t in r["tags"]:
            d[t].append(r)
    g += [(f"tag={k}", d[k]) for k in sorted(d)]
    d = collections.defaultdict(list)
    for r in recs:
        d[str(r["layout"])].append(r)
    g += [(f"layout={k}", d[k]) for k in sorted(d)]
    return [(n, x) for n, x in g if x]


def table(recs, meths, title, legend):
    P(f"\n### {title}")
    P(legend)
    P("")
    P("| group | pages | " + " | ".join(meths) + " |")
    P("|---|---|" + "---|" * len(meths))
    for name, g in groups(recs):
        P(f"| {name} | {len(g)} | " + " | ".join(cell([r[m] for r in g if m in r]) for m in meths) + " |")


def sign(recs, a, b):
    from scipy.stats import binomtest
    rs = [r for r in recs if a in r and b in r and "exact" in r[a] and "exact" in r[b]]
    w = sum(1 for r in rs if r[a]["exact"] and not r[b]["exact"])
    l = sum(1 for r in rs if r[b]["exact"] and not r[a]["exact"])
    pv = binomtest(w, w + l).pvalue if w + l else 1.0
    dt = statistics.mean(r[a]["tau"] - r[b]["tau"] for r in rs) if rs else 0
    ex = 100 * (statistics.mean(r[a]["exact"] for r in rs) - statistics.mean(r[b]["exact"] for r in rs)) if rs else 0
    return w, l, pv, dt, ex, len(rs)


def merged(ds):
    by = {}
    for r in jl(f"ro-{ds}-*.jsonl"):
        by.setdefault(r["id"], {}).update(r)
    return list(by.values())


# ------------------------------------------------------------------ real pipeline
def match(gtb, regs):
    m = {}
    for g, b in enumerate(gtb):
        ga = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        best, bi = 0, None
        for k, r in enumerate(regs):
            i = max(0, min(b[2], r[2]) - max(b[0], r[0])) * max(0, min(b[3], r[3]) - max(b[1], r[1]))
            ra = max(1e-6, (r[2] - r[0]) * (r[3] - r[1]))
            if (i / ga >= 0.5 or i / ra >= 0.8) and i > best:
                best, bi = i, k
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


def dedupe(dets):
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(addons.inter(d["box"], k["box"]) < 0.8 * min(addons.area(d["box"]), addons.area(k["box"])) for k in keep):
            keep.append(d)
    return keep


def safe(f):
    try:
        return f()
    except Exception as e:
        print("method error", type(e).__name__, str(e)[:120])
        return None


def pipeline(pages, rects):
    import models
    MODELS = ["heron", "heron_int8", "egret", "ppl", "ppv3"]
    dets = {m: {} for m in MODELS}
    for m in MODELS:
        for r in jl(f"layout-{m}-*.jsonl"):
            if r["ds"] in ("odb", "drdoc"):
                dets[m][r["id"]] = r
    dl = {r["id"]: r for r in jl("dlrs-*.jsonl") if r.get("ds") in ("odb", "drdoc") and "items" in r}
    out = []
    for p in pages:
        W, H = p["W"], p["H"]
        rec = {k: p[k] for k in ("id", "ds", "split", "layout", "source", "bucket", "tags")}
        rs = rects.get(p["id"], [])
        for m in MODELS:
            r = dets[m].get(p["id"])
            if not r:
                continue
            items = []
            for d in r["dets"]:
                if d[1] < 0.5:
                    continue
                u = models.unify(m, d[0], d[3], d[5], H)
                if not u or u in ("Page-header", "Page-footer"):
                    continue
                items.append(dict(label=models.TO_RO[u], score=d[1], box=d[2:6], order=d[6] if len(d) > 6 else None))
            items = dedupe(items)
            alli = dedupe([dict(label=d[0], score=d[1], box=d[2:6]) for d in r["dets"] if d[1] >= 0.5])
            if alli:
                pl = [d["label"] if m in ("ppl", "ppv3") else pdx_order.FROM_DOCLING.get(d["label"], "text") for d in alli]
                o = safe(lambda: pdx_order.order([d["box"] for d in alli], pl, [""] * len(alli), W, H)[0])
                if o is not None:
                    s = evaluate(p, [d["box"] for d in alli], o)
                    if s:
                        rec[f"{m}:pdx_xycut_enhanced"] = s
            if not items:
                continue
            regs = [i["box"] for i in items]
            labs = [i["label"] for i in items]
            meth = {f"{m}:xycutpp": safe(lambda: ro.xycutpp(regs, W, H)),
                    f"{m}:xycutpp+fig+box+tiny": safe(lambda: addons.order(regs, labs, [""] * len(regs), W, H, rs, {"fig", "box", "tiny"})),
                    f"{m}:docling_rb": safe(lambda: ro.docling_rb(regs, labs, W, H))}
            if m == "ppv3":
                meth["ppv3:own_order"] = sorted(range(len(items)), key=lambda i: items[i]["order"])
            for k, o in meth.items():
                if o is None:
                    continue
                s = evaluate(p, regs, o)
                if s:
                    rec[k] = s
        r = dl.get(p["id"])
        if r and r["items"]:
            alli = r["items"]
            o = safe(lambda: pdx_order.order([i["box"] for i in alli], [pdx_order.FROM_DOCLING.get(i["label"], "text") for i in alli],
                                             [i.get("text", "") for i in alli], W, H)[0])
            if o is not None:
                s = evaluate(p, [i["box"] for i in alli], o)
                if s:
                    rec["dlrs:pdx_xycut_enhanced"] = s
        if r:
            its = [i for i in r["items"] if not i["furniture"] and i["label"] not in ("page_header", "page_footer")]
            if its:
                regs = [i["box"] for i in its]
                labs = [i["label"] for i in its]
                txt = [i.get("text", "") for i in its]
                for k, o in {"dlrs:docling_rs_order": list(range(len(its))), "dlrs:xycutpp": safe(lambda: ro.xycutpp(regs, W, H)),
                             "dlrs:xycutpp+all": safe(lambda: addons.order(regs, labs, txt, W, H, rs, {"fig", "box", "tiny", "num"}))}.items():
                    if o is None:
                        continue
                    s = evaluate(p, regs, o)
                    if s:
                        rec[k] = s
        out.append(rec)
    return out


def main():
    P("## Part 2: reading order")
    allrecs = []
    for ds, title in (("odb", "OmniDocBench English"), ("drdoc", "DrDocBench dev (English)")):
        recs = merged(ds)
        if not recs:
            continue
        for r in recs:
            r["ds"] = ds
        allrecs += recs
        base = ["docling_rb", "pdx_xycut_enhanced", "xycut_xfirst", "xycutpp", "xycutpp+fig", "xycutpp+box", "xycutpp+num", "xycutpp+tiny", "xycutpp+num+tiny", "xycutpp+all"]
        table(recs, base, f"{title}: ground-truth body boxes ({len(recs)} pages)",
              "cells: mean Kendall tau / mean normalised edit distance / % pages fully correct")
        full = ["full:docling_rb", "full:pdx_xycut_enhanced", "full:xycutpp", "full:xycutpp+deco", "full:xycutpp+all-deco", "full:xycutpp+all"]
        table([r for r in recs if "full:xycutpp" in r], full, f"{title}: all boxes incl. headers/footers/page numbers (no labels for XY-Cut++; docling gets labels), scored on body",
              "cells: tau / edit distance / % exact. deco = repeated-decoration removal")
    if allrecs:
        P("\n### Add-on gains (pages only A fully correct vs only B; sign test), both datasets")
        P("| A vs B | subset | pages | A only | B only | p | mean d-tau | d-exact (pts) |")
        P("|---|---|---|---|---|---|---|---|")
        pairs = [(a, "xycutpp") for a in ("xycutpp+fig", "xycutpp+box", "xycutpp+num", "xycutpp+tiny", "xycutpp+num+tiny", "xycutpp+all")]
        pairs += [("full:xycutpp+deco", "full:xycutpp"), ("full:xycutpp+all", "full:xycutpp"), ("full:xycutpp+all", "full:xycutpp+all-deco"),
                  ("xycutpp", "docling_rb"), ("xycutpp+all", "docling_rb"), ("xycutpp", "xycut_xfirst"),
                  ("pdx_xycut_enhanced", "xycutpp"), ("pdx_xycut_enhanced", "docling_rb"), ("xycutpp+all", "pdx_xycut_enhanced"),
                  ("full:pdx_xycut_enhanced", "full:xycutpp"), ("full:xycutpp+all", "full:pdx_xycut_enhanced")]
        for a, b in pairs:
            for sub, rs in (("all", allrecs), ("test", [r for r in allrecs if r["split"] == "test"]),
                            ("complex+edge", [r for r in allrecs if r["bucket"] == "complex" or datasets.edge(r["tags"])])):
                w, l, pv, dt, ex, n = sign(rs, a, b)
                RES.setdefault("sign", []).append(dict(a=a, b=b, subset=sub, n=n, a_only=w, b_only=l, p=pv, dtau=dt, dexact=ex))
                P(f"| {a} vs {b} | {sub} | {n} | {w} | {l} | {pv:.2g} | {dt:+.4f} | {ex:+.1f} |")
        table(allrecs, ["docling_rb", "pdx_xycut_enhanced", "xycutpp", "xycutpp+all", "full:pdx_xycut_enhanced", "full:xycutpp+all"], f"Both datasets together ({len(allrecs)} pages)", "cells: tau / ed / % exact")
    rects = {r["id"]: r.get("rects", []) for r in allrecs}
    pages = datasets.pages("odb") + datasets.pages("drdoc")
    pl = pipeline(pages, rects)
    pl = [r for r in pl if any(":" in k for k in r)]
    if pl:
        ms = ["dlrs:docling_rs_order", "dlrs:xycutpp", "dlrs:xycutpp+all", "heron:xycutpp", "heron:xycutpp+fig+box+tiny", "heron_int8:xycutpp",
              "egret:xycutpp", "egret:xycutpp+fig+box+tiny", "ppl:xycutpp", "ppl:xycutpp+fig+box+tiny", "ppv3:xycutpp", "ppv3:xycutpp+fig+box+tiny",
              "ppv3:own_order", "ppv3:pdx_xycut_enhanced", "heron:pdx_xycut_enhanced", "egret:pdx_xycut_enhanced", "ppl:pdx_xycut_enhanced",
              "dlrs:pdx_xycut_enhanced", "heron:docling_rb", "egret:docling_rb", "ppl:docling_rb", "ppv3:docling_rb"]
        ms = [m for m in ms if any(m in r for r in pl)]
        for ds in ("odb", "drdoc"):
            rs = [r for r in pl if r["ds"] == ds]
            table(rs, ms, f"Real pipeline on {ds}: order on each layout model's boxes (score>=0.5, page header/footer removed by the model's label) ({len(rs)} pages)",
                  "cells: tau / ed / % exact / % GT body regions matched to a detected box; order scored on matched regions")
        P("\n### Real pipeline gaps (sign tests, both datasets)")
        P("| A vs B | pages | A only | B only | p | mean d-tau | d-exact (pts) |")
        P("|---|---|---|---|---|---|---|")
        for a, b in [("ppv3:xycutpp", "ppv3:own_order"), ("ppv3:xycutpp+fig+box+tiny", "ppv3:own_order"), ("dlrs:xycutpp", "dlrs:docling_rs_order"),
                     ("dlrs:xycutpp+all", "dlrs:docling_rs_order"), ("egret:xycutpp", "ppv3:own_order"), ("heron:xycutpp", "ppv3:own_order"),
                     ("ppl:xycutpp", "ppv3:own_order"), ("egret:xycutpp+fig+box+tiny", "ppv3:own_order"),
                     ("ppv3:pdx_xycut_enhanced", "ppv3:own_order"), ("ppv3:xycutpp", "ppv3:pdx_xycut_enhanced"),
                     ("heron:pdx_xycut_enhanced", "heron:xycutpp"), ("dlrs:pdx_xycut_enhanced", "dlrs:xycutpp")]:
            w, l, pv, dt, ex, n = sign(pl, a, b)
            RES.setdefault("pipeline_sign", []).append(dict(a=a, b=b, n=n, a_only=w, b_only=l, p=pv, dtau=dt, dexact=ex))
            P(f"| {a} vs {b} | {n} | {w} | {l} | {pv:.2g} | {dt:+.4f} | {ex:+.1f} |")
    P("\n### Logic cost per page (GT boxes, from the ro jobs; shared runners, indicative)")
    for m in ("docling_rb", "pdx_xycut_enhanced", "xycutpp", "xycutpp+all", "full:xycutpp+all"):
        v = [r[m]["ms"] for r in allrecs if m in r and "ms" in r[m]]
        if v:
            v.sort()
            P(f"- {m}: median {statistics.median(v):.2f} ms, p95 {v[int(0.95 * (len(v) - 1))]:.2f} ms")
    v = sorted(r["rect_ms"] for r in allrecs if "rect_ms" in r)
    if v:
        P(f"- boxed-section detection from the page image: median {statistics.median(v):.0f} ms, p95 {v[int(0.95 * (len(v) - 1))]:.0f} ms")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(dict(RES, pipeline_pages=len(pl)), open(OUTJ, "w"), indent=1, default=str)
    with open(os.path.join(os.path.dirname(OUT), "ro_records.jsonl"), "w") as f:
        for r in allrecs:
            f.write(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "pred"}) for k, v in r.items()}) + "\n")


if __name__ == "__main__":
    main()
