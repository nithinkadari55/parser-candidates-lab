import os, sys, json, glob, statistics, random

MODELS = os.environ.get("AGG_MODELS", "tatr32,tatr16,tatr,tatr4,potatr,potatr8,tcn,tfv2,tfv2q,slanet1m,tableformer").split(",")
SETS = ["ptn", "ftn", "pt1m", "pt2", "icdar", "scitsr", "tabrec", "rot"]
MAIN = ["ptn", "ftn", "pt1m"]
BUCKETS = ["simple", "medium", "complex", "edge"]
TAGS = ["ruled", "borderless", "multirow_header", "nested_header", "hier_first_col", "wide", "long", "very_wide", "very_long", "many_spans",
        "multiline", "empty_cells", "footnote_marks", "tiny", "split_page", "rotated", "rotated_page"]
H = ("| model | set | n | TEDS (n) | TEDS-S | GriTS-Top (n) | GriTS-Con | rows+cols exact % | merged-row errs (tables %) | header % (n) | fails (no output) |\n"
     "|---|---|---|---|---|---|---|---|---|---|---|")


def load(root, m, s, scored=True):
    d = {}
    pat = f"{m}_{s}_s*.scored.jsonl" if scored else f"{m}_{s}_s*.jsonl"
    for p in sorted(glob.glob(os.path.join(root, "**", pat), recursive=True)):
        if not scored and p.endswith(".scored.jsonl"): continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); d[r["id"]] = r
    return d or None


def mean(v): return sum(v) / len(v) if v else None


def summ(rs):
    n = len(rs)
    if not n: return None
    t = [r["teds"] for r in rs if r.get("teds") is not None]; ts = [r["teds_s"] for r in rs if r.get("teds_s") is not None]
    g = [r["grits_top"] for r in rs if r.get("grits_top") is not None]; gc = [r["grits_con"] for r in rs if r.get("grits_con") is not None]
    h = [r["header_ok"] for r in rs if r.get("header_ok") is not None]
    mr = [r for r in rs if r.get("merged_row_err") is not None]
    return dict(n=n, teds=mean(t), nt=len(t), teds_s=mean(ts), grits_top=mean(g), ng=len(g), grits_con=mean(gc),
                rc=100 * sum(r["rc_exact"] for r in rs) / n, merged=(sum(r["merged_row_err"] for r in mr) if mr else None),
                merged_tab=(100 * sum(r["merged_row_err"] > 0 for r in mr) / len(mr) if mr else None),
                hdr=(100 * sum(h) / len(h) if h else None), nh=len(h), fail=sum(r["fail"] for r in rs), empty=sum(1 for r in rs if not r.get("n_cells")))


def fmt(m, s, x):
    f = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"
    hd = "n/a" if x["hdr"] is None else f"{x['hdr']:.1f} ({x['nh']})"
    mg = "n/a" if x["merged"] is None else f"{x['merged']} ({x['merged_tab']:.1f}%)"
    return (f"| {m} | {s} | {x['n']} | {f(x['teds'])} ({x['nt']}) | {f(x['teds_s'])} | {f(x['grits_top'])} ({x['ng']}) | {f(x['grits_con'])} | {x['rc']:.1f} | "
            f"{mg} | {hd} | {x['fail']} ({x['empty']}) |")


def key(r):
    return r["teds_s"] if r.get("teds_s") is not None else r.get("grits_top")


def paired(a, b, ids):
    from scipy.stats import wilcoxon
    pairs = []
    for i in ids:
        for k in ("teds_s", "grits_top"):
            if a[i].get(k) is not None and b[i].get(k) is not None:
                pairs.append((a[i][k], b[i][k])); break
    if len(pairs) < 10: return None
    d = [x - y for x, y in pairs]
    rng = random.Random(0); boots = []
    for _ in range(2000):
        smp = [d[rng.randrange(len(d))] for _ in d]; boots.append(sum(smp) / len(smp))
    boots.sort()
    try: p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs], zero_method="zsplit").pvalue
    except ValueError: p = 1.0
    return dict(n=len(d), diff=sum(d) / len(d), lo=boots[50], hi=boots[1949], p=p, wins=sum(1 for v in d if v > 1e-9), losses=sum(1 for v in d if v < -1e-9))


def struct(r):
    return sorted((c["r0"], c["r1"], c["c0"], c["c1"]) for c in (r.get("cells") or []))


def main():
    root = sys.argv[1]; out = []; P = out.append
    data = {(m, s): load(root, m, s) for m in MODELS for s in SETS}
    have = [m for m in MODELS if any(data[(m, s)] for s in SETS)]

    def common(s, ms=None):
        ds = [data[(m, s)] for m in (ms or have) if data[(m, s)]]
        return set.intersection(*(set(d) for d in ds)) if ds else set()
    def cell(rs, f):
        x = summ(rs) if rs else None
        if not x: return "-"
        return f(x)
    for title, f in (("TEDS-Struct (GriTS-Top where TEDS is capped)", lambda x: f"{(x['teds_s'] if x['teds_s'] is not None else x['grits_top'] or 0):.3f}"),
                     ("rows+cols exact %", lambda x: f"{x['rc']:.0f}"), ("tables with merged-row errors %", lambda x: "n/a" if x["merged_tab"] is None else f"{x['merged_tab']:.0f}")):
        P(f"## 0. Compact: {title}, per dataset/bucket (all tables each model has)")
        P("| set/bucket | n | " + " | ".join(have) + " |\n|" + "---|" * (len(have) + 2))
        for s in SETS:
            for b in [None] + BUCKETS:
                rows_ = {m: [r for r in (data[(m, s)] or {}).values() if b is None or r["bucket"] == b] for m in have}
                n = max((len(v) for v in rows_.values()), default=0)
                if n: P(f"| {s}/{b or 'all'} | {n} | " + " | ".join(cell(rows_[m], f) for m in have) + " |")
        P("")
    P("## 0b. Compact per tag, TEDS-Struct (GriTS-Top where capped), all datasets pooled")
    P("| tag | n | " + " | ".join(have) + " |\n|" + "---|" * (len(have) + 2))
    for t in TAGS:
        rows_ = {m: [r for s in SETS for r in (data[(m, s)] or {}).values() if r["tags"].get(t)] for m in have}
        n = max((len(v) for v in rows_.values()), default=0)
        if n: P(f"| {t} | {n} | " + " | ".join(cell(rows_[m], lambda x: f"{(x['teds_s'] if x['teds_s'] is not None else x['grits_top'] or 0):.3f}") for m in have) + " |")
    P("")
    P("## 1. Per dataset (all tables each model has)\n" + H)
    for s in SETS:
        for m in have:
            if data[(m, s)]: P(fmt(m, s, summ(list(data[(m, s)].values()))))
    P("\n## 2. Per dataset and difficulty bucket\n" + H)
    for s in SETS:
        for b in BUCKETS:
            for m in have:
                d = data[(m, s)]
                if d:
                    rs = [r for r in d.values() if r["bucket"] == b]
                    if rs: P(fmt(m, f"{s}/{b}", summ(rs)))
    P("\n## 3. Per tag (all datasets pooled, ids every model has)\n" + H)
    for t in TAGS:
        for m in have:
            rs = []
            for s in SETS:
                if data[(m, s)]: rs += [data[(m, s)][i] for i in common(s) if data[(m, s)][i]["tags"].get(t)]
            if rs: P(fmt(m, "tag:" + t, summ(rs)))
    P("\n## 4. Main datasets pooled (PubTabNet + FinTabNet + PubTables-1M), per bucket\n" + H)
    for b in [None] + BUCKETS:
        for m in have:
            rs = []
            for s in MAIN:
                if data[(m, s)]: rs += [data[(m, s)][i] for i in common(s) if b is None or data[(m, s)][i]["bucket"] == b]
            if rs: P(fmt(m, "main/" + (b or "all"), summ(rs)))
    P("\n## 5. Only tables where docling's layout found a table (TableFormer produced cells)\n" + H)
    for s in SETS:
        tf = data.get(("tableformer", s))
        if not tf: continue
        ids = {i for i, r in tf.items() if r.get("n_cells")}
        P(f"| {s}: docling found a table on {len(ids)} of {len(tf)} | | | | | | | | | | |")
        for m in have:
            d = data[(m, s)]
            if d: P(fmt(m, f"{s} tf-found", summ([d[i] for i in ids if i in d])))
    P("\n## 6. Paired tests (TEDS-Struct per table, GriTS-Top where TEDS is capped): mean difference A-B, bootstrap 95% CI, Wilcoxon p, wins/losses")
    pooled = {}
    for m in have:
        v = []
        for s in MAIN:
            if data[(m, s)]: v += [key(data[(m, s)][i]) for i in common(s) if key(data[(m, s)][i]) is not None]
        if v: pooled[m] = sum(v) / len(v)
    rank = sorted(pooled, key=lambda m: -pooled[m])
    P("Ranking on main datasets (mean TEDS-S over ids all models have): " + ", ".join(f"{m} {pooled[m]:.4f}" for m in rank))
    pairs = []
    if len(rank) >= 2: pairs.append((rank[0], rank[1]))
    for m in rank:
        if m != "tableformer" and "tableformer" in rank: pairs.append((m, "tableformer"))
    for v in ("tatr32", "tatr16", "tatr4"):
        if v in have and "tatr" in have: pairs.append((v, "tatr"))
    for a_, b_ in (("potatr8", "potatr"), ("tfv2q", "tfv2")):
        if a_ in have and b_ in have: pairs.append((a_, b_))
    seen = set()
    for a_, b_ in pairs:
        if (a_, b_) in seen: continue
        seen.add((a_, b_))
        for s in SETS + ["main"]:
            ss = MAIN if s == "main" else [s]
            A, B, ids = {}, {}, []
            for x in ss:
                da, db = data.get((a_, x)), data.get((b_, x))
                if not da or not db: continue
                for i in set(da) & set(db): A[i] = da[i]; B[i] = db[i]; ids.append(i)
            r = paired(A, B, ids) if ids else None
            if r: P(f"- {a_} vs {b_} [{s}]: n={r['n']} diff={r['diff']:+.4f} CI [{r['lo']:+.4f}, {r['hi']:+.4f}] p={r['p']:.2g} wins/losses {r['wins']}/{r['losses']}")
        if "tableformer" in (a_, b_):
            for s in SETS:
                tf = data.get(("tableformer", s)); da, db = data.get((a_, s)), data.get((b_, s))
                if not tf or not da or not db: continue
                ids = [i for i, r_ in tf.items() if r_.get("n_cells") and i in da and i in db]
                r = paired(da, db, ids)
                if r: P(f"- {a_} vs {b_} [{s}, tf-found only]: n={r['n']} diff={r['diff']:+.4f} CI [{r['lo']:+.4f}, {r['hi']:+.4f}] p={r['p']:.2g} wins/losses {r['wins']}/{r['losses']}")
    P("\n## 7. Precision variants vs their full-precision reference: tables whose predicted structure (cell grid) differs")
    for ref, v in (("tatr32", "tatr16"), ("tatr32", "tatr"), ("tatr32", "tatr4"), ("potatr", "potatr8"), ("tfv2", "tfv2q")):
        for s in SETS:
            a = load(root, ref, s, False); b = load(root, v, s, False)
            if not a or not b: continue
            ids = set(a) & set(b)
            ch = sum(1 for i in ids if struct(a[i]) != struct(b[i]))
            P(f"- {v} vs {ref} [{s}]: {ch} of {len(ids)} tables changed ({100 * ch / max(1, len(ids)):.1f}%)")
    P("\n## 8. Lowest TEDS-Struct per model (ids only)")
    for m in have:
        rs = []
        for s in SETS:
            if data[(m, s)]: rs += list(data[(m, s)].values())
        rs.sort(key=lambda r: (key(r) if key(r) is not None else 9))
        P(f"### {m}")
        for r in rs[:10]:
            P(f"- {r['id']} [{r['bucket']}] S={key(r) if key(r) is None else round(key(r), 2)} gt={r['gt_rows']}x{r['gt_cols']} pred={r.get('rows')}x{r.get('cols')} "
              f"cells={r.get('n_cells')} merged={r.get('merged_row_err')} {r.get('err') or ''} tags={[k for k, v in r['tags'].items() if v]}")
    P("\n## 9. Accuracy-run latency per table (shared matrix runners; clean numbers come from the timing job)")
    for m in have:
        v = sorted(r["ms_total"] for s in SETS if data[(m, s)] for r in data[(m, s)].values() if r.get("ms_total"))
        if v: P(f"- {m}: median {statistics.median(v):.0f} ms, p95 {v[int(0.95 * (len(v) - 1))]:.0f} ms, n={len(v)}")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(root, "report.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
