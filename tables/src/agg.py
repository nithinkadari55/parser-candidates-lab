import os, sys, json, glob, statistics

MODELS = [m for m in os.environ.get("AGG_MODELS", "tatr,slanet1m,tableformer,tatr-trim,slanet1m-trim").split(",")]
SETS = ["ptn", "ftn", "pt1m", "icdar"]
BUCKETS = ["simple", "medium", "edge", "complex"]
TAGS = ["ruled", "borderless", "multirow_header", "nested_header", "hier_first_col", "wide", "long", "many_spans", "multiline", "empty_cells", "footnote_marks", "tiny", "split_page"]
H = ("| model | set | n | TEDS | TEDS-S | GriTS-Top | GriTS-Con | rows+cols exact % | merged-row errs (tables % / tables with fewer rows than GT %) | header % (n) | fails (no output) |\n"
     "|---|---|---|---|---|---|---|---|---|---|---|")


def load(root, m, s):
    d = {}
    for p in sorted(glob.glob(os.path.join(root, "**", f"{m}_{s}_s*.scored.jsonl"), recursive=True)):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); d[r["id"]] = r
    return d or None


def summ(rs):
    n = len(rs)
    if not n: return None
    g = [r["grits_top"] for r in rs if r.get("grits_top") is not None]
    gc = [r["grits_con"] for r in rs if r.get("grits_con") is not None]
    h = [r["header_ok"] for r in rs if r.get("header_ok") is not None]
    return dict(n=n, teds=sum(r["teds"] for r in rs) / n, teds_s=sum(r["teds_s"] for r in rs) / n,
                grits_top=(sum(g) / len(g) if g else None), ng=len(g), grits_con=(sum(gc) / len(gc) if gc else None),
                rc=100 * sum(r["rc_exact"] for r in rs) / n, merged=sum(r["merged_row_err"] for r in rs),
                merged_tab=100 * sum(r["merged_row_err"] > 0 for r in rs) / n,
                merged_struct=100 * sum(r["merged_row_err"] > 0 and (r.get("rows") or 0) < r["gt_rows"] for r in rs) / n,
                hdr=(100 * sum(h) / len(h) if h else None), nh=len(h), fail=sum(r["fail"] for r in rs), empty=sum(1 for r in rs if not r.get("n_cells")))


def fmt(m, s, x):
    f = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"
    hd = "n/a" if x["hdr"] is None else f"{x['hdr']:.1f} ({x['nh']})"
    return (f"| {m} | {s} | {x['n']} | {f(x['teds'])} | {f(x['teds_s'])} | {f(x['grits_top'])} | {f(x['grits_con'])} | {x['rc']:.1f} | "
            f"{x['merged']} ({x['merged_tab']:.1f}% / {x['merged_struct']:.1f}%) | {hd} | {x['fail']} ({x['empty']}) |")


def main():
    root = sys.argv[1]; out = []
    P = out.append
    data = {(m, s): load(root, m, s) for m in MODELS for s in SETS}

    def common(s):
        ds = [data[(m, s)] for m in MODELS if data[(m, s)] and "-" not in m]
        return set.intersection(*(set(d) for d in ds)) if ds else set()
    P("## All tables per model\n" + H)
    for s in SETS:
        for m in MODELS:
            if data[(m, s)]: P(fmt(m, s, summ(list(data[(m, s)].values()))))
    P("\n## Per difficulty bucket (all tables per model)\n" + H)
    for s in SETS:
        for b in BUCKETS:
            for m in MODELS:
                d = data[(m, s)]
                if d:
                    rs = [r for r in d.values() if r["bucket"] == b]
                    if rs: P(fmt(m, f"{s}/{b}", summ(rs)))
    P("\n## Paired: only ids every model has (should equal 'all' when all runs completed)\n" + H)
    for s in SETS:
        c = common(s)
        for m in MODELS:
            if data[(m, s)] and c: P(fmt(m, f"{s} paired", summ([data[(m, s)][i] for i in c])))
    P("\n## Per tag (all sets pooled, paired ids)\n" + H)
    for t in TAGS:
        for m in MODELS:
            rs = []
            for s in SETS:
                if data[(m, s)]: rs += [data[(m, s)][i] for i in common(s) if data[(m, s)][i]["tags"].get(t)]
            if rs: P(fmt(m, "tag:" + t, summ(rs)))
    P("\n## Accuracy-run latency per table (shared matrix runners; see timing job for clean numbers)")
    for m in MODELS:
        for s in SETS:
            d = data[(m, s)]
            if not d: continue
            v = sorted(r["ms_total"] for r in d.values() if r.get("ms_total"))
            if v: P(f"- {m} {s}: median {statistics.median(v):.0f} ms, p95 {v[int(0.95 * (len(v) - 1))]:.0f} ms")
    P("\n## Lowest TEDS-Struct per model (ids only)")
    for m in MODELS:
        rs = []
        for s in SETS:
            if data[(m, s)]: rs += list(data[(m, s)].values())
        rs.sort(key=lambda r: (r["teds_s"], r["teds"]))
        P(f"### {m}")
        for r in rs[:12]:
            P(f"- {r['id']} [{r['bucket']}] TEDS-S={r['teds_s']:.2f} TEDS={r['teds']:.2f} gt={r['gt_rows']}x{r['gt_cols']} pred={r.get('rows')}x{r.get('cols')} "
              f"cells={r.get('n_cells')} merged={r['merged_row_err']} {r.get('err') or ''} tags={[k for k, v in r['tags'].items() if v]}")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(root, "report.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
