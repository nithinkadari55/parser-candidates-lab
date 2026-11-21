"""Gap 13 report. usage: agg_r4.py ARTDIR OUT.md OUT.json"""
import glob
import json
import os
import statistics
import sys

from scipy.stats import binomtest

ART, OUT, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3]
MARGINS = [None, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
L = []


def P(s=""):
    L.append(s)


def load():
    R = []
    for f in sorted(glob.glob(os.path.join(ART, "**", "r4-*.jsonl"), recursive=True)):
        for line in open(f, encoding="utf-8"):
            R.append(json.loads(line))
    return R


def pick(entry, m):
    """-> (exact, tau, ed) of the chosen candidate; None if unscorable"""
    c = entry["c"]
    if not c or c[0] is None or c[0][1] is None:
        return None
    j = 0
    if m is not None:
        best = max(range(len(c)), key=lambda k: (c[k][0] if c[k] and c[k][1] is not None else -1e9))
        if c[best] and c[best][1] is not None and c[best][0] - c[0][0] >= m:
            j = best
    r = c[j]
    return {"tau": r[1], "ed": r[2], "exact": r[3], "changed": j != 0}


def rate(recs, mode, cfg, m):
    v = [pick(r[mode][cfg], m) for r in recs if mode in r and cfg in r[mode]]
    v = [x for x in v if x]
    if not v:
        return None, 0, None
    return 100 * statistics.mean(x["exact"] for x in v), len(v), statistics.mean(x["tau"] for x in v)


def sign(recs, mode, cfg, m, base=("base", None)):
    w = l = 0
    for r in recs:
        if mode not in r:
            continue
        a, b = pick(r[mode][cfg], m), pick(r[mode][base[0]], base[1])
        if not a or not b:
            continue
        w += a["exact"] and not b["exact"]
        l += b["exact"] and not a["exact"]
    return w, l, (binomtest(w, w + l).pvalue if w + l else 1.0)


def main():
    R = load()
    dev = [r for r in R if r["split"] == "dev"]
    test = [r for r in R if r["split"] == "test"]
    SL = [("hard (newspaper, magazine, colourful textbook, exam)", lambda r: r.get("hard_src")),
          ("  newspaper + magazine", lambda r: r.get("source") in ("newspaper", "magazine")),
          ("  colourful textbook", lambda r: r.get("source") == "colorful_textbook"),
          ("  exam", lambda r: r.get("source") == "exam_paper"),
          ("boxed or shaded region (any source)", lambda r: "boxed_or_shaded_region" in r["tags"]),
          ("ordinary (everything else)", lambda r: not r.get("hard_src")),
          ("all", lambda r: True)]
    res = {"pages": len(R), "test": len(test)}
    P("# Round 4, gap 13: newspaper, magazine and sidebar reading order")
    P()
    P(f"Pages: {len(R)} (OmniDocBench English + DrDocBench dev); test split {len(test)} (document-hashed split as in "
      f"rounds 1-3; margins and policies chosen on the dev split only). Current pick = candidate 0 of `base` "
      f"(full: XY-Cut++ + repeated-decoration removal on GT boxes; h8: XY-Cut++ on heron int8 boxes).")
    P()
    chosen = {}
    for mode in ("full", "h8"):
        devh = [r for r in dev if r.get("hard_src")]
        devo = [r for r in dev if not r.get("hard_src")]
        for cfg in ("base", "side_inline", "side_after", "side_end", "grid", "jump", "all_inline", "all_after", "all_end"):
            bo = rate(devo, mode, "base", None)[0]
            best = None
            for m in MARGINS:
                h = rate(devh, mode, cfg, m)[0]
                o = rate(devo, mode, cfg, m)[0]
                if h is None or o is None or bo is None:
                    continue
                if o < bo - 0.5:
                    continue
                key = (h, 99.0 if m is None else m)
                if best is None or key > best[0]:
                    best = (key, m)
            chosen[(mode, cfg)] = best[1] if best else None
        res[mode] = {}
        P(f"## Box source: {'GT boxes (full page incl. furniture)' if mode == 'full' else 'heron int8 detections'}")
        P()
        P("Fully correct pages, % (test split). `+cont` uses the continuity margin chosen on dev (shown).")
        P()
        cols = []
        for cfg in ("base", "side_inline", "side_after", "side_end", "grid", "jump", "all_inline", "all_after", "all_end"):
            cols.append((cfg, None))
            if chosen[(mode, cfg)] is not None:
                cols.append((cfg, chosen[(mode, cfg)]))
        P("| slice | pages | " + " | ".join(f"{c}{'' if m is None else f' +cont({m})'}" for c, m in cols) + " |")
        P("|---|---|" + "---|" * len(cols))
        for name, f in SL:
            rs = [r for r in test if f(r)]
            vals = [rate(rs, mode, c, m) for c, m in cols]
            n = vals[0][1]
            P(f"| {name} | {n} | " + " | ".join("-" if v[0] is None else f"{v[0]:.1f}" for v in vals) + " |")
            res[mode][name] = {f"{c}|{m}": v[0] for (c, m), v in zip(cols, vals)}
        P()
        P("Sign tests vs the current pick (test split; pages only the variant gets fully right vs only the pick):")
        P()
        P("| variant | slice | variant only | pick only | p | d-exact pts |")
        P("|---|---|---|---|---|---|")
        for c, m in cols[1:]:
            for name, f in (SL[0], SL[4], SL[5], SL[6]):
                rs = [r for r in test if f(r)]
                w, l, pv = sign(rs, mode, c, m)
                a, b = rate(rs, mode, c, m)[0], rate(rs, mode, "base", None)[0]
                if a is None or b is None:
                    continue
                P(f"| {c}{'' if m is None else f' +cont({m})'} | {name.strip()} | {w} | {l} | {pv:.2g} | {a - b:+.1f} |")
        P()
        trig = {k: 0 for k in ("sidebars", "grid", "jumps")}
        changed = 0
        npg = 0
        for r in test:
            if mode not in r:
                continue
            npg += 1
            for k in trig:
                trig[k] += any(r[mode][c]["trig"][k] for c in r[mode])
            x = pick(r[mode]["base"], chosen[(mode, "base")])
            changed += bool(x and x["changed"])
        P(f"Triggers on {npg} test pages: sidebar units on {trig['sidebars']}, grid numbering applied on {trig['grid']}, "
          f"continuation lines on {trig['jumps']}; continuity choice changed the base order on {changed}.")
        ms = sorted(r[mode]["all_after"]["ms"] for r in R if mode in r)
        if ms:
            P(f"Cost of the full add-on stack (all_after, all candidates): median {statistics.median(ms):.2f} ms, "
              f"p95 {ms[int(0.95 * (len(ms) - 1))]:.2f} ms per page.")
        P()
    pr = glob.glob(os.path.join(ART, "**", "american_stories_probe.txt"), recursive=True)
    if pr:
        P("## American Stories")
        P()
        P(open(pr[0], encoding="utf-8").read())
    res["chosen"] = {f"{a}:{b}": v for (a, b), v in chosen.items()}
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(res, open(OUTJ, "w"), indent=1)


if __name__ == "__main__":
    main()
