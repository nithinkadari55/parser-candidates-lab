"""Milestone 14: the PP-LCNet confidence gate for scanned tables, chosen on the dev split of the licence-allowed sets from
the round-8 per-table files (meta.jsonl of run 45676430603, artifacts r8pub-*; no models are run here).

Per-table fields used (tatr8.py): on the answer-key crop of every scan table, gt_lcnet / gt_lcnet_conf (PP-LCNet on the
upright crop) and the scores gt_tatr (OCR + TATR on the crop as is) and gt_tatr_lcnet (on the crop turned by PP-LCNet's
label); for half of them the crop turned by rs_theta, with rs_lcnet / rs_lcnet_conf and the scores rs_none (as turned) and
rs_lcnet (after PP-LCNet's turn). On docling's crops of every found table, lcnet / lcnet_conf (rotpdf rows also carry
truth_turn).

Rule, fixed before looking at the test split: the gate is the lowest value on GRID such that, on the dev split of the
sets allowed for training (licences14.TRAIN),
  (a) upright scanned answer-key crops do not lose on average (TEDS-Struct change with the gated turn >= 0), and
  (b) PP-LCNet turns at most MAX_FALSE_TURN of the upright docling crops (all allowed sets except rotated pages:
      rendered table images and scans; upright by construction).
Evaluation-only sets (WTW, cTDaR, ICDAR-2013) never choose anything; they are reported on the test split with the rest.

usage: gate14.py PUB_ROOT OUT_DIR"""
import os, sys, json, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from licences14 import TRAIN, EVAL_ONLY, source_set

GRID = [round(0.5 + 0.05 * k, 2) for k in range(9)]
MAX_FALSE_TURN = 0.005
PREVIOUS = 0.8


def value(s):
    if not s or s.get("score_err"): return None
    return s["teds_s"] if s.get("teds_s") is not None else s.get("grits_top")


def load(pub_root):
    rows = []
    for p in sorted(glob.glob(os.path.join(pub_root, "**", "meta.jsonl"), recursive=True)):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def groups(rows, split, sets):
    up = [r for r in rows if r["split"] == split and r["set"] in sets and "gt_lcnet" in r
          and value(r["scores"].get("gt_tatr")) is not None and value(r["scores"].get("gt_tatr_lcnet")) is not None]
    rs = [r for r in rows if r["split"] == split and r["set"] in sets and "rs_theta" in r
          and value(r["scores"].get("rs_none")) is not None and value(r["scores"].get("rs_lcnet")) is not None]
    dl = [r for r in rows if r["split"] == split and r["set"] in sets and r["set"] != "rotpdf" and r.get("found")
          and not r.get("err") and "lcnet" in r]
    rot = [r for r in rows if r["split"] == split and r["set"] == "rotpdf" and r.get("found") and not r.get("err")
           and "lcnet" in r and "truth_turn" in r and source_set(r["id"], r["set"]) in sets]
    return up, rs, dl, rot


def mean(xs): return sum(xs) / len(xs) if xs else None


def at(rows, split, sets, g):
    up, rs, dl, rot = groups(rows, split, sets)
    on = lambda lab, conf: bool(lab) and conf >= g
    return dict(
        g=g,
        n_upright=len(up), n_turned=len(rs), n_docling=len(dl), n_rotpdf=len(rot),
        upright_change=mean([(value(r["scores"]["gt_tatr_lcnet"]) if on(r["gt_lcnet"], r["gt_lcnet_conf"]) else value(r["scores"]["gt_tatr"]))
                             - value(r["scores"]["gt_tatr"]) for r in up]),
        upright_turned=mean([1.0 if on(r["gt_lcnet"], r["gt_lcnet_conf"]) else 0.0 for r in up]),
        turned_change=mean([(value(r["scores"]["rs_lcnet"]) if on(r["rs_lcnet"], r["rs_lcnet_conf"]) else value(r["scores"]["rs_none"]))
                            - value(r["scores"]["rs_none"]) for r in rs]),
        turned_fixed=mean([1.0 if on(r["rs_lcnet"], r["rs_lcnet_conf"]) and r["rs_lcnet"] == r["rs_need"] else 0.0 for r in rs]),
        docling_false_turn=mean([1.0 if on(r["lcnet"], r["lcnet_conf"]) else 0.0 for r in dl]),
        rotpdf_fixed=mean([1.0 if on(r["lcnet"], r["lcnet_conf"]) and r["lcnet"] == r["truth_turn"] else 0.0 for r in rot]))


def choose(rows):
    allowed = set(TRAIN)
    curve = [at(rows, "dev", allowed, g) for g in GRID]
    ok = [c for c in curve if c["upright_change"] is not None and c["upright_change"] >= 0.0
          and c["docling_false_turn"] is not None and c["docling_false_turn"] <= MAX_FALSE_TURN]
    return (ok[0]["g"] if ok else GRID[-1]), curve


def fmt(c):
    f = lambda x, k=100, d=2: "–" if x is None else f"{k * x:+.{d}f}"
    p = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    return (f"| {c['g']:.2f} | {c['n_upright']} | {f(c['upright_change'])} | {p(c['upright_turned'])} | {c['n_turned']} | "
            f"{f(c['turned_change'])} | {p(c['turned_fixed'])} | {c['n_docling']} | {p(c['docling_false_turn'])} | {c['n_rotpdf']} | {p(c['rotpdf_fixed'])} |")


HEAD = ["| gate | upright scan crops | TEDS-S change (pts) | turned | turned crops | TEDS-S change (pts) | fixed | upright docling crops | turned (false) | rotated text pages | fixed |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]


def main():
    pub_root, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    rows = load(pub_root)
    g, curve = choose(rows)
    allowed, every = set(TRAIN), set(TRAIN) | set(EVAL_ONLY) | {"rotpdf"}
    test = {name: [at(rows, "test", sets, x) for x in sorted({g, PREVIOUS})]
            for name, sets in (("allowed sets", allowed), ("evaluation-only sets", set(EVAL_ONLY)), ("all sets", every))}
    per_set = {s: [at(rows, "test", {s}, x) for x in sorted({g, PREVIOUS})] for s in sorted(allowed | set(EVAL_ONLY)) if groups(rows, "test", {s})[0] or groups(rows, "test", {s})[2]}
    json.dump(dict(gate=g, grid=GRID, max_false_turn=MAX_FALSE_TURN, dev_curve_allowed=curve, test=test, test_per_set=per_set),
              open(os.path.join(out, "gate14.json"), "w"), indent=1)
    md = ["# PP-LCNet gate for scanned tables (gate14.py)", "",
          f"Chosen on the dev split of the licence-allowed sets: **{g:.2f}** (lowest gate with no average loss on upright scanned crops and at most {100 * MAX_FALSE_TURN:.1f}% of upright docling crops turned).", "",
          "## Dev split, allowed sets (choice)", ""] + HEAD + [fmt(c) for c in curve]
    for name, cs in test.items():
        md += ["", f"## Test split, {name}", ""] + HEAD + [fmt(c) for c in cs]
    md += ["", "## Test split per set", "", "| set | " + HEAD[0][2:], "|---" + HEAD[1]]
    for s, cs in per_set.items():
        md += [f"| {s} " + fmt(c) for c in cs]
    open(os.path.join(out, "gate14.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
