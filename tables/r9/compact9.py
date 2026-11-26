"""Round 9: compact markdown tables from analyze9.py's summary.json (for the round report).
usage: compact9.py SUMMARY_JSON > compact.md"""
import sys, json

S = json.load(open(sys.argv[1]))
R = S["groups"]
NAMES = {"tf": "TableFormer", "tatr": "TATR", "combo": "combo", "tatr_c": "TATR (re-run)", "rapid_en": "SLANet en", "rapid_zh": "SLANet ch",
         "rapid_plus": "SLANet_plus", "paddle_slanet": "Paddle SLANet (=ch)", "paddle_slanet_plus": "Paddle SLANet_plus", "ae3": "AE3-TL",
         "mineru": "MinerU-4", "pptable": "PP-TableRec-v2", "tablemaster": "TableMaster*", "mutab_ptn": "MuTabNet-PTN*", "mutab_ftn": "MuTabNet-FTN*",
         "lore_wireless": "LORE wireless", "lore_ptn": "LORE PTN", "lore_wtw": "LORE WTW*", "trace_base": "TRACE base*", "trace_wtw": "TRACE WTW*"}
ORDER = ["tf", "tatr", "combo", "rapid_en", "rapid_zh", "rapid_plus", "paddle_slanet", "paddle_slanet_plus", "ae3", "mineru", "pptable",
         "lore_wireless", "lore_ptn", "lore_wtw", "trace_base", "trace_wtw", "mutab_ptn", "mutab_ftn", "tablemaster"]
arms = [a for a in ORDER if a in S["arms"]]


def f(x, pct=False):
    if x is None: return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def table(groups, metric, pct=False, title=""):
    out = [f"**{title}**", "", "| group | n | " + " | ".join(NAMES[a] for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
    for g in groups:
        if g not in R: continue
        v = R[g]
        out.append(f"| {g.replace('type ', '').replace('set ', '')} | {v['n']} | " + " | ".join(f(v["m"].get(a, {}).get(metric), pct) for a in arms) + " |")
    return out


def diffs(groups, title):
    cs = [a for a in arms if a not in ("combo",)]
    out = [f"**{title}**", "", "| group | " + " | ".join(NAMES[a] for a in cs) + " |", "|---|" + "---|" * len(cs)]
    for g in groups:
        if g not in R: continue
        row = []
        for a in cs:
            d = R[g]["d"].get(a, {}).get("teds_s")
            row.append("–" if not d else (f"{100 * d['diff']:+.1f}" + ("↑" if d["lo"] > 0 else ("↓" if d["hi"] < 0 else ""))))
        out.append(f"| {g.replace('type ', '').replace('set ', '')} | " + " | ".join(row) + " |")
    return out


types = [g for g in R if g.startswith("type ")]
sets = [g for g in R if g.startswith("set ")]
md = []
md += table(["all"] + types, "teds_s", title="TEDS-Struct, overall and by document type (candidates marked * ran on the 25% core sample, TableMaster on 3%; each arm's mean is over the tables it ran on)")
md += [""] + diffs(["all"] + types + sets, "Candidate − combo, TEDS-Struct points on the same tables (↑/↓: 95% CI above/below zero)")
md += [""] + table(["all"] + sets, "teds_s", title="TEDS-Struct by dataset")
md += [""] + table(["all"] + types, "shape", pct=True, title="Exact grid shape %, overall and by document type")
md += [""] + table(["all"] + types, "teds", title="TEDS with content (common words; sets with answer-key text)")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(md))
