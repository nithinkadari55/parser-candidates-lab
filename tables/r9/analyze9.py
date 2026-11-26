"""Round 9 analysis: baselines (TableFormer, TATR, combo with the milestone-14 selector) and every candidate on the same
tables; per dataset, per document type, per language and slice; paired document-bootstrap 95% CIs against the combo;
speed (same-runner TATR reference), memory and size; the decision rule fixed before the run.

usage: analyze9.py BASE_PUB CAND_ROOT SELECTOR_JSON OUT [R8_PUB]
  BASE_PUB   directory with base9 pub files (s<k>.jsonl), any depth
  CAND_ROOT  directory with candidate artifacts (<model>/s<k>p<p>.jsonl, run_*.json, ref/ref_s<k>.json), any depth
  R8_PUB     optional round-8 pub artifacts (meta.jsonl) for the baseline reproduction check"""
import os, sys, json, glob, math, gzip, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common9 import *
import numpy as np

SELECTOR_SHA_M14 = None  # recorded, not enforced (the m14-sel artifact's table_selector.json)
MIN_N = 50
NAMES = {"tf": "TableFormer", "tatr": "TATR v1.1 fp16", "combo": "combo (TF + TATR + M14 selector)", "tatr_c": "TATR (candidate runner, check)",
         "rapid_en": "SLANet en (RapidTable)", "rapid_zh": "SLANet ch (RapidTable)", "rapid_plus": "SLANet_plus (RapidTable)",
         "paddle_slanet": "SLANet (PaddlePaddle ONNX)", "paddle_slanet_plus": "SLANet_plus (PaddlePaddle ONNX)", "ae3": "AE3-TL",
         "mineru": "MinerU-4 table bundle", "pptable": "PP-TableRec-v2 (SLANet_plus + RT-DETR cells)", "tablemaster": "TableMaster (PDF-Extract-Kit)",
         "mutab_ptn": "MuTabNet (PubTabNet ckpt)", "mutab_ftn": "MuTabNet (FinTabNet ckpt)", "lore_wireless": "LORE wireless", "lore_ptn": "LORE PubTabNet",
         "lore_wtw": "LORE WTW", "trace_base": "TRACE base", "trace_wtw": "TRACE WTW"}


def val(s):
    """score8.value: TEDS-Struct, GriTS-Top where TEDS is capped"""
    if not s: return None
    return s["teds_s"] if s.get("teds_s") is not None else s.get("grits_top")


def boot(diffs, docs, n=2000, seed=0):
    """mean of paired differences with a 95% CI resampling documents (vectorised cluster bootstrap)"""
    if not len(diffs): return None
    u, inv = np.unique(np.asarray(docs), return_inverse=True)
    tot = np.bincount(inv, weights=np.asarray(diffs, float)); cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); m = len(u); out = []
    for k in range(0, n, 200):
        idx = rng.integers(0, m, size=(min(200, n - k), m))
        out.append(tot[idx].sum(1) / cnt[idx].sum(1))
    bs = np.sort(np.concatenate(out))
    return float(tot.sum() / cnt.sum()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])


def load_jsonl(pattern):
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        out += jl(p)
    return out


def main():
    base_pub, cand_root, sel_path, out = sys.argv[1:5]
    r8_pub = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(out, exist_ok=True)
    meta = {r["id"]: r for r in load_jsonl(os.path.join(base_pub, "**", "s*.jsonl")) if "set" in r}
    print("base rows", len(meta), flush=True)
    forest = json.load(open(sel_path)); keys = forest["features"]
    sel_sha = hashlib.sha256(open(sel_path, "rb").read()).hexdigest()
    from r5_sel import leaf_sum, PARAMS
    # ---------- rows ----------
    rows = []; excl = collections.Counter()
    for tid, m in meta.items():
        if not m.get("found"): excl["not found by docling"] += 1; continue
        if m.get("err"): excl["base error"] += 1; continue
        sc = m.get("scores") or {}
        if val(sc.get("tf")) is None or val(sc.get("tatr")) is None: excl["answer key over 1,200 cells (unscorable)"] += 1; continue
        r = dict(id=tid, set=m["set"], type=m["type"], lang=m.get("lang"), kind=m.get("kind"), bdoc=m["doc"].split(":", 1)[1], doc=m["doc"],
                 turn=m.get("turn") or 0, empty=(m.get("n_tatr_cells") or 0) == 0, x=m.get("x"), tags=m.get("tags") or {}, line=m.get("line_type"),
                 n_span=m.get("n_span") or 0, gt_rows=m.get("gt_rows") or 0, gt_cols=m.get("gt_cols") or 0, has_text=m.get("has_gt_text", True),
                 src_meta=m.get("src_meta") or {}, s={"tf": sc["tf"], "tatr": sc["tatr"]}, ms={"tf": m.get("ms_tf"), "tatr": m.get("ms_tatr")})
        rows.append(r)
    print("rows", len(rows), "excluded", dict(excl), flush=True)
    # ---------- combo: milestone-14 selector, out-of-fold on its training sets ----------
    import xgboost as XG
    from sklearn.model_selection import GroupKFold
    vec = lambda x: [x[k] for k in keys]
    for r in rows:
        r["p_ship"] = float(1 / (1 + math.exp(-float(np.float32(forest["base_margin"]) + leaf_sum(forest["trees"], np.array(vec(r["x"]), np.float32))))))
    tr = [r for r in rows if r["set"] in SEL_TRAIN and r["turn"] == 0 and not r["empty"]]
    X = np.array([vec(r["x"]) for r in tr], float)
    y = np.array([1 if val(r["s"]["tatr"]) > val(r["s"]["tf"]) + 1e-9 else 0 for r in tr]); g = np.array([r["bdoc"] for r in tr])
    folds = list(GroupKFold(n_splits=5).split(X, y, g)); fold_of = {}
    models = []
    for k, (a, b) in enumerate(folds):
        mdl = XG.XGBClassifier(**PARAMS).fit(X[a], y[a]); models.append(mdl)
        pr = mdl.predict_proba(X[b])[:, 1]
        for i, p in zip(b, pr): tr[i]["p"] = float(p); fold_of[tr[i]["bdoc"]] = k
    for r in rows:
        if "p" in r: continue
        if r["set"] in SEL_TRAIN and r["bdoc"] in fold_of:
            r["p"] = float(models[fold_of[r["bdoc"]]].predict_proba(np.array([vec(r["x"])], float))[:, 1][0])
        elif r["set"] in SEL_TRAIN:
            r["p"] = float(np.mean([mm.predict_proba(np.array([vec(r["x"])], float))[:, 1][0] for mm in models]))
        else:
            r["p"] = r["p_ship"]
    for r in rows:
        r["pick"] = "tf" if r["empty"] else ("tatr" if r["turn"] else ("tatr" if r["p"] > 0.5 else "tf"))
        r["s"]["combo"] = r["s"][r["pick"]]
    # ---------- candidates ----------
    cands = collections.OrderedDict(); runs = collections.defaultdict(list); refs = []
    byid = {r["id"]: r for r in rows}
    for d in sorted(glob.glob(os.path.join(cand_root, "**", "s*p*.jsonl"), recursive=True)):
        model = os.path.basename(os.path.dirname(d)); model = "tatr_c" if model == "tatr" else model
        for c in jl(d):
            r = byid.get(c["id"])
            if r is None: continue
            r["s"][model] = c.get("s") or c.get("so") or {"teds_s": 0.0, "teds": 0.0, "rc_exact": False, "fail": True}
            if c.get("so"): r.setdefault("so", {})[model] = c["so"]
            r["ms"][model] = c.get("ms_total")
            r.setdefault("err", {})[model] = c.get("err")
            cands[model] = True
    for p in glob.glob(os.path.join(cand_root, "**", "run_s*.json"), recursive=True):
        j = json.load(open(p))
        rp = os.path.join(os.path.dirname(os.path.dirname(p)), "ref", f"ref_s{j['shard']}.json")  # same job, same machine
        if os.path.exists(rp):
            t = json.load(open(rp)).get("tatr_ms") or []
            j["ref_ms"] = float(np.median(t)) if t else None
        runs["tatr_c" if j["model"] == "tatr" else j["model"]].append(j)
    arms = ["tf", "tatr", "combo"] + list(cands)
    print("arms", arms, flush=True)
    # ---------- tables ----------
    def metrics(sub, a):
        v = [val(r["s"][a]) for r in sub if a in r["s"]]
        v = [x for x in v if x is not None]
        t = [r["s"][a].get("teds") for r in sub if a in r["s"] and r["has_text"] and r["s"][a].get("teds") is not None]
        e = [1.0 if r["s"][a].get("rc_exact") else 0.0 for r in sub if a in r["s"]]
        return dict(n=len(v), teds_s=np.mean(v) if v else None, teds=np.mean(t) if t else None, shape=np.mean(e) if e else None)

    def paired(sub, a, b="combo", what="teds_s"):
        pr = []
        for r in sub:
            if a not in r["s"] or b not in r["s"]: continue
            if what == "teds_s":
                x, y = val(r["s"][a]), val(r["s"][b])
            elif what == "teds":
                if not r["has_text"]: continue
                x, y = r["s"][a].get("teds"), r["s"][b].get("teds")
            else:
                x, y = float(bool(r["s"][a].get("rc_exact"))), float(bool(r["s"][b].get("rc_exact")))
            if x is None or y is None: continue
            pr.append((x - y, r["doc"], x, y))
        if len(pr) < 5: return None
        res = boot([p[0] for p in pr], [p[1] for p in pr])
        return dict(n=len(pr), diff=res[0], lo=res[1], hi=res[2], a=float(np.mean([p[2] for p in pr])), b=float(np.mean([p[3] for p in pr])))

    groups = collections.OrderedDict()
    groups["all"] = rows
    for s in ALL_SETS9:
        sub = [r for r in rows if r["set"] == s]
        if sub: groups["set " + s] = sub
    for t in TYPES9:
        sub = [r for r in rows if r["type"] == t]
        if sub: groups["type " + t] = sub
    for lg in sorted({r["lang"] for r in rows if r["set"] in ("pulse", "mustard", "odb")}):
        sub = [r for r in rows if r["lang"] == lg and r["set"] in ("pulse", "mustard", "odb")]
        if len(sub) >= 20: groups["lang " + str(lg)] = sub
    groups["kind text-layer PDFs"] = [r for r in rows if r["kind"] in ("pdf", "synth")]
    groups["kind scans/images (OCR words)"] = [r for r in rows if r["kind"] == "scan"]
    groups["slice >= 3 spanning cells"] = [r for r in rows if r["n_span"] >= 3]
    groups["slice long (>= 30 rows)"] = [r for r in rows if r["gt_rows"] >= 30]
    groups["slice wide (>= 10 cols)"] = [r for r in rows if r["gt_cols"] >= 10]
    groups["slice wired (iflytab/tabrec all-line)"] = [r for r in rows if r["line"] in ("wired", "all-line")]
    groups["slice wireless (iflytab/tabrec no-line)"] = [r for r in rows if r["line"] in ("wireless", "no-line")]
    groups["slice rotated (turn applied)"] = [r for r in rows if r["turn"]]
    res = collections.OrderedDict()
    for gname, sub in groups.items():
        res[gname] = dict(n=len(sub), docs=len({r["doc"] for r in sub}), m={a: metrics(sub, a) for a in arms},
                          d={a: {w: paired(sub, a, "combo", w) for w in ("teds_s", "teds", "shape")} for a in arms if a != "combo"})
        print("group", gname, len(sub), flush=True)
    # own-text TEDS for text-reading models
    own = {}
    for a in arms:
        sub = [r for r in rows if r.get("so", {}).get(a) and r["has_text"]]
        if sub: own[a] = dict(n=len(sub), teds_own=float(np.mean([r["so"][a]["teds"] for r in sub if r["so"][a].get("teds") is not None])),
                              teds_words=float(np.mean([r["s"][a].get("teds") or 0 for r in sub])))
    # oracle complementarity: best of combo and candidate per table
    oracle = {}
    for a in cands:
        sub = [r for r in rows if a in r["s"]]
        if sub: oracle[a] = float(np.mean([max(val(r["s"][a]) or 0, val(r["s"]["combo"]) or 0) for r in sub])) - float(np.mean([val(r["s"]["combo"]) or 0 for r in sub]))
    # ---------- decision rule (fixed before the run) ----------
    dec = {}
    tgroups = [g for g in groups if g.startswith("type ") and len(groups[g]) >= 100]
    sgroups = [g for g in groups if g.startswith("set ") and len(groups[g]) >= MIN_N]
    for a in cands:
        if a == "tatr_c": continue
        o = res["all"]["d"][a]["teds_s"]; osh = res["all"]["d"][a]["shape"]
        if not o: continue
        wins_t = [g for g in tgroups if (x := res[g]["d"][a]["teds_s"]) and x["n"] >= 50 and x["lo"] > 0]
        loss_t = [g for g in tgroups if (x := res[g]["d"][a]["teds_s"]) and x["n"] >= 50 and x["hi"] < -0.02]
        wins_s = [g for g in sgroups if (x := res[g]["d"][a]["teds_s"]) and x["n"] >= MIN_N and x["lo"] > 0]
        cov_t = [g for g in tgroups if (x := res[g]["d"][a]["teds_s"]) and x["n"] >= 50]
        cov_s = [g for g in sgroups if (x := res[g]["d"][a]["teds_s"]) and x["n"] >= MIN_N]
        adopt = (o["lo"] > 0 and o["diff"] >= 0.01 and (osh is None or osh["hi"] >= 0) and len(wins_t) >= max(2, math.ceil(len(cov_t) / 2))
                 and len(wins_s) >= max(3, math.ceil(len(cov_s) / 2)) and not loss_t)
        arm_t = [g for g in tgroups if (x := res[g]["d"][a]["teds_s"]) and x["n"] >= 100 and x["lo"] > 0 and x["diff"] >= 0.01]
        dec[a] = dict(adopt=adopt, overall=o, shape=osh, wins_types=wins_t, of_types=len(cov_t), wins_sets=wins_s, of_sets=len(cov_s), big_losses=loss_t,
                      selector_arm_types=[] if adopt else arm_t)
    # ---------- speed, memory, size ----------
    speed = {}
    for a in ["tf", "tatr"] + list(cands):
        ms = [r["ms"].get(a) for r in rows if r["ms"].get(a) is not None]
        rr = runs.get(a, [])
        ratio = [j["ms_total_median"] / j["ref_ms"] for j in rr if j.get("ms_total_median") and j.get("ref_ms")]
        speed[a] = dict(p50=float(np.median(ms)) if ms else None, p90=float(np.percentile(ms, 90)) if ms else None,
                        vs_tatr=float(np.median(ratio)) if ratio else None,
                        peak_mb=max((j["peak_rss_kb"] - (j.get("rss_before_kb") or 0)) / 1024 for j in rr) if rr else None,
                        weights_mb=max(j["weights_bytes"] for j in rr) / 1e6 if rr else None, errors=sum(j["errors"] for j in rr), n=sum(j["n"] for j in rr),
                        cpus=dict(collections.Counter(j["cpu"] for j in rr)))
    # ---------- baseline reproduction against round 8 ----------
    repro = None
    if r8_pub:
        r8 = {m["id"]: m for m in load_jsonl(os.path.join(r8_pub, "**", "meta.jsonl"))}
        same = {"tf": [0, 0], "tatr": [0, 0]}
        for r in rows:
            m = r8.get(r["id"])
            if not m or not m.get("scores"): continue
            for a in same:
                x, y = val(r["s"][a]), val(m["scores"].get(a))
                if x is None or y is None: continue
                same[a][1] += 1; same[a][0] += abs(x - y) < 1e-6
        repro = {a: dict(n=v[1], identical=v[0]) for a, v in same.items()}
    same = [abs((val(r["s"]["tatr_c"]) or 0) - (val(r["s"]["tatr"]) or 0)) < 1e-9 for r in rows if "tatr_c" in r["s"]]
    bchk = [json.load(open(p)) for p in glob.glob(os.path.join(cand_root, "**", "bundle_s*.json"), recursive=True)]
    input_check = dict(tatr_rerun_tables=len(same), tatr_rerun_identical=float(np.mean(same)) if same else None,
                       bundle_jobs=len(bchk), crop_hash_mismatches=sum(b["crop_hash_mismatches"] for b in bchk), crops_checked=sum(b["tables"] for b in bchk))
    S = dict(input_check=input_check, rows=len(rows), excluded=dict(excl), selector_sha=sel_sha, picks_tatr=float(np.mean([r["pick"] == "tatr" for r in rows])),
             oof_rows=len(tr), groups=res, own=own, oracle=oracle, decision=dec, speed=speed, repro=repro, arms=arms,
             coverage={s: dict(answer_keys=sum(1 for m in meta.values() if m["set"] == s), found=sum(1 for m in meta.values() if m["set"] == s and m.get("found")),
                               scored=sum(1 for r in rows if r["set"] == s)) for s in ALL_SETS9})
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    with gzip.open(os.path.join(out, "per_table.jsonl.gz"), "wt") as f:
        for r in rows:
            f.write(json.dumps(dict(id=r["id"], set=r["set"], type=r["type"], lang=r["lang"], pick=r["pick"], p=r["p"],
                                    v={a: val(r["s"][a]) for a in r["s"]}, shape={a: bool(r["s"][a].get("rc_exact")) for a in r["s"]})) + "\n")
    write_md(out, S, groups)


def f4(x): return "–" if x is None else f"{x:.4f}"
def pc(x): return "–" if x is None else f"{100 * x:.1f}"
def ci(x): return "–" if not x else f"{100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"


def write_md(out, S, groups):
    arms = S["arms"]; R = S["groups"]
    md = ["## Generated report (analyze9.py)", "",
          f"Scored tables: {S['rows']} (excluded: {S['excluded']}). Combo picks TATR on {100 * S['picks_tatr']:.1f}%; selector sha256 {S['selector_sha'][:12]}…; out-of-fold rows {S['oof_rows']}.", ""]
    md += ["### Coverage (answer-key tables, found by docling's layout, scored)", "", "| set | answer keys | found | scored |", "|---|---|---|---|"]
    md += [f"| {s} | {v['answer_keys']} | {v['found']} | {v['scored']} |" for s, v in S["coverage"].items() if v["answer_keys"]]
    for metric, label in (("teds_s", "TEDS-Struct"), ("teds", "TEDS (content, common words)"), ("shape", "exact grid shape %")):
        md += ["", f"### {label} by group", "", "| group | tables | " + " | ".join(NAMES.get(a, a) for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
        for g, v in R.items():
            cells = []
            for a in arms:
                m = v["m"].get(a) or {}
                x = m.get(metric)
                cells.append((pc(x) if metric == "shape" else f4(x)) + (f" (n={m['n']})" if m.get("n") and m["n"] != v["n"] else ""))
            md.append(f"| {g} | {v['n']} | " + " | ".join(cells) + " |")
    md += ["", "### Candidate − combo, TEDS-Struct points [95% CI by document]", "", "| group | " + " | ".join(NAMES.get(a, a) for a in arms if a != "combo") + " |", "|---|" + "---|" * (len(arms) - 1)]
    for g, v in R.items():
        md.append(f"| {g} | " + " | ".join(ci(v["d"][a]["teds_s"]) for a in arms if a != "combo") + " |")
    md += ["", "### Candidate − combo, exact shape pp and TEDS points (overall)", "", "| arm | shape pp | TEDS pts | best-of(candidate, combo) gain, TEDS-S pts |", "|---|---|---|---|"]
    for a in arms:
        if a == "combo": continue
        d = R["all"]["d"][a]
        md.append(f"| {NAMES.get(a, a)} | {ci(d['shape'])} | {ci(d['teds'])} | {('%+.2f' % (100 * S['oracle'][a])) if a in S['oracle'] else '–'} |")
    if S["own"]:
        md += ["", "### Models that read cell text themselves: TEDS with their own text vs with the common words", "", "| arm | tables | own text | common words |", "|---|---|---|---|"]
        md += [f"| {NAMES.get(a, a)} | {v['n']} | {v['teds_own']:.4f} | {v['teds_words']:.4f} |" for a, v in S["own"].items()]
    md += ["", "### Speed, memory, size (CPU, 4 vCPU GitHub runners)", "", "| arm | p50 ms/table | p90 ms | × TATR on the same runner | peak RSS over load (MB) | weights MB | errors / tables |", "|---|---|---|---|---|---|---|"]
    for a, v in S["speed"].items():
        md.append(f"| {NAMES.get(a, a)} | {f4(v['p50']) if v['p50'] is None else round(v['p50'])} | {'–' if v['p90'] is None else round(v['p90'])} | {'–' if v['vs_tatr'] is None else '%.2f' % v['vs_tatr']} | "
                  f"{'–' if v['peak_mb'] is None else round(v['peak_mb'])} | {'–' if v['weights_mb'] is None else '%.1f' % v['weights_mb']} | {v['errors']} / {v['n']} |")
    md += ["", "### Decision rule (fixed before the run)", "",
           "Adopt only if, against the combo: overall TEDS-Struct gain >= 1.0 point with the 95% CI above zero; exact shape not significantly worse; "
           "CI above zero on at least half (and at least 2) of the document types with >= 100 tables and at least half (and at least 3) of the datasets with >= 50 tables; "
           "no document type significantly worse by more than 2 points. Otherwise, a candidate with a >= 1 point CI-positive win on a document type (>= 100 tables) is a possible selector arm.", "",
           "| candidate | tables | candidate / combo TEDS-S on those tables | overall TEDS-S diff | shape | wins: doc types | wins: datasets | big losses | adopt | possible selector arm for |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a, d in S["decision"].items():
        md.append(f"| {NAMES.get(a, a)} | {d['overall']['n']} | {d['overall']['a']:.4f} / {d['overall']['b']:.4f} | {ci(d['overall'])} | {ci(d['shape'])} | {len(d['wins_types'])}/{d['of_types']} | {len(d['wins_sets'])}/{d['of_sets']} | "
                  f"{', '.join(g[5:] for g in d['big_losses']) or '–'} | {'**yes**' if d['adopt'] else 'no'} | {', '.join(g[5:] for g in d['selector_arm_types']) or '–'} |")
    md += ["", f"Input identity check: {S['input_check']}"]
    if S["repro"]:
        md += ["", f"Baseline reproduction against round 8 (same table ids): {S['repro']}"]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
