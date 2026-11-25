"""Round 8 analysis: every arm on the same tables, the logic repairs tuned on the dev split, the selectors, the
statistics and the decision rule fixed in advance.

usage: analyze8.py PUB_ROOT PRIV_ROOT SELECTOR_JSON TF_ROOT OUT_DIR
  PUB_ROOT      meta.jsonl of every tatr8 shard (scores, metadata)
  PRIV_ROOT     priv.jsonl.gz of every tatr8 shard (cells and words)
  SELECTOR_JSON the harness's committed selector trees (crates/contextual-worker/src/tables/table_selector.json)
  TF_ROOT       tf8 outputs (sizes.json)
Value per table: TEDS-Struct (GriTS-Top where the answer key or the grid exceeds the TEDS cap, as rounds 3-5).
CIs: 95% bootstrap over documents; p: Wilcoxon signed-rank (zsplit)."""
import os, sys, json, glob, gzip, math, time, random, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import numpy as np
import score8 as S8
import logic8 as L
import r3_select as R3
from r5_sel import leaf_sum, PARAMS
from multiprocessing import Pool

SELECTOR_SHA = "c27d7295112ab514e46cf0a1595782468c932b06084d10ad3247d093a2ed34f1"
MIN_GAIN = 0.0003
TYPE_GUARD = 0.002
META, PRIV, GTS = {}, {}, {}
CFG_POOL = {}


# ---------- geometry ----------

def _corners(b, f):
    pts = [f(b[0], b[1]), f(b[2], b[3]), f(b[0], b[3]), f(b[2], b[1])]
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def turn_box(b, t, w, h):
    return _corners(b, {90: lambda x, y: (y, w - x), 180: lambda x, y: (w - x, h - y), 270: lambda x, y: (h - y, x)}.get(t, lambda x, y: (x, y)))


def unturn_box(b, t, w, h):
    return _corners(b, {90: lambda x, y: (w - y, x), 180: lambda x, y: (w - x, h - y), 270: lambda x, y: (y, h - x)}.get(t, lambda x, y: (x, y)))


def gt_of(tid):
    if tid not in GTS: GTS[tid] = load_gt(META[tid]["set"], tid)
    return GTS[tid]


def logic_cells(tid, cfg):
    m, p = META[tid], PRIV[tid]
    cells, words, t = p["cells"]["tatr"], p["words"], m.get("turn") or 0
    if not cells: return []
    if t:
        W, H = p["W"], p["H"]
        cells = [dict(c, bbox=turn_box(c["bbox"], t, W, H)) for c in cells]
        words = [dict(w, bbox=turn_box(w["bbox"], t, W, H)) for w in words]
    out = L.apply(cells, words, cfg)
    if t: out = [dict(c, bbox=unturn_box(c["bbox"], t, p["W"], p["H"])) for c in out]
    return out


def _logic_job(a):
    tid, ck = a
    t0 = time.perf_counter()
    try:
        cells = logic_cells(tid, CFG_POOL[ck])
    except Exception as e:
        return tid, None, None, repr(e)[:200], 0.0
    ms = (time.perf_counter() - t0) * 1000
    return tid, L.signature(cells), cells, None, ms


def _struct_job(a):
    tid, cells = a
    return tid, S8.score_cells(gt_of(tid), PRIV[tid]["words"], cells, structure_only=True)


def _full_job(a):
    tid, cells = a
    return tid, S8.score_cells(gt_of(tid), PRIV[tid]["words"], cells)


def _feat_job(a):
    tid, arm = a
    p = PRIV[tid]; words = [dict(text=w["text"], bbox=w["bbox"]) for w in p["words"]]
    ca, cb = p["cells"]["tf"], (p["cells"][arm] if arm in p["cells"] else p.get("_logic") or [])
    t0 = time.perf_counter()
    fa, fb = R3.feats(ca, words), R3.feats(cb, words)
    x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
    x.update({"d_" + k: fb[k] - fa[k] for k in fa})
    x["agree"] = R3.agreement(ca, cb); x["n_words"] = len(words)
    return tid, arm, x, (time.perf_counter() - t0) * 1000


# ---------- loading ----------

def load(pub_root, priv_root):
    for p in glob.glob(os.path.join(pub_root, "**", "meta.jsonl"), recursive=True):
        for r in jl(p): META[r["id"]] = r
    for p in glob.glob(os.path.join(priv_root, "**", "priv.jsonl.gz"), recursive=True):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line); PRIV[r["id"]] = r
    print("meta", len(META), "priv", len(PRIV), flush=True)


def base_doc(doc): return doc.split(":", 1)[1] if ":" in doc else doc


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x): return "–" if x is None else f"{100 * x:.1f}"


def f4(x): return "–" if x is None else f"{x:.4f}"


def boot(rows, a, b, key):
    rr = [dict(doc=r["bdoc"], x=r[key][a], y=r[key][b]) for r in rows if r[key].get(a) is not None and r[key].get(b) is not None]
    if len(rr) < 5: return None
    c = cluster_boot(rr, "x", "y", group="doc")
    p = wilcoxon_p([r["x"] for r in rr], [r["y"] for r in rr])
    w = sum(1 for r in rr if r["x"] > r["y"] + 1e-9); l = sum(1 for r in rr if r["x"] < r["y"] - 1e-9)
    return dict(n=len(rr), diff=c[0], lo=c[1], hi=c[2], p=p, wins=w, losses=l)


def fmt_boot(t, scale=1.0, unit=""):
    if not t: return "–"
    return f"{scale * t['diff']:+.2f}{unit} [{scale * t['lo']:+.2f}, {scale * t['hi']:+.2f}] p={t['p']:.2g} (n={t['n']}, +{t['wins']}/−{t['losses']})"


# ---------- main ----------

def main():
    pub_root, priv_root, sel_json, tf_root, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    load(pub_root, priv_root)
    md = []; P = md.append; summary = {}
    # coverage
    cov = collections.defaultdict(collections.Counter)
    for m in META.values():
        c = cov[m["set"]]; c["tables"] += 1; c["found"] += m["found"]; c["tf_err"] += bool(m.get("tf_err")); c["err"] += bool(m.get("err"))
    rows = []
    excluded = collections.Counter()
    for tid, m in META.items():
        if not m["found"]: continue
        if m.get("err") or tid not in PRIV: excluded["tatr job error"] += 1; continue
        sc = m.get("scores") or {}
        if "tf" not in sc or "tatr" not in sc: excluded["missing arm"] += 1; continue
        if S8.value(sc["tf"]) is None or S8.value(sc["tatr"]) is None:
            excluded["answer key over 1,200 cells (unscorable)"] += 1; continue
        r = dict(id=tid, set=m["set"], type=m["type"], split=m["split"], bdoc=base_doc(m["doc"]), bucket=m["bucket"], tags=m.get("tags", {}),
                 n_span=m.get("n_span") or 0, turn=m.get("turn") or 0, has_text=m.get("has_gt_text", True), v={}, shape={}, gtop={}, gcon={}, teds={})
        for a, s in sc.items(): put(r, a, s)
        rows.append(r)
    print("rows", len(rows), "excluded", dict(excluded), flush=True)
    by = {r["id"]: r for r in rows}

    global CFG_POOL
    pool = Pool(4)

    # ---------- arm 3: logic, tuned on dev ----------
    dev = [r for r in rows if r["split"] == "dev"]
    cache = {}  # (tid, signature) -> structure score

    def run_cfg(cfg, ids):
        global CFG_POOL
        CFG_POOL["c"] = cfg
        pool2 = Pool(4)  # a fresh pool sees the new config (fork)
        res = {}; todo = []; ms = []
        for tid, sig, cells, err, t in pool2.imap_unordered(_logic_job, [(i, "c") for i in ids], chunksize=8):
            ms.append(t)
            if err or cells is None: res[tid] = None; continue
            if sig == L.signature(PRIV[tid]["cells"]["tatr"]): res[tid] = "same"; continue
            if (tid, sig) in cache: res[tid] = cache[(tid, sig)]; continue
            todo.append((tid, cells, sig))
        sigs = {tid: sig for tid, _, sig in todo}
        for tid, s in pool2.imap_unordered(_struct_job, [(t, c) for t, c, _ in todo], chunksize=1):
            cache[(tid, sigs[tid])] = s; res[tid] = s
        pool2.close(); pool2.join()
        return res, ms

    def judge(res, ids):
        d_v, d_s, per_type, changed = [], [], collections.defaultdict(list), 0
        for tid in ids:
            r = by[tid]; s = res.get(tid)
            if s is None or s == "same": nv, ns = r["v"]["tatr"], r["shape"]["tatr"]
            else:
                changed += 1; nv = S8.value(s); ns = float(bool(s.get("rc_exact")))
                if nv is None: nv = r["v"]["tatr"]
            d_v.append(nv - r["v"]["tatr"]); d_s.append(ns - r["shape"]["tatr"]); per_type[r["type"]].append(nv - r["v"]["tatr"])
        worst = min((sum(v) / len(v) for t, v in per_type.items() if len(v) >= 30), default=0.0)
        return dict(dv=sum(d_v) / max(1, len(d_v)), ds=sum(d_s) / max(1, len(d_s)), worst_type=worst, changed=changed,
                    per_type={t: sum(v) / len(v) for t, v in per_type.items()})

    dev_ids = [r["id"] for r in dev]
    tuning = []; best = {}
    t0 = time.time()
    for rule in L.ORDER:
        cands = []
        for st in L.SPACE[rule]:
            res, _ = run_cfg({rule: st}, dev_ids); j = judge(res, dev_ids)
            tuning.append(dict(rule=rule, setting=st, **j)); cands.append((st, j))
            print("tune", rule, st, {k: round(v, 5) if isinstance(v, float) else v for k, v in j.items() if k != "per_type"}, round(time.time() - t0), "s", flush=True)
        ok = [(st, j) for st, j in cands if j["dv"] >= MIN_GAIN and j["ds"] >= 0 and j["worst_type"] >= -TYPE_GUARD]
        if ok: best[rule] = max(ok, key=lambda x: x[1]["dv"])
    if "J" in best:
        st = dict(best["J"][0], header=True)
        res, _ = run_cfg({"J": st}, dev_ids); j = judge(res, dev_ids)
        tuning.append(dict(rule="J", setting=st, **j))
        if j["dv"] > best["J"][1]["dv"] and j["ds"] >= 0 and j["worst_type"] >= -TYPE_GUARD: best["J"] = (st, j)
    chosen = {}; cur = dict(dv=0.0, ds=0.0)
    for rule, (st, j) in sorted(best.items(), key=lambda x: -x[1][1]["dv"]):
        trial = dict(chosen); trial[rule] = st
        res, _ = run_cfg(trial, dev_ids); jj = judge(res, dev_ids)
        tuning.append(dict(rule="+".join(sorted(trial)), setting=trial, combined=True, **jj))
        if jj["dv"] >= cur["dv"] + MIN_GAIN and jj["ds"] >= cur["ds"] and jj["worst_type"] >= -TYPE_GUARD:
            chosen = trial; cur = jj
    print("logic chosen on dev", chosen, cur, flush=True)
    json.dump(dict(chosen=chosen, dev=cur, tuning=tuning, min_gain=MIN_GAIN, type_guard=TYPE_GUARD), open(os.path.join(out, "logic_tuning.json"), "w"), default=str)

    # apply the chosen repairs to every table, full scoring
    logic_ms = {}
    if chosen:
        CFG_POOL["c"] = chosen
        pool.close(); pool = Pool(4)
        todo = []
        for tid, sig, cells, err, t in pool.imap_unordered(_logic_job, [(r["id"], "c") for r in rows], chunksize=8):
            logic_ms[tid] = t
            if err or cells is None: cells = PRIV[tid]["cells"]["tatr"]
            PRIV[tid]["_logic"] = cells
            if sig == L.signature(PRIV[tid]["cells"]["tatr"]):
                put(by[tid], "tatr_logic", META[tid]["scores"]["tatr"]); by[tid]["logic_changed"] = False
            else:
                todo.append((tid, cells)); by[tid]["logic_changed"] = True
        pool.close(); pool = Pool(4)
        for tid, s in pool.imap_unordered(_full_job, todo, chunksize=1): put(by[tid], "tatr_logic", s)
    else:
        for r in rows: put(r, "tatr_logic", META[r["id"]]["scores"]["tatr"]); r["logic_changed"] = False; PRIV[r["id"]]["_logic"] = PRIV[r["id"]]["cells"]["tatr"]
    print("logic applied; changed tables", sum(r.get("logic_changed", False) for r in rows), flush=True)

    # ---------- arm 4: selectors ----------
    forest = json.load(open(sel_json))
    assert hashlib.sha256(open(sel_json, "rb").read()).hexdigest() == SELECTOR_SHA, "not the harness's committed selector"
    pool.close(); pool = Pool(4)
    feats = collections.defaultdict(dict); fms = []
    arms_f = ["tatr"] + (["tatr_logic"] if chosen else [])
    for tid, arm, x, t in pool.imap_unordered(_feat_job, [(r["id"], a) for r in rows for a in arms_f], chunksize=4):
        feats[tid][arm] = x
        if arm == "tatr": fms.append(t)
    pool.close()
    keys = forest["features"]
    for r in rows:
        x = feats[r["id"]]["tatr"]
        m = float(np.float32(forest["base_margin"]) + leaf_sum(forest["trees"], np.array([x[k] for k in keys], np.float32)))
        r["p_committed"] = 1 / (1 + math.exp(-m))
    import xgboost as XG
    from sklearn.model_selection import GroupKFold
    def retrain(arm, label):
        tr = [r for r in rows if r["turn"] == 0 and PRIV[r["id"]]["cells"]["tatr"]]
        X = np.array([[feats[r["id"]][arm][k] for k in keys] for r in tr], float)
        y = np.array([1 if r["v"][arm] > r["v"]["tf"] + 1e-9 else 0 for r in tr])
        g = np.array([r["bdoc"] for r in tr]); p = np.zeros(len(tr))
        for a, b in GroupKFold(n_splits=5).split(X, y, g):
            p[b] = XG.XGBClassifier(**PARAMS).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        for r, q in zip(tr, p): r[label] = float(q)
        return len(tr)
    n_tr = retrain("tatr", "p_retrained")
    if chosen: retrain("tatr_logic", "p_retrained_logic")
    for r in rows:
        tatr_empty = not PRIV[r["id"]]["cells"]["tatr"]
        def pick(pkey, arm):
            if tatr_empty: return "tf"
            if r["turn"]: return arm
            return arm if r.get(pkey, 0.0) > 0.5 else "tf"
        for name, pkey, arm in (("combo", "p_committed", "tatr"), ("combo_retrained", "p_retrained", "tatr"), ("combo_retrained_logic", "p_retrained_logic", "tatr_logic")):
            if name == "combo_retrained_logic" and not chosen: continue
            a = pick(pkey, arm); r["pick_" + name] = a
            for k in ("v", "shape", "gtop", "gcon", "teds"): r[k][name] = r[k].get(a)
        for k in ("v", "shape", "gtop", "gcon", "teds"):
            vals = [r[k].get(a) for a in ("tf", "tatr") if r[k].get(a) is not None]
            r[k]["oracle"] = max(vals) if vals else None
    print("selectors done", flush=True)

    # ---------- report ----------
    ARMS = [("tf", "TableFormer (docling-rs 2.84.0)"), ("tatr", "TATR v1.1 fp16 alone"), ("tatr_logic", "TATR + logic repairs"),
            ("combo", "Combo: TF + TATR + committed selector"), ("combo_retrained", "Combo, selector retrained (CV)")]
    if chosen: ARMS.append(("combo_retrained_logic", "Combo, retrained, TF + TATR+logic"))
    ARMS.append(("oracle", "Best of TF/TATR per table"))
    test = [r for r in rows if r["split"] == "test"]
    summary["coverage"] = {s: dict(c) for s, c in cov.items()}; summary["excluded"] = dict(excluded)
    summary["logic"] = dict(chosen=chosen, dev=cur)

    def arm_table(sub, title):
        P(f"\n### {title} ({len(sub)} tables, {len(set(r['bdoc'] for r in sub))} documents)\n")
        P("| arm | TEDS-Struct | GriTS-Top | GriTS-Con | exact shape % |")
        P("|---|---|---|---|---|")
        res = {}
        for a, name in ARMS:
            v = mean([r["v"].get(a) for r in sub]); gt = mean([r["gtop"].get(a) for r in sub]); gc = mean([r["gcon"].get(a) for r in sub if r["has_text"]])
            sh = mean([r["shape"].get(a) for r in sub])
            res[a] = dict(teds_s=v, grits_top=gt, grits_con=gc, shape=sh)
            P(f"| {name} | {f4(v)} | {f4(gt)} | {f4(gc)} | {pct(sh)} |")
        return res

    def group_table(groups, title, arms=("tf", "tatr", "tatr_logic", "combo", "combo_retrained")):
        P(f"\n### {title}\n")
        P("| group | tables | " + " | ".join(dict(ARMS)[a] for a in arms) + " | combo − TATR(+logic) TEDS-S pts [95% CI] |")
        P("|---|---|" + "---|" * len(arms) + "---|")
        out = {}
        for name, sub in groups:
            if not sub: continue
            cells = []
            for a in arms:
                v = mean([r["v"].get(a) for r in sub]); sh = mean([r["shape"].get(a) for r in sub])
                cells.append(f"{f4(v)} / {pct(sh)}%")
                out.setdefault(name, {})[a] = dict(teds_s=v, shape=sh)
            t = boot(sub, "combo", CAND, "v")
            out[name]["combo_minus_cand"] = t
            P(f"| {name} | {len(sub)} | " + " | ".join(cells) + f" | {fmt_boot(t, 100)} |")
        return out

    # the TATR candidate: + logic only if it beats TATR alone on the test split (TEDS-Struct up, exact shape not down)
    lt = boot(test, "tatr_logic", "tatr", "v"); ls_ = boot(test, "tatr_logic", "tatr", "shape")
    global CAND
    CAND = "tatr_logic" if (chosen and lt and lt["diff"] > 0 and ls_ and ls_["diff"] >= 0) else "tatr"
    summary["candidate"] = CAND

    P("# Tables round 8: TATR alone vs the TableFormer + TATR combo, broad benchmark\n")
    P("All numbers below are on the **test split** (70% of documents, fixed by a hash of the document id) unless marked; the logic repairs were tuned on the dev split only.")
    P(f"Value per table: TEDS-Struct (GriTS-Top where the answer key or the predicted grid exceeds the TEDS cap). Exact shape: rows and columns both right. CIs: 95% bootstrap over documents; p: Wilcoxon.")
    summary["overall_test"] = arm_table(test, "Overall, test split")
    summary["overall_all"] = arm_table(rows, "Overall, all tables (dev + test; logic numbers on dev are in-sample)")
    types = [(t, [r for r in test if r["type"] == t]) for t in TYPES]
    summary["by_type"] = group_table(types, "By document type (test split): TEDS-Struct / exact shape")
    sets = [(f"{s}: {SET_INFO[s][2]}", [r for r in test if r["set"] == s]) for s in ALL_SETS]
    summary["by_set"] = group_table(sets, "By dataset (test split): TEDS-Struct / exact shape")
    sl = [("complex: >= 3 spanning cells", [r for r in test if (r["n_span"] or 0) >= 3]),
          ("nested header", [r for r in test if r["tags"].get("nested_header")]),
          ("long (>= 30 rows)", [r for r in test if r["tags"].get("long")]),
          ("wide (>= 10 columns)", [r for r in test if r["tags"].get("wide")]),
          ("multi-line cells", [r for r in test if r["tags"].get("multiline")]),
          ("SciTSR COMP (complicated)", [r for r in test if r["tags"].get("scitsr_comp")])] + \
         [(f"bucket {b}", [r for r in test if r["bucket"] == b]) for b in ("simple", "medium", "complex", "edge")]
    summary["slices"] = group_table(sl, "Slices (test split): TEDS-Struct / exact shape")

    # paired comparisons
    P("\n### Paired differences (test split; TEDS-Struct in points, exact shape in percentage points)\n")
    P("| comparison | TEDS-Struct | exact shape |"); P("|---|---|---|")
    pairs = [("tatr", "tf", "TATR − TableFormer"), ("tatr_logic", "tatr", "TATR+logic − TATR"), ("combo", "tatr", "combo − TATR"),
             ("combo", "tatr_logic", "combo − TATR+logic"), ("combo", "tf", "combo − TableFormer"), ("combo_retrained", "combo", "retrained − committed selector")]
    if chosen: pairs.append(("combo_retrained_logic", "combo_retrained", "retrained combo with TATR+logic − retrained combo"))
    summary["paired"] = {}
    for a, b, name in pairs:
        tv, ts = boot(test, a, b, "v"), boot(test, a, b, "shape")
        summary["paired"][name] = dict(teds_s=tv, shape=ts)
        P(f"| {name} | {fmt_boot(tv, 100)} | {fmt_boot(ts, 100)} |")

    # decision
    ov = summary["overall_test"]
    d_teds = 100 * (ov["combo"]["teds_s"] - ov[CAND]["teds_s"]); d_shape = 100 * (ov["combo"]["shape"] - ov[CAND]["shape"])
    worst = {t: 100 * (summary["by_type"][t]["combo"]["teds_s"] - summary["by_type"][t][CAND]["teds_s"]) for t in summary["by_type"]}
    worst_shape = {t: 100 * (summary["by_type"][t]["combo"]["shape"] - summary["by_type"][t][CAND]["shape"]) for t in summary["by_type"]}
    ok1, ok2, ok3 = d_teds < 0.5, d_shape < 1.0, all(v <= 2.0 for v in worst.values())
    dec = "TATR alone" + (" + logic" if CAND == "tatr_logic" else "") if (ok1 and ok2 and ok3) else "keep the combo"
    # the same rule with the retrained selector as the combo
    d_teds_r = 100 * (ov["combo_retrained"]["teds_s"] - ov[CAND]["teds_s"]); d_shape_r = 100 * (ov["combo_retrained"]["shape"] - ov[CAND]["shape"])
    worst_r = {t: 100 * (summary["by_type"][t]["combo_retrained"]["teds_s"] - summary["by_type"][t][CAND]["teds_s"]) for t in summary["by_type"]}
    dec_r = "TATR" if (d_teds_r < 0.5 and d_shape_r < 1.0 and all(v <= 2.0 for v in worst_r.values())) else "combo"
    summary["decision"] = dict(candidate=CAND, d_teds_pts=d_teds, d_shape_pp=d_shape, type_gap_pts=worst, type_gap_shape_pp=worst_shape,
                               rule=dict(teds_under_0_5=ok1, shape_under_1=ok2, no_type_over_2=ok3), decision=dec,
                               with_retrained=dict(d_teds_pts=d_teds_r, d_shape_pp=d_shape_r, type_gap_pts=worst_r, decision=dec_r))
    P("\n## Decision (rule fixed in advance)\n")
    P(f"- TATR candidate: **{dict(ARMS)[CAND]}** (TATR+logic counts only if it beats TATR alone on the test split).")
    P(f"- Combo (committed selector) − candidate, overall: TEDS-Struct **{d_teds:+.2f} points** (limit < 0.5), exact shape **{d_shape:+.2f} pp** (limit < 1).")
    P("- Combo − candidate by type (TEDS-Struct points; limit: candidate not more than 2 worse): " + ", ".join(f"{t} {v:+.2f}" for t, v in worst.items()))
    P("- Same by type, exact shape pp (for information): " + ", ".join(f"{t} {v:+.2f}" for t, v in worst_shape.items()))
    P(f"- **Decision: {dec}.** With the retrained selector as the combo: TEDS {d_teds_r:+.2f}, shape {d_shape_r:+.2f}, worst type {max(worst_r.values()):+.2f} → {dec_r}.")

    # logic
    P("\n## Logic repairs (arm 3), tuned on the dev split\n")
    P(f"Chosen: `{json.dumps(chosen)}`; dev gain TEDS-Struct {100 * cur.get('dv', 0):+.2f} points, shape {100 * cur.get('ds', 0):+.2f} pp. Tables changed (all splits): {sum(r.get('logic_changed', False) for r in rows)}.")
    P("\n| rule | setting | dev Δ TEDS-S (pts) | dev Δ shape (pp) | worst type Δ (pts) | tables changed |"); P("|---|---|---|---|---|---|")
    for t in tuning:
        P(f"| {t['rule']} | `{json.dumps(t['setting'])}` | {100 * t['dv']:+.3f} | {100 * t['ds']:+.3f} | {100 * t['worst_type']:+.3f} | {t['changed']} |")

    # selector
    P("\n## Selectors\n")
    for name in ("combo", "combo_retrained") + (("combo_retrained_logic",) if chosen else ()):
        n_t = sum(1 for r in test if r["pick_" + name] != "tf")
        P(f"- {name}: picks TATR on {n_t} of {len(test)} test tables ({100 * n_t / max(1, len(test)):.1f}%).")
    P(f"- Retrained selector: XGBoost (round-5 settings), 5-fold GroupKFold by document over {n_tr} unrotated tables; out-of-fold picks.")
    fs = sorted(fms)
    if fs: P(f"- Selector feature time (Python): p50 {fs[len(fs) // 2]:.1f} ms, p90 {fs[int(0.9 * (len(fs) - 1))]:.1f} ms per table.")

    # orientation
    P("\n## Orientation\n")
    up = [m for m in META.values() if m["found"] and not m.get("err") and m["set"] != "rotpdf" and m.get("lcnet") is not None]
    P("**PP-LCNet on upright tables (label ≠ 0 = it would turn an upright table):**\n")
    P("| set | tables | wrongly turned % |"); P("|---|---|---|")
    summary["lcnet_upright"] = {}
    for s in [x for x in ALL_SETS if x != "rotpdf"]:
        sub = [m for m in up if m["set"] == s]
        if sub:
            w = sum(1 for m in sub if m["lcnet"] != 0) / len(sub); summary["lcnet_upright"][s] = dict(n=len(sub), wrong=w)
            P(f"| {s} | {len(sub)} | {100 * w:.1f} |")
    for t in TYPES:
        sub = [m for m in up if m["type"] == t]
        if sub: P(f"| type: {t} | {len(sub)} | {100 * sum(1 for m in sub if m['lcnet'] != 0) / len(sub):.1f} |")
    gts = [m for m in META.values() if m["set"] in SCAN_SETS and "gt_tatr" in (m.get("scores") or {}) and not m.get("gt_err")]
    P("\n**Scans on the answer-key crops** (every scan table, found by docling or not; OCR words; no TableFormer here):\n")
    P("| set | tables | TATR + OCR TEDS-Struct | exact shape % | PP-LCNet turns an upright crop % |"); P("|---|---|---|---|---|")
    summary["scan_gt_crops"] = {}
    for st in SCAN_SETS:
        sub = [m for m in gts if m["set"] == st]
        if not sub: continue
        v = mean([S8.value(m["scores"]["gt_tatr"]) for m in sub]); sh = mean([float(bool(m["scores"]["gt_tatr"].get("rc_exact"))) for m in sub])
        wr = sum(1 for m in sub if m.get("gt_lcnet")) / len(sub)
        summary["scan_gt_crops"][st] = dict(n=len(sub), teds_s=v, shape=sh, lcnet_wrong=wr)
        P(f"| {st} | {len(sub)} | {f4(v)} | {pct(sh)} | {100 * wr:.1f} |")
    P(f"\nAnswer-key crop errors: {sum(1 for m in META.values() if m.get('gt_err'))}.")
    ur = []
    for m in gts:
        a, b = m["scores"].get("gt_tatr"), m["scores"].get("gt_tatr_lcnet")
        if a and b and S8.value(a) is not None and S8.value(b) is not None:
            ur.append(dict(bdoc=base_doc(m["doc"]), v=dict(lc=S8.value(b), no=S8.value(a)), shape=dict(lc=float(bool(b.get("rc_exact"))), no=float(bool(a.get("rc_exact"))))))
    if ur:
        t4, t5 = boot(ur, "lc", "no", "v"), boot(ur, "lc", "no", "shape")
        P(f"\nUpright scans, answer-key crops ({len(ur)}): with PP-LCNet − without: TEDS-Struct {fmt_boot(t4, 100)}; shape {fmt_boot(t5, 100, ' pp')}.")
        summary["lcnet_upright_scans_gt"] = dict(n=len(ur), teds=t4, shape=t5)
    rs = [m for m in META.values() if m.get("rs_theta") and "rs_none" in (m.get("scores") or {}) and "rs_lcnet" in m["scores"] and "gt_tatr" in m["scores"]]
    if rs:
        acc = sum(1 for m in rs if m["rs_lcnet"] == m["rs_need"]) / len(rs); alt = sum(1 for m in rs if m["rs_lcnet"] == m["rs_theta"]) / len(rs)
        rr = [dict(bdoc=base_doc(m["doc"]), v=dict(none=S8.value(m["scores"]["rs_none"]), lc=S8.value(m["scores"]["rs_lcnet"]), up=S8.value(m["scores"]["gt_tatr"])),
                   shape=dict(none=float(bool(m["scores"]["rs_none"].get("rc_exact"))), lc=float(bool(m["scores"]["rs_lcnet"].get("rc_exact"))), up=float(bool(m["scores"]["gt_tatr"].get("rc_exact")))))
              for m in rs]
        rr = [r for r in rr if None not in r["v"].values()]
        P(f"\n**Rotated scans** ({len(rr)} scan tables turned 90/180/270, OCR + TATR on the turned crop): PP-LCNet right on {100 * acc:.1f}% (the opposite-direction reading would be right on {100 * alt:.1f}%, a convention check).\n")
        P("| arm | TEDS-Struct | exact shape % |"); P("|---|---|---|")
        for k, name in (("none", "no orientation fix"), ("lc", "PP-LCNet then OCR + TATR"), ("up", "upright (oracle)")):
            P(f"| {name} | {f4(mean([r['v'][k] for r in rr]))} | {pct(mean([r['shape'][k] for r in rr]))} |")
        t1 = boot(rr, "lc", "none", "v"); P(f"\nPP-LCNet − no fix on rotated scans: {fmt_boot(t1, 100)}")
        summary["rotscan"] = dict(n=len(rr), acc=acc, alt=alt, gain=t1, none=mean([r['v']['none'] for r in rr]), lc=mean([r['v']['lc'] for r in rr]), up=mean([r['v']['up'] for r in rr]))
    scans = [r for r in rows if r["set"] in SCAN_SETS and r["v"].get("tatr_lcnet") is not None]
    if scans:
        t2 = boot(scans, "tatr_lcnet", "tatr", "v"); t3 = boot(scans, "tatr_lcnet", "tatr", "shape")
        P(f"\nUpright scans on the run-time crops ({len(scans)}, all splits): with PP-LCNet − without: TEDS-Struct {fmt_boot(t2, 100)}; shape {fmt_boot(t3, 100, ' pp')}.")
        summary["lcnet_upright_scans"] = dict(teds=t2, shape=t3)
    rp = [m for m in META.values() if m["set"] == "rotpdf" and m["found"] and not m.get("err") and m.get("truth_turn") is not None]
    if rp:
        ga = sum(1 for m in rp if m.get("turn") == m["truth_turn"]) / len(rp)
        la = sum(1 for m in rp if m.get("lcnet") == m["truth_turn"]) / len(rp)
        upd = [m for m in META.values() if m["found"] and not m.get("err") and set_kind(m["set"]) != "scan" and m["set"] != "rotpdf" and m.get("turn") is not None]
        P(f"\nRotated text-PDF tables docling found: {len(rp)}; glyph-angle vote right on {100 * ga:.1f}% (PP-LCNet {100 * la:.1f}%). Upright text tables wrongly turned by the vote: {100 * sum(1 for m in upd if m['turn']) / max(1, len(upd)):.2f}% of {len(upd)}.")
        summary["rotpdf"] = dict(n=len(rp), glyph=ga, lcnet=la)

    # coverage, time, sizes
    P("\n## Coverage (tables docling's layout found; only those are compared)\n")
    P("| set | answer-key tables | docling found | conversion errors | TATR-job errors |"); P("|---|---|---|---|---|")
    for s in ALL_SETS:
        c = cov.get(s)
        if c: P(f"| {s} | {c['tables']} | {c['found']} ({100 * c['found'] / max(1, c['tables']):.1f}%) | {c['tf_err']} | {c['err']} |")
    P(f"\nExcluded from the comparison: {dict(excluded)}.")
    P("\n## CPU time per table (GitHub-hosted runners, 4 vCPU; medians, p90)\n")
    def q(xs, f):
        xs = sorted(x for x in xs if x is not None)
        return xs[min(len(xs) - 1, int(f * (len(xs) - 1)))] if xs else None
    tim = {}
    for name, key, sub in (("TableFormer (docling stage, per table)", "ms_tf", None), ("TATR fp16 (pre + model + post, Python)", "ms_tatr", None),
                           ("TATR model only", "ms_tatr_model", None), ("PP-OCRv6 small on the crop (scans)", "ms_ocr", None),
                           ("PP-LCNet", "ms_lcnet", None), ("glyph-angle vote", "ms_vote", None)):
        xs = [META[r["id"]].get(key) for r in rows]
        tim[name] = (q(xs, .5), q(xs, .9))
    tim["selector features (Python)"] = (q(fms, .5), q(fms, .9))
    if logic_ms: tim["logic repairs (Python)"] = (q(list(logic_ms.values()), .5), q(list(logic_ms.values()), .9))
    P("| step | p50 ms | p90 ms |"); P("|---|---|---|")
    for k, (a, b) in tim.items(): P(f"| {k} | {'–' if a is None else f'{a:.0f}'} | {'–' if b is None else f'{b:.0f}'} |")
    summary["time_ms"] = tim
    cpus = collections.Counter(m.get("cpu") for m in META.values()); P(f"\nCPUs seen in the TATR jobs: {dict(cpus)}")
    sizes = {}
    for p in glob.glob(os.path.join(tf_root, "**", "sizes.json"), recursive=True): sizes.update(json.load(open(p)))
    for p in glob.glob(os.path.join(pub_root, "**", "summary.json"), recursive=True): sizes.update(json.load(open(p)).get("sizes", {}))
    P("\n**Model files (MB):** " + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in sorted(sizes.items())))
    summary["sizes"] = sizes

    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), default=str, indent=1)
    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "set", "type", "split", "bdoc", "bucket", "n_span", "turn", "v", "shape", "gtop", "gcon")} |
                               {k: r.get(k) for k in ("p_committed", "p_retrained", "p_retrained_logic", "pick_combo", "pick_combo_retrained", "logic_changed")}) + "\n")
    print("\n".join(md))


CAND = "tatr"


def put(r, a, s):
    r["v"][a] = S8.value(s); r["shape"][a] = float(bool(s.get("rc_exact"))) if s else None
    r["gtop"][a] = s.get("grits_top") if s else None; r["gcon"][a] = s.get("grits_con") if s else None; r["teds"][a] = s.get("teds") if s else None


if __name__ == "__main__":
    main()
