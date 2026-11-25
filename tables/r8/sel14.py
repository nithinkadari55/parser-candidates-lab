"""Milestone 14 (after review): the harness's table selector retrained on the licence-allowed round-8 tables only.

Data: the round-8 cells and words (run 45676430603, Actions caches r8priv-*) and per-table scores (artifacts r8pub-*).
Features: the committed selector's 56 (round-3 feats() + agreement + n_words) on the words the harness reads at run time
(docling-pdf word cells on text-layer pages, PP-OCRv6 small words on scans). Same XGBoost settings (r5_sel.PARAMS).

Training rows: unrotated tables with TATR output from the sets in licences14.TRAIN. WTW, cTDaR and ICDAR-2013
(licences14.EVAL_ONLY) are only evaluated on: never trained on, never used to choose a variant.

Variants: PubTabNet (answer-key words: optimistic) and SynFinTabs (synthetic) weighted 1, 0.5, 0.25 or 0.
Evaluation, per table:
  cv    5-fold GroupKFold by document over the training rows (out-of-fold picks);
  lodo  leave one dataset out: the model trained on the other allowed sets picks for this set's tables;
  evaluation-only sets get the model trained on every training row (they are held out by construction).
Rule fixed in advance: the variant with the best LODO TEDS-Struct on the dev split of the allowed real-PDF-word sets
(r4pt, r4scitsr, r4brgm, pt2, ftnc) wins; a variant within MIN_GAIN of the best that uses more training data is
preferred. Everything is reported on the test split, per set and on the real-PDF groups, against TableFormer alone, TATR
alone and the round-5 selector on main (whose trees saw r4pt, r4scitsr, r4brgm and r4icdar in training).

Scanned tables (docling crops of scan sets): the run-time rule for a crop PP-LCNet turns (label != 0, confidence >=
the gate chosen by gate14.py) is compared three ways: forced TATR on the turned crop (prefer_tatr_when_rotated), the
selector between TableFormer and TATR on the turned crop, and no PP-LCNet (the selector on the crop as is).

The final model of the chosen variant is fitted on every training row and exported for the Rust evaluator with an
equality check. Fixtures for the harness: selector_fixture.json (80 training rows, xgboost margins and probabilities)
and features_fixture.json (12 tables of r4pt, pt2 and ftnc with cells, words and the 56 features), exactly as committed
in the harness (crates/contextual-worker/tests/fixtures/tables/).

usage: sel14.py PUB_ROOT PRIV_ROOT COMMITTED_SELECTOR OUT_DIR"""
import os, sys, json, gzip, glob, math, random, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import numpy as np
import score8 as S8
import r3_select as R3
from r5_sel import leaf_sum, PARAMS, export
from licences14 import TRAIN, EVAL_ONLY, REAL_PDF, REAL_WORDS, source_set
import gate14
from multiprocessing import Pool

SELECTOR_SHA = "c27d7295112ab514e46cf0a1595782468c932b06084d10ad3247d093a2ed34f1"
WEAK = ("ptn", "synfin")
VARIANTS = {"w1": 1.0, "w0.5": 0.5, "w0.25": 0.25, "w0": 0.0}
MIN_GAIN = 0.0003
DECIDE = [s for s in REAL_WORDS if s in TRAIN]
META, PRIV = {}, {}


def base_doc(doc): return doc.split(":", 1)[1] if ":" in doc else doc


def _feat(job):
    tid, alt = job
    p = PRIV[tid]
    if alt:
        words = [dict(text=w["text"], bbox=w["bbox"]) for w in (p.get("words_alt") or {}).get("tatr_lcnet", p["words"])]
        cb = p["cells"]["tatr_lcnet"]
    else:
        words = [dict(text=w["text"], bbox=w["bbox"]) for w in p["words"]]
        cb = p["cells"]["tatr"]
    ca = p["cells"]["tf"]
    fa, fb = R3.feats(ca, words), R3.feats(cb, words)
    x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
    x.update({"d_" + k: fb[k] - fa[k] for k in fa})
    x["agree"] = R3.agreement(ca, cb); x["n_words"] = len(words)
    return tid, alt, x


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def paired(rows, a, b, key="v"):
    rr = [dict(doc=r["bdoc"], x=r[key][a], y=r[key][b]) for r in rows if r[key].get(a) is not None and r[key].get(b) is not None]
    if not rr: return None
    c = cluster_boot(rr, "x", "y", group="doc")
    return dict(n=len(rr), diff=c[0], lo=c[1], hi=c[2], p=wilcoxon_p([r["x"] for r in rr], [r["y"] for r in rr]))


def main():
    pub_root, priv_root, committed, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    for p in glob.glob(os.path.join(pub_root, "**", "meta.jsonl"), recursive=True):
        for r in jl(p): META[r["id"]] = r
    for p in glob.glob(os.path.join(priv_root, "**", "priv.jsonl.gz"), recursive=True):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line); PRIV[r["id"]] = r
    assert hashlib.sha256(open(committed, "rb").read()).hexdigest() == SELECTOR_SHA
    forest = json.load(open(committed)); keys = forest["features"]
    gate, _ = gate14.choose(list(META.values()))
    rows = []
    for tid, m in META.items():
        sc = m.get("scores") or {}
        if not m["found"] or m.get("err") or tid not in PRIV or "tf" not in sc or "tatr" not in sc: continue
        if S8.value(sc["tf"]) is None or S8.value(sc["tatr"]) is None: continue
        src = source_set(tid, m["set"])
        r = dict(id=tid, set=m["set"], src=src, type=m["type"], split=m["split"], bdoc=base_doc(m["doc"]), turn=m.get("turn") or 0,
                 empty=not PRIV[tid]["cells"]["tatr"], allowed=src in TRAIN,
                 v=dict(tf=S8.value(sc["tf"]), tatr=S8.value(sc["tatr"])),
                 s=dict(tf=float(bool(sc["tf"].get("rc_exact"))), tatr=float(bool(sc["tatr"].get("rc_exact")))))
        lab, conf = m.get("lcnet") or 0, m.get("lcnet_conf") or 0.0
        if set_kind(m["set"]) == "scan" and lab and conf >= gate and "tatr_lcnet" in sc and S8.value(sc["tatr_lcnet"]) is not None \
                and PRIV[tid]["cells"].get("tatr_lcnet"):
            r["lcnet_turn"] = lab
            r["v"]["tatr_lcnet"] = S8.value(sc["tatr_lcnet"]); r["s"]["tatr_lcnet"] = float(bool(sc["tatr_lcnet"].get("rc_exact")))
        rows.append(r)
    print("rows", len(rows), "gate", gate, flush=True)
    jobs = [(r["id"], False) for r in rows] + [(r["id"], True) for r in rows if "lcnet_turn" in r]
    with Pool(4) as pool:
        feats = {(tid, alt): x for tid, alt, x in pool.imap_unordered(_feat, jobs, chunksize=8)}
    for r in rows:
        r["x"] = feats[(r["id"], False)]
        if "lcnet_turn" in r: r["x_lcnet"] = feats[(r["id"], True)]
        m = float(np.float32(forest["base_margin"]) + leaf_sum(forest["trees"], np.array([r["x"][k] for k in keys], np.float32)))
        r["p"] = {"r5": 1 / (1 + math.exp(-m))}
    vec = lambda x: [x[k] for k in keys]
    tr = [r for r in rows if r["allowed"] and r["turn"] == 0 and not r["empty"] and r["set"] != "rotpdf"]
    assert all(r["src"] in TRAIN for r in tr)
    import xgboost as XG
    from sklearn.model_selection import GroupKFold
    X = np.array([vec(r["x"]) for r in tr], float)
    y = np.array([1 if r["v"]["tatr"] > r["v"]["tf"] + 1e-9 else 0 for r in tr]); g = np.array([r["bdoc"] for r in tr])
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    fold_of = {}
    for k, (_, b) in enumerate(folds):
        for i in b: fold_of[tr[i]["bdoc"]] = k
    in_tr = {id(r) for r in tr}
    others = [r for r in rows if id(r) not in in_tr]
    by_src = collections.defaultdict(list)
    for i, r in enumerate(tr): by_src[r["src"]].append(i)

    def fit(idx, w):
        idx = [i for i in idx if w[i] > 0]
        return XG.XGBClassifier(**PARAMS).fit(X[idx], y[idx], sample_weight=w[idx])

    def predict(model, xs):
        return model.predict_proba(np.array(xs, float))[:, 1] if xs else []
    models = {}
    for name, wv in VARIANTS.items():
        w = np.array([wv if r["src"] in WEAK else 1.0 for r in tr])
        everything = list(range(len(tr)))
        final = fit(everything, w)
        fm = [fit(list(a), w) for a, _ in folds]
        models[name] = dict(final=final, folds=fm, w=w)
        cv = np.zeros(len(tr))
        for k, (_, b) in enumerate(folds): cv[b] = predict(fm[k], [X[i] for i in b])
        lodo = np.zeros(len(tr))
        lodo_models = {s: fit([i for i in everything if tr[i]["src"] != s], w) for s in TRAIN}
        for s, idx in by_src.items():
            lodo[idx] = predict(lodo_models[s], [X[i] for i in idx])
        for i, r in enumerate(tr): r["p"][name + "_cv"] = float(cv[i]); r["p"][name + "_lodo"] = float(lodo[i])
        # tables that are not training rows: rotated, TATR empty, evaluation-only sets. Allowed rows of a training
        # document take the fold model that did not see the document, and the LODO model of their set.
        for r in others:
            xs = [vec(r["x"])]
            if r["allowed"] and r["bdoc"] in fold_of:
                pc = float(predict(fm[fold_of[r["bdoc"]]], xs)[0])
            else:
                pc = float(predict(final, xs)[0])
            if r["allowed"]:
                pl = float(predict(lodo_models[r["src"]], xs)[0])
            else:
                pl = pc
            r["p"][name + "_cv"] = pc; r["p"][name + "_lodo"] = pl
        # the selector on the PP-LCNet-turned crop (scans), with the same held-out models
        for r in rows:
            if "x_lcnet" not in r: continue
            xs = [vec(r["x_lcnet"])]
            model = fm[fold_of[r["bdoc"]]] if r["allowed"] and r["bdoc"] in fold_of else final
            r["p"][name + "_cv_lcnet"] = float(predict(model, xs)[0])
        print("variant", name, "done", flush=True)

    def pick(r, key):
        if r["empty"]: return "tf"
        if r["turn"]: return "tatr"
        return "tatr" if r["p"][key] > 0.5 else "tf"
    arms = ["tf", "tatr", "r5"] + [n + e for n in VARIANTS for e in ("_cv", "_lodo")]
    for r in rows:
        for a in arms[2:]:
            arm = pick(r, a); r["v"][a] = r["v"][arm]; r["s"][a] = r["s"][arm]

    def table(sub, names):
        return {a: dict(teds_s=mean([r["v"][a] for r in sub]), shape=mean([r["s"][a] for r in sub])) for a in names}

    def grouped(split, names):
        sub = [r for r in rows if r["split"] == split]
        res = collections.OrderedDict()
        for label, f in (("all allowed sets", lambda r: r["allowed"]),
                         ("real PDFs, allowed (r4pt, r4scitsr, r4brgm)", lambda r: r["set"] in REAL_PDF and r["allowed"]),
                         ("real PDFs incl. ICDAR-2013 (eval only)", lambda r: r["set"] in REAL_PDF),
                         ("real PDF words, allowed (+ pt2, ftnc)", lambda r: r["set"] in REAL_WORDS and r["allowed"]),
                         ("real PDF words incl. ICDAR-2013", lambda r: r["set"] in REAL_WORDS),
                         ("scans, allowed (tabrec)", lambda r: r["set"] == "tabrec"),
                         ("evaluation-only sets (wtw, ctdarm, ctdarh, r4icdar)", lambda r: not r["allowed"]),
                         ("everything", lambda r: True)):
            st = [r for r in sub if f(r)]
            if st: res[label] = dict(n=len(st), **table(st, names))
        for s in ALL_SETS:
            st = [r for r in sub if r["set"] == s]
            if st: res["set " + s] = dict(n=len(st), **table(st, names))
        for t in TYPES:
            st = [r for r in sub if r["type"] == t and r["allowed"]]
            if st: res["type " + t + " (allowed)"] = dict(n=len(st), **table(st, names))
        return res
    dev, test = grouped("dev", arms), grouped("test", arms)
    # the variant, on the dev split of the allowed real-PDF-word sets, LODO
    dsub = [r for r in rows if r["split"] == "dev" and r["set"] in DECIDE]
    score = {n: mean([r["v"][n + "_lodo"] for r in dsub]) for n in VARIANTS}
    best = max(score.values())
    choice = next(n for n in VARIANTS if score[n] >= best - MIN_GAIN)
    ch_cv, ch_lodo = choice + "_cv", choice + "_lodo"
    tsub = [r for r in rows if r["split"] == "test"]
    comparisons = {}
    for label, f in (("real PDF words, allowed", lambda r: r["set"] in REAL_WORDS and r["allowed"]),
                     ("real PDFs, allowed", lambda r: r["set"] in REAL_PDF and r["allowed"]),
                     ("real PDFs incl. ICDAR-2013", lambda r: r["set"] in REAL_PDF),
                     ("evaluation-only sets", lambda r: not r["allowed"]),
                     ("all allowed sets", lambda r: r["allowed"]),
                     ("everything", lambda r: True)):
        st = [r for r in tsub if f(r)]
        comparisons[label] = {f"{a} - {b} {k2}": paired(st, a, b, key) for a in (ch_cv, ch_lodo) for b in ("tf", "tatr", "r5")
                              for key, k2 in (("v", "teds_s"), ("s", "shape"))}
    # scanned tables PP-LCNet turns: forced TATR vs the selector on the turned crop vs no turn
    scan = {}
    for split in ("dev", "test"):
        st = [r for r in rows if r["split"] == split and set_kind(r["set"]) == "scan" and r["turn"] == 0]
        for r in st:
            base = pick(r, ch_cv)
            r["v"]["no_turn"] = r["v"][base]; r["s"]["no_turn"] = r["s"][base]
            if "lcnet_turn" in r:
                r["v"]["forced"] = r["v"]["tatr_lcnet"]; r["s"]["forced"] = r["s"]["tatr_lcnet"]
                arm = "tatr_lcnet" if r["p"][ch_cv + "_lcnet"] > 0.5 else "tf"
                r["v"]["selector_turned"] = r["v"][arm]; r["s"]["selector_turned"] = r["s"][arm]
                r["selector_turned_arm"] = arm
            else:
                for k in ("forced", "selector_turned"):
                    r["v"][k] = r["v"]["no_turn"]; r["s"][k] = r["s"]["no_turn"]
        res = {}
        for label, f in (("allowed (tabrec)", lambda r: r["allowed"]), ("evaluation-only (wtw, ctdarm, ctdarh)", lambda r: not r["allowed"]),
                         ("all scans", lambda r: True)):
            sub = [r for r in st if f(r)]
            turned = [r for r in sub if "lcnet_turn" in r]
            if not sub: continue
            res[label] = dict(n=len(sub), n_turned=len(turned),
                              all=table(sub, ["tf", "no_turn", "forced", "selector_turned"]),
                              turned=table(turned, ["tf", "no_turn", "forced", "selector_turned"]) if turned else None,
                              selector_keeps_tf=(sum(1 for r in turned if r["selector_turned_arm"] == "tf") / len(turned)) if turned else None,
                              forced_vs_no_turn=paired(sub, "forced", "no_turn"), selector_vs_no_turn=paired(sub, "selector_turned", "no_turn"),
                              selector_vs_forced=paired(sub, "selector_turned", "forced"),
                              forced_vs_no_turn_shape=paired(sub, "forced", "no_turn", "s"), selector_vs_forced_shape=paired(sub, "selector_turned", "forced", "s"))
        scan[split] = res
    # final model, export, fixtures
    final = models[choice]["final"]
    used = [i for i in range(len(tr)) if models[choice]["w"][i] > 0]
    Xu = X[used]
    model_json, margins, err = export(final, keys, Xu)
    sets_used = sorted({tr[i]["src"] for i in used})
    model_json.update(words="docling-pdf 2.84.0 PdfPage::word_cells (text-layer pages); PP-OCRv6 small det + rec on the crop (scans)",
                      training="parser-candidates-lab tables/r8/sel14.py on tables-r8 run 45676430603: %d tables, %d documents, sets %s "
                               "(licence-allowed only; %s weight %g)" % (len(used), len({tr[i]["bdoc"] for i in used}), ", ".join(sets_used),
                                                                         "/".join(WEAK), VARIANTS[choice]),
                      training_sets={s: TRAIN[s][1] for s in sets_used}, tatr_rows_joined="none", split_row_features=False, extra_features=[])
    json.dump(model_json, open(os.path.join(out, "table_selector.json"), "w"), separators=(",", ":"))
    probs = final.predict_proba(Xu)[:, 1]
    rng = random.Random(14); idx = sorted(rng.sample(range(len(used)), min(80, len(used))))
    json.dump(dict(features=keys, rows=[dict(id=tr[used[i]]["id"], x=[float(v) for v in Xu[i]], margin=float(margins[i]), prob=float(probs[i])) for i in idx]),
              open(os.path.join(out, "selector_fixture.json"), "w"))
    # feature fixture: tables of the CDLA-Permissive sets only (their words may be published), the 56 features
    small = [r for r in tr if r["set"] in ("r4pt", "pt2", "ftnc") and 3 <= len(PRIV[r["id"]]["words"]) <= 60
             and 0 < len(PRIV[r["id"]]["cells"]["tf"]) <= 80 and 0 < len(PRIV[r["id"]]["cells"]["tatr"]) <= 80]
    rng.shuffle(small); fx = []; per = collections.Counter()
    for r in small:
        if per[r["set"]] < 4: fx.append(r); per[r["set"]] += 1
        if len(fx) >= 12: break
    strip = lambda cs: [dict(r0=c["r0"], r1=c["r1"], c0=c["c0"], c1=c["c1"], bbox=c["bbox"]) for c in cs]
    json.dump(dict(tables=[dict(id=r["id"], tableformer=strip(PRIV[r["id"]]["cells"]["tf"]), tatr=strip(PRIV[r["id"]]["cells"]["tatr"]),
                                words=[dict(text=w["text"], bbox=w["bbox"]) for w in PRIV[r["id"]]["words"]], features={k: r["x"][k] for k in keys}) for r in fx]),
              open(os.path.join(out, "features_fixture.json"), "w"))
    for name in ("selector_fixture.json", "features_fixture.json"):
        ids = [t["id"] for t in json.load(open(os.path.join(out, name)))["rows" if name.startswith("selector") else "tables"]]
        assert all(source_set(i, i.split("_")[0]) in TRAIN for i in ids), name
    # per-table picks and scores (no table text), for checks outside this job
    with gzip.open(os.path.join(out, "sel14_rows.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "set", "src", "type", "split", "bdoc", "turn", "empty", "allowed", "v", "s", "p") if k in r}
                               | {"lcnet_turn": r.get("lcnet_turn")}) + "\n")
    summary = dict(rows=len(rows), training=len(tr), used=len(used), docs=len(set(g)), gate=gate, variant_scores_dev_lodo=score, choice=choice,
                   training_sets=sets_used, dev=dev, test=test, comparisons=comparisons, scans=scan, export_max_margin_diff=err,
                   picks_tatr_test={a: mean([1.0 if pick(r, a) == "tatr" else 0.0 for r in tsub]) for a in arms[2:]})
    json.dump(summary, open(os.path.join(out, "sel14_summary.json"), "w"), indent=1)
    write_md(out, summary, arms, choice, gate)


NAMES = dict(tf="TableFormer", tatr="TATR", r5="round-5 selector (main)")


def name(a):
    if a in NAMES: return NAMES[a]
    v, e = a.rsplit("_", 1)
    return f"{v} {e}"


def write_md(out, S, arms, choice, gate):
    f4 = lambda x: "–" if x is None else f"{x:.4f}"
    pc = lambda x: "–" if x is None else f"{100 * x:.1f}"
    show = ["tf", "tatr", "r5", choice + "_cv", choice + "_lodo"]
    md = ["# Milestone 14: table selector retrained on the licence-allowed round-8 tables", "",
          f"Training rows: {S['used']} tables ({S['docs']} documents before weighting) from {', '.join(S['training_sets'])}; WTW, cTDaR and ICDAR-2013 are evaluated on only.",
          f"Variant (PubTabNet + SynFinTabs weight) chosen on the dev split, LODO TEDS-Struct of the allowed real-PDF-word sets: "
          + ", ".join(f"{n} {v:.4f}" for n, v in S["variant_scores_dev_lodo"].items()) + f" -> **{choice}**.", "",
          "cv: 5-fold GroupKFold by document over the training rows; lodo: the model trained without the table's set. Evaluation-only sets: the model trained on all training rows. The round-5 selector saw r4pt, r4scitsr, r4brgm and r4icdar in training (in-sample there).", ""]
    for split in ("test", "dev"):
        md += [f"## {split} split: TEDS-Struct / exact shape %", "", "| group | n | " + " | ".join(name(a) for a in show) + " |", "|---|---|" + "---|" * len(show)]
        for g, v in S[split].items():
            md.append(f"| {g} | {v['n']} | " + " | ".join(f"{f4(v[a]['teds_s'])} / {pc(v[a]['shape'])}" for a in show) + " |")
        md.append("")
    md += ["## All variants, test split (TEDS-Struct / exact shape %)", "", "| group | n | " + " | ".join(name(a) for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
    for g, v in S["test"].items():
        if g.startswith("set ") or g.startswith("type "): continue
        md.append(f"| {g} | {v['n']} | " + " | ".join(f"{f4(v[a]['teds_s'])} / {pc(v[a]['shape'])}" for a in arms) + " |")
    md += ["", "## Paired, test split (points; 95% CI by document; Wilcoxon p)", ""]
    for g, cs in S["comparisons"].items():
        md.append(f"**{g}**")
        md += [f"- {k}: {100 * v['diff']:+.2f} [{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] p={v['p']:.2g} (n={v['n']})" for k, v in cs.items() if v]
        md.append("")
    md += [f"## Scanned tables PP-LCNet turns (gate {gate:.2f}, docling crops)", "",
           "no_turn: the selector on the crop as is; forced: TATR on the turned crop (prefer_tatr_when_rotated); selector_turned: the selector between TableFormer and TATR on the turned crop.", ""]
    for split, res in S["scans"].items():
        for g, v in res.items():
            md.append(f"**{split}, {g}**: {v['n']} tables, {v['n_turned']} turned; selector keeps TableFormer on {pc(v['selector_keeps_tf'])}% of turned")
            for part in ("all", "turned"):
                if v[part]: md.append(f"- {part}: " + ", ".join(f"{k} {f4(x['teds_s'])} / {pc(x['shape'])}" for k, x in v[part].items()))
            for k in ("forced_vs_no_turn", "selector_vs_no_turn", "selector_vs_forced", "forced_vs_no_turn_shape", "selector_vs_forced_shape"):
                x = v[k]
                if x: md.append(f"- {k}: {100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}] p={x['p']:.2g}")
            md.append("")
    md += [f"Exported {S['export_max_margin_diff']:.2e} max |margin difference|; TATR picked on the test split: " + json.dumps({k: round(v, 3) for k, v in S["picks_tatr_test"].items()})]
    open(os.path.join(out, "sel14.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
