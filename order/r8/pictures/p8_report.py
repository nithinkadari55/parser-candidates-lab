"""Round 8, test A report: docling's picture classifier vs frozen-backbone heads on the shared label set.
Reads the per-item files of the model jobs (p8-items.jsonl, p8-docling.jsonl, p8-mnv3.jsonl, p8-dinov2.jsonl,
*-info.json, status-*.json) and writes the report with macro-F1, per-class F1, 95% cluster-bootstrap CIs
(resampling page / document / chart groups), paired differences, per-source accuracy, chart subtypes,
speed and size, and the decision.
usage: p8_report.py ARTDIR OUT.md [OUT.json]"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p8_common import CLASSES, D26, D26_SUB, D26_TO_SHARED, SOURCES, SUBTYPES  # noqa: E402

K = len(CLASSES)
B = 2000
HEADS = [("mnv3", "fp32"), ("dinov2", "fp32"), ("dinov2", "int8")]
CHART_IDX = [D26.index(c) for c in D26_SUB]


def load_jsonl(pattern):
    out = {}
    for f in glob.glob(pattern, recursive=True):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def docling_preds(p26):
    p = np.array(p26)
    raw = D26[int(p.argmax())]
    agg = np.zeros(K)
    for i, c in enumerate(D26):
        if c in D26_TO_SHARED:
            agg[CLASSES.index(D26_TO_SHARED[c])] += p[i]
    sub = D26_SUB[D26[CHART_IDX[int(p[CHART_IDX].argmax())]]]
    return {"docling_raw": D26_TO_SHARED.get(raw, "other:" + raw), "docling_restricted": CLASSES[int(agg.argmax())],
            "docling_raw26": raw, "docling_sub": sub, "docling_sub_raw": D26_SUB.get(raw)}


def counts(items, arm):
    """per-group TP/FP/FN per class -> (groups, G x 3K matrix)"""
    gi = {}
    rows = []
    for m in items:
        g = gi.setdefault(m["group"], len(gi))
        t = CLASSES.index(m["label"])
        p = m["pred"].get(arm)
        v = np.zeros(3 * K)
        if p == m["label"]:
            v[t] += 1
        else:
            v[2 * K + t] += 1
            if p in CLASSES:
                v[K + CLASSES.index(p)] += 1
        rows.append((g, v))
    M = np.zeros((len(gi), 3 * K))
    for g, v in rows:
        M[g] += v
    return M


def macro(S, present):
    tp, fp, fn = S[..., :K], S[..., K:2 * K], S[..., 2 * K:]
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), np.nan)
    f1 = f1[..., present]
    return np.nanmean(f1, axis=-1), f1


def evaluate(items, arms, seed=0, min_n=1):
    """macro-F1 with 95% cluster-bootstrap CI for each arm (same resamples for all arms) and per-class F1;
    macro over the classes with at least MIN_N labelled crops in the set"""
    if not items:
        return {}
    present = np.array([sum(m["label"] == c for m in items) >= min_n for c in CLASSES])
    mats = {a: counts(items, a) for a in arms}
    G = next(iter(mats.values())).shape[0]
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).astype(float)
    res = {}
    boots = {}
    for a, M in mats.items():
        point, f1 = macro(M.sum(0), present)
        bs, _ = macro(W @ M, present)
        boots[a] = bs
        res[a] = {"macro_f1": float(point), "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                  "per_class": {c: float(v) for c, v in zip([c for c, p in zip(CLASSES, present) if p], f1)},
                  "acc": float(np.mean([m["pred"].get(a) == m["label"] for m in items]))}
    res["_boots"] = boots
    res["_n"] = len(items)
    res["_groups"] = G
    res["_class_n"] = {c: sum(m["label"] == c for m in items) for c in CLASSES}
    return res


def fmt_ci(r):
    return f"{r['macro_f1']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"


def main():
    art, outp = sys.argv[1], sys.argv[2]
    outj = sys.argv[3] if len(sys.argv) > 3 else None
    items = load_jsonl(os.path.join(art, "**", "p8-items.jsonl"))
    doc = load_jsonl(os.path.join(art, "**", "p8-docling.jsonl"))
    heads = {k: load_jsonl(os.path.join(art, "**", f"p8-{k}.jsonl")) for k in ("mnv3", "dinov2")}
    infos = {}
    for f in glob.glob(os.path.join(art, "**", "p8-*-info.json"), recursive=True):
        d = json.load(open(f))
        infos[d["kind"]] = d
    status = {}
    for f in glob.glob(os.path.join(art, "**", "status-*.json"), recursive=True):
        status.update(json.load(open(f)))
    for i, m in items.items():
        m["pred"] = {}
        if i in doc:
            m["pred"].update(docling_preds(doc[i]["p26"]))
        for k, d in heads.items():
            if i in d:
                m["pred"].update({a: v for a, v in d[i].items() if a != "id"})
    all_items = list(items.values())
    arms = ["docling_restricted", "docling_raw"]
    for k, q in HEADS:
        for v in ("all", "perm"):
            a = f"{k}_{q}_{v}"
            if any(a in m["pred"] for m in all_items):
                arms.append(a)
    pool = [m for m in all_items if SOURCES[m["src"]][3] == "pool"]
    real = [m for m in all_items if SOURCES[m["src"]][3] == "test"]
    paper = [m for m in all_items if m["src"] in ("ubpmc", "scicap")]
    sets = {"pool": pool, "realdoc": real, "realdoc_doclaynet": [m for m in real if m["src"] == "realdoc_doclaynet"],
            "realdoc_omnidocbench": [m for m in real if m["src"] == "realdoc_omnidocbench"],
            "paper_sources": paper}
    E = {k: evaluate(v, [a for a in arms if all(a in m["pred"] for m in v)] if v else [],
                     min_n=10 if k.startswith("realdoc") else 1) for k, v in sets.items()}

    L = []
    P = L.append
    P("# Round 8, test A: picture classifier on a broad, multi-source set\n")
    P("Shared label set (classes that docling's DocumentFigureClassifier-v2.5 and the heads both support): "
      + ", ".join(CLASSES) + ".")
    P("docling's 26 classes map as: line/bar/pie/scatter/box chart -> chart; flow_chart, engineering_drawing, "
      "chemistry_structure -> diagram; photograph -> photo; screenshot_from_computer/manual -> screenshot; "
      "geographical/topographical_map -> map; table -> table_image; logo, signature, stamp as is. "
      "**docling (restricted)** = argmax of the summed probabilities of the shared classes (its best case on a "
      "closed set, used for the decision); **docling (raw)** = its top-1 class, where icon, other, page_thumbnail, "
      "qr_code, bar_code, music, calendar, full_page_image and crossword_puzzle count as wrong.\n")
    P("Heads: logistic regression (class-balanced, C by grouped CV inside each fold) on frozen features. All arms "
      "see the same crop resized to 224x224 (squash, bilinear), the way docling.rs feeds its classifier. "
      "**Held-out sources:** every source is predicted by a head that never saw it (leave-one-source-out). "
      "*_all = trained on all other sources; *_perm = trained only on the other permissively licensed sources "
      "(CC-BY, CC0/PD, Apache, MIT, CDLA) - the shippable candidate. Hand-labelled real-document crops are never "
      "trained on. CIs: 95% bootstrap over groups (page, document, paper or chart), 2,000 resamples.\n")

    # data table
    P("## Data\n")
    P("| source | role | permissive | licence (as read from the files) | crops kept | per class | status |")
    P("|---|---|---|---|---|---|---|")
    by_src = defaultdict(Counter)
    for m in all_items:
        by_src[m["src"]][m["label"]] += 1
    dropped = (infos.get("docling") or {}).get("dropped_duplicates", {})
    for s, (grp, lic, perm, role) in SOURCES.items():
        st = status.get(s, {}).get("status", "not run")
        c = by_src.get(s, Counter())
        P(f"| {s} | {role} | {'yes' if perm else 'no'} | {lic} | {sum(c.values())} | "
          + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + f" | {st[:120]} |")
    P(f"\nNear-duplicates removed across sources (first copy kept): {json.dumps(dropped)}\n")
    cls_src = defaultdict(set)
    cls_perm = defaultdict(set)
    for m in pool:
        cls_src[m["label"]].add(m["src"])
        if SOURCES[m["src"]][2]:
            cls_perm[m["label"]].add(m["src"])
    P("Sources per class (pool): " + "; ".join(f"{c}: {len(cls_src[c])} ({len(cls_perm[c])} permissive)"
                                                for c in CLASSES) + "\n")

    def table(name, title):
        e = E.get(name) or {}
        if not e:
            P(f"\n### {title}: no items\n")
            return
        P(f"\n### {title}: {e['_n']} crops, {e['_groups']} groups\n")
        P("Class counts: " + ", ".join(f"{c} {n}" for c, n in e["_class_n"].items() if n) + "\n")
        cl = [c for c in CLASSES if e["_class_n"][c]]
        P("| arm | macro-F1 [95% CI] | accuracy | " + " | ".join(cl) + " |")
        P("|---|---|---|" + "---|" * len(cl))
        for a in [x for x in arms if x in e]:
            r = e[a]
            P(f"| {a} | {fmt_ci(r)} | {r['acc']:.3f} | " + " | ".join(f"{r['per_class'].get(c, float('nan')):.2f}"
                                                                     for c in cl) + " |")
        base = "docling_restricted"
        if base in e:
            P("\nPaired difference vs docling (restricted), same bootstrap resamples:\n")
            P("| arm | macro-F1 - docling [95% CI] |")
            P("|---|---|")
            for a in [x for x in arms if x in e and x != base]:
                d = e[a]["macro_f1"] - e[base]["macro_f1"]
                bd = e["_boots"][a] - e["_boots"][base]
                P(f"| {a} | {d:+.3f} [{np.nanpercentile(bd, 2.5):+.3f}, {np.nanpercentile(bd, 97.5):+.3f}] |")

    P("\n## Results\n")
    table("pool", "Held-out sources (leave-one-source-out, all pool sources)")
    table("realdoc", "Real-document picture crops (DocLayNet test + OmniDocBench, hand-labelled, never trained on)")
    table("realdoc_doclaynet", "  of which DocLayNet test pictures")
    table("realdoc_omnidocbench", "  of which OmniDocBench figures")
    table("paper_sources", "Figures cropped from real papers with source labels (UB-PMC, SciCap)")

    # per-source accuracy
    P("\n## Accuracy per held-out source\n")
    P("| source | crops | " + " | ".join(arms) + " |")
    P("|---|---|" + "---|" * len(arms))
    for s in sorted(by_src):
        ms = [m for m in all_items if m["src"] == s]
        P(f"| {s} | {len(ms)} | " + " | ".join(
            (f"{np.mean([m['pred'].get(a) == m['label'] for m in ms]):.2f}" if all(a in m['pred'] for m in ms) else "-")
            for a in arms) + " |")

    # confusion on real documents
    for a in ["docling_restricted", "docling_raw"] + [x for x in arms if x.endswith("_perm")]:
        if not real or not all(a in m["pred"] for m in real):
            continue
        cols = CLASSES + sorted({m["pred"][a] for m in real} - set(CLASSES))
        P(f"\n**Real-document crops, confusion of {a}** (rows = label, columns = prediction):\n")
        P("| label | " + " | ".join(cols) + " |")
        P("|---|" + "---|" * len(cols))
        for c in CLASSES:
            rs = [m for m in real if m["label"] == c]
            if rs:
                cc = Counter(m["pred"][a] for m in rs)
                P(f"| {c} ({len(rs)}) | " + " | ".join(str(cc.get(x, 0) or "") for x in cols) + " |")
    if real:
        P("\ndocling raw 26-class top-1 on the real-document crops: "
          + json.dumps(Counter(m["pred"].get("docling_raw26") for m in real).most_common()))

    # chart subtypes
    sub_items = [m for m in pool if m["label"] == "chart" and m.get("sub") in SUBTYPES]
    if sub_items:
        P("\n## Chart subtype (bar / line / pie / scatter / box), chart crops whose source gives it\n")
        P(f"{len(sub_items)} crops; by source " + json.dumps(Counter(m["src"] for m in sub_items)) + "\n")
        P("| arm | accuracy | " + " | ".join(SUBTYPES) + " |")
        P("|---|---|" + "---|" * len(SUBTYPES))
        sub_arms = [("docling top-1 (any of 26)", "docling_sub_raw"), ("docling restricted to its 5 chart classes", "docling_sub")]
        sub_arms += [(f"{k} {q} head (leave-one-source-out)", f"{k}_{q}_sub") for k, q in HEADS]
        for name, a in sub_arms:
            ms = [m for m in sub_items if a in m["pred"]]
            if not ms:
                continue
            per = [np.mean([m["pred"][a] == s for m in ms if m["sub"] == s]) if any(m["sub"] == s for m in ms) else float("nan")
                   for s in SUBTYPES]
            P(f"| {name} | {np.mean([m['pred'][a] == m['sub'] for m in ms]):.3f} | " + " | ".join(f"{v:.2f}" for v in per) + " |")

    # speed and size
    P("\n## Size and speed (CPU, 1 thread, per crop, GitHub runner)\n")
    P("| model | MB | ms p50 | ms p95 | graph = feature path |")
    P("|---|---|---|---|---|")
    if "docling" in infos:
        d = infos["docling"]
        P(f"| docling DocumentFigureClassifier-v2.5 | {d['mb']:.1f} | {d['ms_p50']:.1f} | {d['ms_p95']:.1f} | - |")
    for k, q in HEADS:
        g = (infos.get(k) or {}).get(f"graph_{q}")
        if g:
            P(f"| {k} {q} + head | {g['mb']:.1f} | {g['ms_p50']:.1f} | {g['ms_p95']:.1f} | {g['agree_with_feature_path']:.3f} |")

    # decision
    P("\n## Decision\n")
    P("Rule: replace docling's classifier only if the shippable (permissive-data) head beats it clearly - its "
      "macro-F1 CI lies above docling's CI - on the shared classes over held-out sources AND on the real-document "
      "crops. docling is scored at its best (restricted).\n")
    verdict = {}
    for k, q in HEADS:
        a = f"{k}_{q}_perm"
        ok = []
        for sname in ("pool", "realdoc"):
            e = E.get(sname) or {}
            if a in e and "docling_restricted" in e:
                ok.append(e[a]["ci"][0] > e["docling_restricted"]["ci"][1])
            else:
                ok.append(None)
        verdict[a] = ok
        P(f"- {a}: held-out sources {'clear win' if ok[0] else 'no clear win' if ok[0] is False else 'n/a'}; "
          f"real-document crops {'clear win' if ok[1] else 'no clear win' if ok[1] is False else 'n/a'}.")
    winners = [a for a, ok in verdict.items() if ok and all(ok)]
    P("\n**Verdict: " + (f"replace docling's classifier; candidates that pass: {', '.join(winners)}."
                         if winners else "keep docling's classifier (no shippable head passes both conditions).") + "**")
    text = "\n".join(L) + "\n"
    open(outp, "w", encoding="utf-8").write(text)
    if outj:
        js = {k: {a: {kk: vv for kk, vv in r.items()} for a, r in v.items() if not a.startswith("_")}
              for k, v in E.items() if v}
        json.dump({"eval": js, "verdict": verdict, "infos": {k: {kk: vv for kk, vv in v.items() if kk != "C"}
                                                               for k, v in infos.items()}}, open(outj, "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
