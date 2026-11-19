import glob, json, math, os, re, sys
from collections import Counter, defaultdict
from PIL import Image
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnorm import norm_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from taxonomy import difficulty, tags


def bleu4(refs, hyps):
    pn, pd = [0] * 4, [0] * 4
    rl = hl = 0
    for r, h in zip(refs, hyps):
        rl += len(r)
        hl += len(h)
        for n in range(1, 5):
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            pn[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            pd[n - 1] += max(len(h) - n + 1, 0)
    if hl == 0 or min(pn) == 0:
        return 0.0
    lp = sum(math.log(pn[i] / pd[i]) for i in range(4)) / 4
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return bp * math.exp(lp)


def load_items(names):
    items = []
    for m in names:
        p = os.path.join(ROOT, "manifests", m + ".json")
        if os.path.exists(p):
            items += json.load(open(p, encoding="utf-8"))
    return items


def load_preds(m):
    d = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "acc", m + ".s*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            d.setdefault(r["id"], r)
    return d


def score(gt_n, pred):
    pn = norm_tokens(pred)
    a, b = "".join(gt_n), "".join(pn)
    return pn, a == b, 1 - Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def main():
    models = sys.argv[1].split(",")
    items = load_items(["own12", "arxiv", "unimer"])
    preds = {m: load_preds(m) for m in models}
    dropped = [m for m in models if sum(1 for it in items if it["id"] in preds[m]) < 0.5 * len(items)]
    models = [m for m in models if m not in dropped]
    common = [it for it in items if all(it["id"] in preds[m] for m in models)]
    res = {"n_items": len(items), "coverage": {m: sum(1 for it in items if it["id"] in preds[m]) for m in models},
           "n_common": len(common), "dropped_low_coverage": dropped, "groups": {}, "items": []}
    acc = defaultdict(lambda: defaultdict(lambda: {"n": 0, "em": 0, "ed": 0.0, "refs": [], "hyps": [], "runaway": 0, "empty": 0, "error": 0}))
    for it in common:
        gt_n = norm_tokens(it["gt"])
        diff = difficulty(it["gt"])
        tg = tags(it)
        src = "arxiv" if it["set"] == "arxiv" else ("unimer" if it["set"].startswith("unimer") else "own12")
        groups = ["ALL", "set:" + it["set"], "diff:" + diff, f"diff:{diff}|{src}"] + ["tag:" + x for x in tg]
        row = {"id": it["id"], "set": it["set"], "diff": diff, "tags": tg, "gt": it["gt"], "pred": {}}
        for m in models:
            r = preds[m][it["id"]]
            pn, em, ed = score(gt_n, r["pred"])
            row["pred"][m] = {"out": r["pred"], "em": em, "ed": round(ed, 4), "runaway": bool(r["runaway"]), "s": r["s"]}
            for g in groups:
                A = acc[g][m]
                A["n"] += 1
                A["em"] += em
                A["ed"] += ed
                A["refs"].append(gt_n)
                A["hyps"].append(pn)
                A["runaway"] += bool(r["runaway"])
                A["empty"] += not r["pred"].strip()
                A["error"] += "error" in r
        res["items"].append(row)
    for g, d in acc.items():
        res["groups"][g] = {m: {"n": A["n"], "exact%": round(100 * A["em"] / A["n"], 1), "edit": round(A["ed"] / A["n"], 4),
                                "bleu4": round(bleu4(A["refs"], A["hyps"]), 4), "runaway": A["runaway"], "empty": A["empty"],
                                "error": A["error"]} for m, A in d.items()}
    if "p2t15_int8" in models and "p2t15_int8_split" in preds:
        a, b = preds["p2t15_int8"], preds["p2t15_int8_split"]
        both = [i for i in a if i in b]
        res["split_identity"] = {"compared": len(both), "identical": sum(a[i]["pred"] == b[i]["pred"] for i in both),
                                 "differs": [i for i in both if a[i]["pred"] != b[i]["pred"]][:20]}
    # docTR protocol comparison: raw whitespace-token metrics on the full UniMER-Test (with HWE)
    uni = load_items(["unimer", "unimer_hwe"])
    res["raw_token_unimer"] = {}
    for m in models:
        if not m.startswith("doctr"):
            continue
        rows = [(it, preds[m][it["id"]]) for it in uni if it["id"] in preds[m]]
        for scope in ["spe+cpe+sce", "spe+cpe+sce+hwe"]:
            sel = [(it, r) for it, r in rows if scope.endswith("hwe") or it["set"] != "unimer_hwe"]
            if not sel:
                continue
            refs = [it["gt"].split() for it, _ in sel]
            hyps = [r["pred"].split() for _, r in sel]
            em = sum(a == b for a, b in zip(refs, hyps)) / len(sel)
            es = sum(1 - Levenshtein.distance(a, b) / max(len(a), len(b), 1) for a, b in zip(refs, hyps)) / len(sel)
            nem = sum(score(norm_tokens(it["gt"]), r["pred"])[1] for it, r in sel) / len(sel)
            res["raw_token_unimer"][f"{m} {scope}"] = {"n": len(sel), "token_exact%": round(100 * em, 1),
                                                       "token_edit_sim": round(es, 4), "bleu4_raw_tokens": round(bleu4(refs, hyps), 4),
                                                       "normalised_exact%": round(100 * nem, 1)}
    json.dump(res, open(os.path.join(ROOT, "results", "eval.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({k: v for k, v in res.items() if k != "items"}, open(os.path.join(ROOT, "results", "metrics.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("items", "groups")}, indent=1))


if __name__ == "__main__":
    main()
