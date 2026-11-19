import json, math, os, re, sys
from collections import Counter, defaultdict
from PIL import Image
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnorm import norm_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = []
MANIFESTS = ["own12.json", "unimer.json", "arxiv.json"]

MEDIUM = {r"\frac", r"\sqrt", r"\sum", r"\int", r"\iint", r"\iiint", r"\oint", r"\prod", r"\binom", r"\bigcup",
          r"\bigcap", r"\coprod", r"\over", r"\overset", r"\underset", r"\stackrel", r"\bigoplus", r"\bigotimes",
          r"\choose", r"\lim", r"\sup", r"\inf", r"\max", r"\min", r"\limsup", r"\liminf"}
GREEK = set(r"\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \vartheta \iota \kappa \lambda \mu \nu \xi \pi \varpi \rho \varrho \sigma \varsigma \tau \upsilon \phi \varphi \chi \psi \omega \Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega".split())


def raw_tokens(s):
    return re.findall(r"\\[a-zA-Z]+\*?|\\.|[^\s]", s)


def _nested(s):
    t = raw_tokens(s)
    n = len(t)
    i = 0
    while i < n:
        if t[i] in (r"\frac", r"\dfrac", r"\tfrac", r"\cfrac"):
            j = i + 1
            for _ in range(2):
                if j < n and t[j] == "{":
                    d = 0
                    k = j
                    while k < n:
                        if t[k] == "{":
                            d += 1
                        elif t[k] == "}":
                            d -= 1
                            if d == 0:
                                break
                        if t[k] in (r"\frac", r"\dfrac", r"\tfrac", r"\cfrac") and k > j:
                            return True
                        k += 1
                    j = k + 1
                else:
                    j += 1
        i += 1
    return False


def difficulty(gt):
    t = raw_tokens(gt)
    nt = len(norm_tokens(gt))
    if r"\begin" in t or r"\\" in t or _nested(gt) or nt > 100:
        return "complex"
    if any(x in MEDIUM for x in t):
        return "medium"
    return "simple"


def edge_flags(it):
    gt = it["gt"]
    t = set(raw_tokens(gt))
    try:
        w, h = Image.open(os.path.join(ROOT, it["image"])).size
    except Exception:
        w, h = 1, 1
    f = []
    if w / h >= 10:
        f.append("wide (w/h>=10)")
    if h / w >= 0.5:
        f.append("tall (h/w>=0.5)")
    if it.get("lowres"):
        f.append("low-res 1.5x")
    if t & {r"\text", r"\textrm", r"\mbox", r"\textit", r"\textbf"}:
        f.append("\\text inside")
    if t & {r"\mathcal", r"\mathfrak", r"\mathscr"}:
        f.append("calligraphic/fraktur")
    if sum(1 for x in raw_tokens(gt) if x in GREEK) >= 3:
        f.append("greek-heavy (>=3)")
    if it.get("kind") == "inl":
        f.append("inline tight crop")
    return f


def bleu4(refs, hyps):
    pn = [0] * 4
    pd = [0] * 4
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


def load_items():
    items = []
    for m in MANIFESTS:
        p = os.path.join(ROOT, "manifests", m)
        if os.path.exists(p):
            items += json.load(open(p, encoding="utf-8"))
    return items


def main():
    MODELS[:] = sys.argv[1].split(",")
    items = load_items()
    for it in items:
        it["diff"] = difficulty(it["gt"])
        it["edges"] = edge_flags(it)
        it["gt_n"] = norm_tokens(it["gt"])
    preds = {}
    for m in MODELS:
        preds[m] = {}
        import glob
        for p in sorted(glob.glob(os.path.join(ROOT, "results", "acc", m + ".*jsonl"))):
            if True:
                for l in open(p, encoding="utf-8"):
                    try:
                        r = json.loads(l)
                    except ValueError:
                        continue
                    preds[m].setdefault(r["id"], r)
    common = [it for it in items if all(it["id"] in preds[m] for m in MODELS)]
    res = {"n_items": len(items), "n_common": len(common), "groups": {}, "items": []}

    def groups(it):
        g = ["ALL", "set:" + it["set"], "diff:" + it["diff"]]
        if it["set"].startswith("unimer"):
            g.append("diff:" + it["diff"] + "|unimer")
        if it["set"] == "arxiv":
            g.append("diff:" + it["diff"] + "|arxiv")
        g += ["edge:" + e for e in it["edges"]]
        return g

    acc = defaultdict(lambda: defaultdict(lambda: {"n": 0, "em": 0, "ed": 0.0, "refs": [], "hyps": [], "runaway": 0, "empty": 0}))
    for it in common:
        row = {"id": it["id"], "set": it["set"], "diff": it["diff"], "edges": it["edges"], "gt": it["gt"], "pred": {}}
        for m in MODELS:
            r = preds[m][it["id"]]
            pn = norm_tokens(r["pred"])
            a, b = "".join(it["gt_n"]), "".join(pn)
            em = a == b
            ed = 1 - Levenshtein.distance(a, b) / max(len(a), len(b), 1)
            empty = len(r["pred"].strip()) == 0
            row["pred"][m] = {"out": r["pred"], "em": em, "ed": round(ed, 4), "runaway": r["runaway"], "s": r["s"]}
            for g in groups(it):
                A = acc[g][m]
                A["n"] += 1
                A["em"] += em
                A["ed"] += ed
                A["refs"].append(it["gt_n"])
                A["hyps"].append(pn)
                A["runaway"] += bool(r["runaway"])
                A["empty"] += empty
        res["items"].append(row)
    for g, d in acc.items():
        res["groups"][g] = {m: {"n": A["n"], "exact%": round(100 * A["em"] / A["n"], 1),
                                "edit": round(A["ed"] / A["n"], 4), "bleu4": round(bleu4(A["refs"], A["hyps"]), 4),
                                "runaway": A["runaway"], "empty": A["empty"]} for m, A in d.items()}
    json.dump(res, open(os.path.join(ROOT, "results", "eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({k: v for k, v in res.items() if k != "items"}, open(os.path.join(ROOT, "results", "metrics.json"), "w", encoding="utf-8"), indent=1)
    order = sorted(res["groups"], key=lambda g: (g != "ALL", not g.startswith("set:"), not g.startswith("diff:"), g))
    print("n_common", len(common), "of", len(items))
    print("group\tmodel\tn\texact%\tedit\tbleu4\trunaway\tempty")
    for g in order:
        for m in MODELS:
            if m in res["groups"][g]:
                x = res["groups"][g][m]
                print(f"{g}\t{m}\t{x['n']}\t{x['exact%']}\t{x['edit']}\t{x['bleu4']}\t{x['runaway']}\t{x['empty']}")


if __name__ == "__main__":
    main()
