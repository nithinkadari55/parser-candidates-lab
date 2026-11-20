import os, sys, json, glob, random, statistics

METHODS = ["none", "text", "pplcnet", "pplcnet_inv", "onnxtr", "onnxtr_inv", "oracle"]


def paired(x, y):
    from scipy.stats import wilcoxon
    d = [a - b for a, b in zip(x, y)]
    if len(d) < 10: return None
    rng = random.Random(0); bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    try: p = wilcoxon(x, y, zero_method="zsplit").pvalue
    except ValueError: p = 1.0
    return dict(diff=sum(d) / len(d), lo=bs[50], hi=bs[1949], p=p)


def main():
    root = sys.argv[1]
    recs = [json.loads(l) for p in glob.glob(os.path.join(root, "**", "rot-*.jsonl"), recursive=True) for l in open(p, encoding="utf-8")]
    groups = {}
    for r in recs:
        g = r["kind"] if r["kind"] in ("rot", "icdar_real") else f"{r['kind']}/{r['mode']}"
        groups.setdefault(g, []).append(r)
    groups["all born-digital"] = [r for r in recs if r["kind"].endswith("_bd")]
    out = []; P = out.append
    P("## Test 5: rotated tables")
    P(f"Items: {len(recs)}")
    P("\n### Orientation accuracy (% of crops whose predicted rotation equals the true one)")
    P("| group | n | " + " | ".join(m for m in METHODS if m not in ("none", "oracle")) + " | 'none' (already upright) |\n|" + "---|" * 8)
    for g, rs in groups.items():
        rs2 = [r for r in rs if r.get("truth") is not None]
        if not rs2: continue
        P(f"| {g} | {len(rs2)} | " + " | ".join(f"{100 * sum(r['preds'].get(m) == r['truth'] for r in rs2) / len(rs2):.1f}" for m in METHODS if m not in ("none", "oracle"))
          + f" | {100 * sum(r['truth'] == 0 for r in rs2) / len(rs2):.1f} |")
    for model in ("tableformer", "tatr16"):
        P(f"\n### TEDS-Struct after correcting with each method: {model}")
        P("| group | n | " + " | ".join(METHODS) + " |\n|" + "---|" * (len(METHODS) + 2))
        for g, rs in groups.items():
            vals = {}
            for m in METHODS:
                v = [r["scores"].get(f"{model}@{r['preds'][m]}") for r in rs if m in r["preds"]]
                v = [x for x in v if x is not None]
                vals[m] = f"{sum(v) / len(v):.3f}" if v else "-"
            P(f"| {g} | {len(rs)} | " + " | ".join(vals[m] for m in METHODS) + " |")
        P(f"\nPaired tests vs no correction ({model}):")
        for g, rs in groups.items():
            for m in ("text", "pplcnet", "pplcnet_inv", "onnxtr", "onnxtr_inv"):
                xs, ys = [], []
                for r in rs:
                    a = r["scores"].get(f"{model}@{r['preds'][m]}"); b = r["scores"].get(f"{model}@0")
                    if a is not None and b is not None: xs.append(a); ys.append(b)
                t = paired(xs, ys)
                if t: P(f"- {g} {m} vs none: n={len(xs)} diff={t['diff']:+.4f} CI [{t['lo']:+.4f}, {t['hi']:+.4f}] p={t['p']:.2g}")
    P("\n### Time per crop (ms, median)")
    for m in ("pplcnet", "onnxtr"):
        v = sorted(r["ms"][m] for r in recs)
        if v: P(f"- {m}: median {statistics.median(v):.1f}, p95 {v[int(0.95 * (len(v) - 1))]:.1f}")
    P("- text-layer vote: no model; reading glyph angles is part of the text extraction")
    P("\n### Examples where the text-layer vote was wrong")
    bad = [r for r in recs if r.get("truth") is not None and r["preds"]["text"] != r["truth"]][:10]
    for r in bad: P(f"- {r['kind']} {r['id']} {r['mode']} truth={r['truth']} text={r['preds']['text']} pplcnet={r['preds']['pplcnet']} onnxtr={r['preds']['onnxtr']} words={r['n_words']}")
    txt = "\n".join(out); print(txt)
    os.makedirs(os.path.join(root, "r3"), exist_ok=True)
    open(os.path.join(root, "r3", "rotate.md"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
