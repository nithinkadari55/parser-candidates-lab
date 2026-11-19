import json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "results", "eval.json"), encoding="utf-8"))
items = r["items"]
models = list(items[0]["pred"].keys())
base = "p2t15_fp32"
for m in models:
    if m == base:
        continue
    a = sum(1 for it in items if it["pred"][m]["em"] and not it["pred"][base]["em"])
    b = sum(1 for it in items if not it["pred"][m]["em"] and it["pred"][base]["em"])
    same = sum(1 for it in items if it["pred"][m]["out"].split() == it["pred"][base]["out"].split())
    print(f"paired vs {base}: {m}: only-{m}-correct={a} only-{base}-correct={b} identical-raw-output={same}/{len(items)}")
rnd = random.Random(5)
def cut(s, n=160):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + " ...[" + str(len(s)) + " chars]"
for m in models:
    fails = [it for it in items if not it["pred"][m]["em"]]
    groups = {}
    for it in fails:
        k = it["set"] + ("/inl" if "inline tight crop" in " ".join(it["edges"]) else "") + ("/lowres" if "low-res 1.5x" in it["edges"] else "")
        groups.setdefault(k, []).append(it)
    print("\n== failures", m, len(fails))
    picks = []
    keys = sorted(groups)
    while len(picks) < 8 and any(groups.values()):
        for k in keys:
            if groups[k] and len(picks) < 8:
                g = groups[k]
                picks.append(g.pop(rnd.randrange(len(g))))
    run = [it for it in fails if it["pred"][m]["runaway"]]
    if run and not any(p["pred"][m]["runaway"] for p in picks):
        picks[-1] = run[0]
    for it in picks:
        p = it["pred"][m]
        print(f"- {it['id']} [{it['diff']}] ed={p['ed']}{' RUNAWAY' if p['runaway'] else ''}\n  GT : {cut(it['gt'])}\n  OUT: {cut(p['out'])}")
