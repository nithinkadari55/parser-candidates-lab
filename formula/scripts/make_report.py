import glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
M = json.load(open(os.path.join(R, "metrics.json"), encoding="utf-8"))
G = M["groups"]
models = ["p2t15_fp32", "p2t10_int8_mathorium", "p2t15_int8_own"]
out = [f"# Formula benchmark\n\ncases scored (all 3 models have a result): {M['n_common']} of {M['n_items']}\n"]


def table(title, keys):
    out.append(f"\n## {title}\n\n| group | model | n | exact % | edit score | BLEU-4 | runaway | empty |\n|---|---|---|---|---|---|---|---|")
    for g in keys:
        if g not in G:
            continue
        for m in models:
            if m in G[g]:
                x = G[g][m]
                out.append(f"| {g} | {m} | {x['n']} | {x['exact%']} | {x['edit']} | {x['bleu4']} | {x['runaway']} | {x['empty']} |")


table("Overall and per set", ["ALL"] + sorted(k for k in G if k.startswith("set:")))
table("Per difficulty bucket (all sets)", ["diff:simple", "diff:medium", "diff:complex"])
table("Per difficulty bucket, arXiv only", ["diff:simple|arxiv", "diff:medium|arxiv", "diff:complex|arxiv"])
table("Per difficulty bucket, UniMER only", ["diff:simple|unimer", "diff:medium|unimer", "diff:complex|unimer"])
table("Edge-case slices (overlapping)", sorted(k for k in G if k.startswith("edge:")))

tim = sorted(glob.glob(os.path.join(R, "timing", "*.json")))
if tim:
    out.append("\n## Speed and RAM (dedicated job)\n\n| model | threads | n | load s | median ms | p95 ms | mean ms | mean tokens | ms/token (median) | peak RSS MB |\n|---|---|---|---|---|---|---|---|---|---|")
    for p in tim:
        s = json.load(open(p))["summary"]
        out.append(f"| {s['model']} | {s['threads']} | {s['n']} | {s['load_s']} | {s['median_ms']} | {s['p95_ms']} | {s['mean_ms']} | {s['mean_tokens']} | {s['ms_per_token_median']} | {s['peak_rss_mb']} |")

mi = os.path.join(ROOT, "models", "models_info.json")
if os.path.exists(mi):
    out.append("\n## Model files\n\n| file | bytes | sha256 |\n|---|---|---|")
    for k, v in sorted(json.load(open(mi)).items()):
        out.append(f"| {k} | {v['bytes']} | {v['sha256'][:16]}... {'(same as laptop build)' if v.get('same_as_local_laptop_build') else ''} |")

extra = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "report_extra.py")], capture_output=True, text=True, encoding="utf-8").stdout
out.append("\n## Paired comparison and failure examples\n\n```\n" + extra + "\n```\n")
open(os.path.join(R, "report.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
