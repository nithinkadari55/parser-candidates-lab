import glob, json, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
M = json.load(open(os.path.join(R, "metrics.json"), encoding="utf-8"))
G = M["groups"]
models = list(M["coverage"])
out = [f"# Formula benchmark\n\ncases: {M['n_items']}; scored by every model: {M['n_common']}\n\ncoverage: " +
       ", ".join(f"{m} {c}" for m, c in M["coverage"].items()) + "\n"]


def size_mb(m):
    md, rd = os.path.join(ROOT, "models"), os.path.join(ROOT, "models_ref")
    dirs = {"p2t15_fp32": "breezedeus__pix2text-mfr-1.5", "p2t15_int8": "own_int8_mfr15", "p2t15_fp16": "p2t15_fp16",
            "p2t15_int16": "p2t15_int16", "p2t15_int4_b32": "p2t15_int4_b32", "p2t15_int4_b128": "p2t15_int4_b128",
            "p2t15_int4_b128_int8rest": "p2t15_int4_b128_int8rest"}
    files = []
    if m in dirs:
        files = [os.path.join(md, dirs[m], f) for f in ("encoder_model.onnx", "decoder_model.onnx")]
    elif m == "p2t15_int8_split":
        files = [os.path.join(md, "own_int8_mfr15", "encoder_model.onnx")] + glob.glob(os.path.join(md, "split_p2t15_int8_own", "*.onnx"))
    elif m == "pp_formulanet_plus_s":
        files = glob.glob(os.path.join(rd, "PP_FormulaNet_plus_S", "*.onnx"))
    elif m == "unirec_0_1b":
        files = glob.glob(os.path.join(rd, "unirec", "*.onnx"))
    elif m.startswith("monkey"):
        info = os.path.join(ROOT, "models_monkey", "monkey_info.json")
        if os.path.exists(info):
            i = json.load(open(info))
            k = {"monkey_s_torch": "model_only_fp32_bytes", "monkey_s_onnx_int8": "onnx_int8_bytes", "monkey_s_onnx_fp32": "onnx_fp32_bytes"}[m]
            return round(i.get(k, 0) / 1e6, 1)
    elif m.startswith("doctr"):
        files = glob.glob(os.path.join(rd, "doctr_formulanet_int8" if m.endswith("int8") else "doctr_formulanet", "*.onnx"))
    files = [f for f in files if os.path.exists(f)]
    return round(sum(os.path.getsize(f) for f in files) / 1e6, 1) if files else None


def table(title, keys, cols=("exact%", "edit", "bleu4")):
    keys = [k for k in keys if k in G]
    if not keys:
        return
    out.append(f"\n## {title}\n")
    for col in cols:
        out.append(f"\n**{col}**\n\n| group | n | " + " | ".join(models) + " |\n|---|---|" + "---|" * len(models))
        for g in keys:
            n = next(iter(G[g].values()))["n"]
            out.append(f"| {g} | {n} | " + " | ".join(str(G[g].get(m, {}).get(col, "")) for m in models) + " |")
    out.append(f"\n**runaway / empty / error**\n\n| group | " + " | ".join(models) + " |\n|---|" + "---|" * len(models))
    for g in keys:
        out.append(f"| {g} | " + " | ".join(f"{G[g][m]['runaway']}/{G[g][m]['empty']}/{G[g][m]['error']}" if m in G[g] else "" for m in models) + " |")


table("Overall and per set", ["ALL"] + sorted(k for k in G if k.startswith("set:")))
table("Per difficulty bucket", ["diff:simple", "diff:medium", "diff:complex"] +
      [f"diff:{d}|{s}" for s in ("arxiv", "unimer") for d in ("simple", "medium", "complex")])
table("Per tag (tags overlap)", sorted(k for k in G if k.startswith("tag:")))

if "split_identity" in M:
    s = M["split_identity"]
    out.append(f"\n## Split decoder vs plain INT8\n\nraw outputs identical: {s['identical']} of {s['compared']}; differing ids (first 20): {s['differs']}\n")
if M.get("raw_token_unimer"):
    out.append("\n## docTR FormulaNet, docTR-style raw token metrics on UniMER-Test\n\n| model / subsets | n | token exact % | token edit sim | BLEU-4 (raw tokens) | our normalised exact % |\n|---|---|---|---|---|---|")
    for k, v in M["raw_token_unimer"].items():
        out.append(f"| {k} | {v['n']} | {v['token_exact%']} | {v['token_edit_sim']} | {v['bleu4_raw_tokens']} | {v['normalised_exact%']} |")

tim = {}
for p in glob.glob(os.path.join(R, "timing", "*.json")):
    s = json.load(open(p))["summary"]
    cpu = p[:-5] + ".cpu.txt"
    s["cpu"] = open(cpu).read().split(":")[-1].strip() if os.path.exists(cpu) else ""
    tim[s["model"]] = s
out.append("\n## Size, speed and RAM (one clean job per model, same 330 formulas, onnxruntime 4 threads)\n\n| model | size MB | load s | p50 ms | p95 ms | mean ms | mean tokens | peak RSS MB | runaways | CPU |\n|---|---|---|---|---|---|---|---|---|---|")
for m in models:
    s = tim.get(m, {})
    out.append(f"| {m} | {size_mb(m)} | {s.get('load_s', '')} | {s.get('p50_ms', '')} | {s.get('p95_ms', '')} | {s.get('mean_ms', '')} | {s.get('mean_tokens', '')} | {s.get('peak_rss_mb', '')} | {s.get('runaways', '')} | {s.get('cpu', '')} |")

cdm = {}
for p in glob.glob(os.path.join(R, "cdm", "*.json")):
    d = json.load(open(p))
    cdm.setdefault(d["model"], {"n_input": 0, "details": {}})
    cdm[d["model"]]["n_input"] += d.get("n_input", 0)
    cdm[d["model"]]["details"].update(d.get("details", {}))
if cdm:
    common = set.intersection(*[set(v["details"]) for v in cdm.values()]) if cdm else set()
    out.append(f"\n## CDM (UniMERNet cdm @5a2c80d, pdflatex + ImageMagick), fixed subset\n\nscored ids common to all models (GT rendered): {len(common)}\n\n| model | inputs | scored | mean CDM F1 (common ids) | CDM exact rate (F1=1) |\n|---|---|---|---|---|")
    for m in models:
        if m not in cdm:
            continue
        d = cdm[m]["details"]
        f1 = [d[i]["F1_score"] for i in common]
        out.append(f"| {m} | {cdm[m]['n_input']} | {len(d)} | {round(sum(f1) / max(len(f1), 1), 4)} | {round(sum(x == 1 for x in f1) / max(len(f1), 1), 4)} |")

ev = json.load(open(os.path.join(R, "eval.json"), encoding="utf-8"))
items = ev["items"]
rnd = random.Random(5)
cut = lambda s, n=150: (" ".join(s.split()))[:n] + ("..." if len(" ".join(s.split())) > n else "")
out.append("\n## Paired comparison with p2t15_fp32 (exact match)\n\n| model | only model right | only fp32 right | raw output identical |\n|---|---|---|---|")
for m in models:
    if m == "p2t15_fp32":
        continue
    a = sum(1 for it in items if it["pred"][m]["em"] and not it["pred"]["p2t15_fp32"]["em"])
    b = sum(1 for it in items if not it["pred"][m]["em"] and it["pred"]["p2t15_fp32"]["em"])
    c = sum(1 for it in items if it["pred"][m]["out"] == it["pred"]["p2t15_fp32"]["out"])
    out.append(f"| {m} | {a} | {b} | {c} |")
out.append("\n## Failure examples\n")
for m in models:
    fails = [it for it in items if not it["pred"][m]["em"]]
    out.append(f"\n**{m}** ({len(fails)} not exact)\n")
    by = {}
    for it in fails:
        by.setdefault(it["diff"] + ("/arxiv" if it["set"] == "arxiv" else "/unimer"), []).append(it)
    picks = []
    for k in sorted(by):
        picks += rnd.sample(by[k], min(1, len(by[k])))
    run = [it for it in fails if it["pred"][m]["runaway"]]
    if run:
        picks.append(run[0])
    for it in picks[:7]:
        p = it["pred"][m]
        out.append(f"- `{it['id']}` [{it['diff']}] edit={p['ed']}{' RUNAWAY' if p['runaway'] else ''}<br>GT: `{cut(it['gt'])}`<br>OUT: `{cut(p['out'])}`")
open(os.path.join(R, "report.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
