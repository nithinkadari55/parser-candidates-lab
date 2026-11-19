import glob, json, os, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, shard, nshards, cdm_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ids = json.load(open(os.path.join(ROOT, "manifests", "cdm_subset.json")))
ids = [i for k, i in enumerate(ids) if k % nshards == shard]
gt = {}
for mf in ["own12", "arxiv", "unimer"]:
    for it in json.load(open(os.path.join(ROOT, "manifests", mf + ".json"), encoding="utf-8")):
        gt[it["id"]] = it["gt"]
pred = {}
for p in glob.glob(os.path.join(ROOT, "results", "acc", model + ".s*.jsonl")):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        pred[r["id"]] = r["pred"]
key = lambda i: i.replace("/", "__").replace(".", "_")
rows = [{"img_id": key(i), "gt": gt[i], "pred": pred.get(i, "")} for i in ids if i in pred]
name = f"{model}_s{shard}"
inp = os.path.join(cdm_dir, name + ".json")
json.dump(rows, open(inp, "w", encoding="utf-8"), ensure_ascii=False)
shutil.rmtree(os.path.join(cdm_dir, "output", name), ignore_errors=True)
subprocess.run([sys.executable, "evaluation.py", "-i", inp, "-o", "output", "-p", "8"], cwd=cdm_dir, check=False)
res = os.path.join(cdm_dir, "output", name, "metrics_res.json")
os.makedirs(os.path.join(ROOT, "results", "cdm"), exist_ok=True)
out = {"model": model, "shard": shard, "n_input": len(rows), "ok": os.path.exists(res)}
if out["ok"]:
    m = json.load(open(res))
    back = {key(i): i for i in ids}
    out["details"] = {back.get(k.split(".")[0], k): v for k, v in m.get("details", {}).items()}
    out["mean_score"] = m.get("mean_score")
json.dump(out, open(os.path.join(ROOT, "results", "cdm", name + ".json"), "w"))
print(model, shard, len(rows), out.get("mean_score"), len(out.get("details", {})))
