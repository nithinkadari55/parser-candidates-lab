import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = json.load(open(os.path.join(ROOT, "plan.json")))
kind = sys.argv[1]
inc = []
if kind == "acc":
    for m, n in P["shards"].items():
        sets = P["sets_with_hwe"] if m in P["hwe_models"] else P["sets_default"]
        inc += [{"model": m, "shard": k, "n": n, "sets": sets} for k in range(n)]
elif kind == "timing":
    inc = [{"model": m} for m in list(P["shards"]) + ["p2t15_int8_split", "monkey_s_onnx_fp32"]]
elif kind == "cdm":
    inc = [{"model": m, "shard": k, "n": P["cdm_shards"]} for m in P["shards"] for k in range(P["cdm_shards"])]
print(json.dumps({"include": inc}))
