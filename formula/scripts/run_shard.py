import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfr import MFR, ROOT
model, shard, nshards, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
items = []
for mf in ["own12.json", "arxiv.json", "unimer.json"]:
    items += json.load(open(os.path.join(ROOT, "manifests", mf), encoding="utf-8"))
mine = [it for k, it in enumerate(items) if k % nshards == shard]
os.makedirs(os.path.join(ROOT, "results", "acc"), exist_ok=True)
outp = os.path.join(ROOT, "results", "acc", f"{model}.s{shard}.jsonl")
m = MFR(model, threads)
t0 = time.time()
with open(outp, "w", encoding="utf-8", newline="\n") as f:
    for i, it in enumerate(mine):
        t = time.perf_counter()
        pred, ntok, runaway = m(os.path.join(ROOT, it["image"]))
        f.write(json.dumps({"id": it["id"], "pred": pred, "ntok": ntok, "runaway": runaway, "s": round(time.perf_counter() - t, 3)}, ensure_ascii=False) + "\n")
        f.flush()
        if i % 50 == 0:
            print(i, len(mine), round(time.time() - t0), flush=True)
print("done", model, shard, len(mine), round(time.time() - t0))
