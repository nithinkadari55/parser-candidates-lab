import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, shard, nshards, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sets = sys.argv[5].split(",")
budget = float(os.environ.get("BUDGET_S", "6600"))
items = []
for mf in sets:
    items += json.load(open(os.path.join(ROOT, "manifests", mf + ".json"), encoding="utf-8"))
mine = [it for k, it in enumerate(items) if k % nshards == shard]
os.makedirs(os.path.join(ROOT, "results", "acc"), exist_ok=True)
names = [model] + (["p2t15_int8_split"] if model == "p2t15_int8" else [])
runners = {n: load(n, threads) for n in names}
files = {n: open(os.path.join(ROOT, "results", "acc", f"{n}.s{shard}.jsonl"), "w", encoding="utf-8", newline="\n") for n in names}
t0 = time.time()
done = 0
for i, it in enumerate(mine):
    if time.time() - t0 > budget:
        print("BUDGET STOP", done, "of", len(mine), flush=True)
        break
    for n in names:
        t = time.perf_counter()
        try:
            pred, ntok, runaway = runners[n](os.path.join(ROOT, it["image"]))
            err = None
        except Exception as e:
            pred, ntok, runaway, err = "", 0, False, f"{type(e).__name__}: {str(e)[:200]}"
            if i < 3:
                traceback.print_exc()
        rec = {"id": it["id"], "pred": pred, "ntok": ntok, "runaway": runaway, "s": round(time.perf_counter() - t, 3)}
        if err:
            rec["error"] = err
        files[n].write(json.dumps(rec, ensure_ascii=False) + "\n")
        files[n].flush()
    done += 1
    if i % 100 == 0:
        print(i, len(mine), round(time.time() - t0), flush=True)
print("done", model, shard, done, "of", len(mine), round(time.time() - t0))
