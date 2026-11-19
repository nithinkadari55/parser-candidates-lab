import json, os, resource, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, threads = sys.argv[1], int(sys.argv[2])
items = json.load(open(os.path.join(ROOT, "manifests", "timing.json"), encoding="utf-8"))
t = time.perf_counter()
from registry import load
m = load(model, threads)
load_s = time.perf_counter() - t
rss_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
for it in items[:3]:
    m(os.path.join(ROOT, it["image"]))
ts, toks, run, preds = [], [], 0, []
for it in items:
    t = time.perf_counter()
    pr, n, r = m(os.path.join(ROOT, it["image"]))
    preds.append(pr)
    ts.append((time.perf_counter() - t) * 1000)
    toks.append(n)
    run += bool(r)
s = sorted(ts)
q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
out = {"model": model, "threads": threads, "n": len(ts), "load_s": round(load_s, 2), "peak_rss_after_load_mb": round(rss_load),
       "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "p50_ms": round(q(0.5)), "p95_ms": round(q(0.95)),
       "mean_ms": round(statistics.mean(ts)), "mean_tokens": round(statistics.mean(toks), 1), "runaways": run,
       "ms_per_token_p50": round(statistics.median(a / max(b, 1) for a, b in zip(ts, toks)), 2)}
print(json.dumps(out), flush=True)
TD = os.path.join(ROOT, "results", "timing", os.environ.get("TIMING_GROUP", ""))
os.makedirs(TD, exist_ok=True)
json.dump({"summary": out, "ms": [round(x, 1) for x in ts], "tokens": toks, "ids": [it["id"] for it in items], "preds": preds},
          open(os.path.join(TD, model + ".json"), "w"))
