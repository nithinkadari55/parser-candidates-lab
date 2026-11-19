import json, os, sys, time, statistics
import psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = sys.argv[1]
items = json.load(open(os.path.join(ROOT, "manifests", "timing.json"), encoding="utf-8"))
proc = psutil.Process()
rss0 = proc.memory_info().rss
t = time.perf_counter()
from mfr import MFR, MFRFast
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
FAST = len(sys.argv) > 3 and sys.argv[3] == "fast"
m = (MFRFast if FAST else MFR)(model, THREADS)
load = time.perf_counter() - t
rss_load = proc.memory_info().rss
m(os.path.join(ROOT, items[0]["image"]))
ts, toks = [], []
for it in items:
    t = time.perf_counter()
    _, n, _ = m(os.path.join(ROOT, it["image"]))
    ts.append((time.perf_counter() - t) * 1000)
    toks.append(n)
mi = proc.memory_info()
try:
    import resource
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except ImportError:
    peak_mb = getattr(mi, "peak_wset", mi.rss) / 2**20
ts_sorted = sorted(ts)
p95 = ts_sorted[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))]
out = {"model": model + ("_split" if FAST else ""), "threads": THREADS, "n": len(ts), "load_s": round(load, 3), "rss_before_mb": round(rss0 / 2**20), "rss_after_load_mb": round(rss_load / 2**20),
       "peak_rss_mb": round(peak_mb), "rss_end_mb": round(mi.rss / 2**20), "median_ms": round(statistics.median(ts)), "p95_ms": round(p95),
       "mean_ms": round(statistics.mean(ts)), "mean_tokens": round(statistics.mean(toks), 1), "ms_per_token_median": round(statistics.median(a / max(b, 1) for a, b in zip(ts, toks)), 1)}
print(json.dumps(out), flush=True)
os.makedirs(os.path.join(ROOT, "results", "timing"), exist_ok=True)
json.dump({"summary": out, "ms": [round(x, 1) for x in ts], "tokens": toks}, open(os.path.join(ROOT, "results", "timing", out["model"] + ".json"), "w"))
