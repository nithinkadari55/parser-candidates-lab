import json, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets, run_heron


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


n, out = int(sys.argv[1]), sys.argv[2]
r0 = rss()
t = time.perf_counter()
conv = run_heron.setup()
ps = datasets.pages("odb")[::25][:n]
conv.convert(ps[0]["img"])
first = time.perf_counter() - t
ts = []
for p in ps:
    t = time.perf_counter()
    conv.convert(p["img"])
    ts.append(time.perf_counter() - t)
ts.sort()
res = dict(model="dlrs (docling-rs 2.84.0 image page, heron fp32 + OCR + order)", file_mb=0, load_s=first, rss_before_mb=r0, peak_rss_mb=rss(),
           median_s=ts[len(ts) // 2], mean_s=sum(ts) / len(ts), p95_s=ts[int(0.95 * (len(ts) - 1))], n=len(ts))
json.dump(res, open(out, "w"), indent=1)
print(res)
