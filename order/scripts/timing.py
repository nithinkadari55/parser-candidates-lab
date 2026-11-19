import json, os, platform, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ro, odb
import numpy as np
from PIL import Image

out = sys.argv[1]
ps = [p for p in odb.pages() if len(p["boxes"]) >= 2]
res = dict(cpu=platform.processor() or platform.machine(), cores=os.cpu_count(), pages=len(ps))


def stats(ts):
    ts = sorted(ts)
    return dict(mean=statistics.mean(ts), median=statistics.median(ts), p95=ts[int(0.95 * (len(ts) - 1))], max=ts[-1], n=len(ts))


for name, fn in [("xycut_xfirst_ms", lambda p: ro.xycut(p["boxes"], p["W"], p["H"], "x")),
                 ("xycutpp_ms", lambda p: ro.xycutpp(p["boxes"], p["W"], p["H"])),
                 ("xycutpp_lab_ms", lambda p: ro.xycutpp_lab(p["boxes"], p["labels"], p["W"], p["H"])),
                 ("docling_rb_python_ms", lambda p: ro.docling_rb(p["boxes"], p["labels"], p["W"], p["H"]))]:
    fn(ps[0])
    ts = []
    for p in ps:
        t = time.perf_counter()
        fn(p)
        ts.append((time.perf_counter() - t) * 1000)
    res[name] = stats(ts)
    print(name, res[name], flush=True)
res["regions_per_page"] = stats([len(p["boxes"]) for p in ps])

import run_pp
sample = ps[::37][:20]
imgs = [np.asarray(Image.open(os.path.join(odb.D, "images", p["img"])).convert("RGB")) for p in sample]
run_pp.items(imgs[0])
ts = []
for a in imgs:
    t = time.perf_counter()
    run_pp.items(a)
    ts.append(time.perf_counter() - t)
res["ppdlv3_infer_s"] = stats(ts)
print("pp", res["ppdlv3_infer_s"], flush=True)

import run_heron
conv = run_heron.setup()
ts = []
conv.convert(os.path.join(odb.D, "images", sample[0]["img"]))
for p in sample[:10]:
    t = time.perf_counter()
    conv.convert(os.path.join(odb.D, "images", p["img"]))
    ts.append(time.perf_counter() - t)
res["docling_rs_image_page_with_ocr_s"] = stats(ts)
print("docling", res["docling_rs_image_page_with_ocr_s"], flush=True)
json.dump(res, open(out, "w"), indent=1)
