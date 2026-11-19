import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ro, odb

GROUPS = {
    "docling": {"docling_rb": lambda p: ro.docling_rb(p["boxes"], p["labels"], p["W"], p["H"])},
    "logic": {
        "xycut_yfirst": lambda p: ro.xycut(p["boxes"], p["W"], p["H"], "y"),
        "xycut_xfirst": lambda p: ro.xycut(p["boxes"], p["W"], p["H"], "x"),
        "xycutpp": lambda p: ro.xycutpp(p["boxes"], p["W"], p["H"]),
        "xycutpp_lab": lambda p: ro.xycutpp_lab(p["boxes"], p["labels"], p["W"], p["H"]),
        "xycutpp_row": lambda p: ro.xycutpp(p["boxes"], p["W"], p["H"], row_rule=True),
        "xycutpp_nostack": lambda p: ro.xycutpp(p["boxes"], p["W"], p["H"], stack_rule=False),
    },
}
group, k, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pages = odb.shard([p for p in odb.pages() if len(p["boxes"]) >= 2], k, n)
with open(out, "w", encoding="utf-8") as f:
    for p in pages:
        gt = list(range(len(p["boxes"])))
        rec = dict(img=p["img"], split=p["split"], layout=p["layout"], source=p["source"], bucket=p["bucket"], tags=p["tags"], n=len(gt))
        for name, fn in GROUPS[group].items():
            t = time.perf_counter()
            pred = fn(p)
            dt = time.perf_counter() - t
            assert sorted(pred) == gt, (name, p["img"])
            s = ro.score(pred, gt)
            s["ms"] = dt * 1000
            s["pred"] = pred
            rec[name] = s
        f.write(json.dumps(rec) + "\n")
print(group, k, n, len(pages))
