import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ro, addons, datasets, pdx_order

ADD = {
    "xycutpp+fig": {"fig"}, "xycutpp+box": {"box"}, "xycutpp+num": {"num"}, "xycutpp+tiny": {"tiny"},
    "xycutpp+num+tiny": {"num", "tiny"}, "xycutpp+all": {"fig", "box", "num", "tiny"},
}
FULL_ADD = {"full:xycutpp+deco": {"deco"}, "full:xycutpp+all": {"fig", "box", "num", "tiny", "deco"},
            "full:xycutpp+all-deco": {"fig", "box", "num", "tiny"}}


def body_of(pred_full, fbody):
    return [fbody[i] for i in pred_full if fbody[i] >= 0]


def main():
    ds, group, k, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    allp = datasets.pages(ds)
    deco = addons.decorations(allp) if group == "logic" else {}
    ps = datasets.shard(allp, k, n)
    with open(out, "w", encoding="utf-8") as f:
        for p in ps:
            W, H = p["W"], p["H"]
            gt = list(range(len(p["boxes"])))
            rec = {key: p[key] for key in ("id", "ds", "split", "layout", "source", "bucket", "tags", "doc")}
            rec["n"] = len(gt)
            nbody = sum(1 for x in p["fbody"] if x >= 0)
            if nbody != len(gt):
                rec["full_mismatch"] = [nbody, len(gt)]

            def run(name, fn, full=False):
                t = time.perf_counter()
                try:
                    pred = fn()
                except Exception as e:
                    rec[name] = dict(error=(type(e).__name__ + ": " + str(e))[:200])
                    return
                dt = time.perf_counter() - t
                if full:
                    pred = body_of(pred, p["fbody"])
                if sorted(pred) != gt:
                    rec[name] = dict(error="bad permutation")
                    return
                s = ro.score(pred, gt)
                s["ms"] = dt * 1000
                s["pred"] = pred
                rec[name] = s

            if group == "docling":
                run("docling_rb", lambda: ro.docling_rb(p["boxes"], p["labels"], W, H))
                if nbody == len(gt):
                    run("full:docling_rb", lambda: ro.docling_rb(p["fboxes"], p["flabels"], W, H), full=True)
            else:
                t = time.perf_counter()
                rects = addons.detect_rects(p["img"], W, H)
                rec["rects"] = [[round(v, 1) for v in r] for r in rects]
                rec["rect_ms"] = (time.perf_counter() - t) * 1000
                dset = deco.get(p["id"], set())
                rec["deco"] = sorted(dset)
                rec["deco_furniture"] = sum(1 for i in dset if p["fbody"][i] < 0)
                run("xycut_yfirst", lambda: ro.xycut(p["boxes"], W, H, "y"))
                run("xycut_xfirst", lambda: ro.xycut(p["boxes"], W, H, "x"))
                run("xycutpp", lambda: ro.xycutpp(p["boxes"], W, H))
                for name, a in ADD.items():
                    run(name, lambda a=a: addons.order(p["boxes"], p["labels"], p["texts"], W, H, rects, a))
                pdx = {}

                def pdx_run(bx, cats, tx):
                    o, miss = pdx_order.order(bx, [pdx_order.FROM_ODB.get(c, "text") for c in cats], tx, W, H)
                    pdx["missing"] = pdx.get("missing", 0) + miss
                    return o
                run("pdx_xycut_enhanced", lambda: pdx_run(p["boxes"], p["cats"], p["texts"]))
                if nbody == len(gt):
                    run("full:xycutpp", lambda: ro.xycutpp(p["fboxes"], W, H), full=True)
                    for name, a in FULL_ADD.items():
                        run(name, lambda a=a: addons.order(p["fboxes"], p["flabels"], p["ftexts"], W, H, rects, a, dset), full=True)
                    run("full:pdx_xycut_enhanced", lambda: pdx_run(p["fboxes"], p["fcats"], p["ftexts"]), full=True)
                rec["pdx_missing"] = pdx.get("missing", 0)
            f.write(json.dumps(rec) + "\n")
    print(ds, group, k, n, len(ps))


if __name__ == "__main__":
    main()
