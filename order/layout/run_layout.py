"""Run one layout model over a shard of DocLayNet test + OmniDocBench English + DrDocBench dev pages.

usage: run_layout.py MODEL SHARD NSHARDS OUT.jsonl [--gt GT.jsonl]
       run_layout.py --bench MODEL N OUT.json
"""
import json, os, resource, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import models, dln
from PIL import Image


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def det_line(ds, pid, w, h, dets, sec, extra=None):
    d = dict(ds=ds, id=pid, W=w, H=h, sec=sec, dets=[[t[0], round(t[1], 4)] + [round(v, 1) for v in t[2:6]] + list(t[6:]) for t in dets])
    if extra:
        d.update(extra)
    return json.dumps(d)


def main():
    if sys.argv[1] == "--bench":
        return bench(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    name, k, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    gt_out = sys.argv[sys.argv.index("--gt") + 1] if "--gt" in sys.argv else None
    sets = os.environ.get("LAYOUT_SETS", "dln,odb,drdoc").split(",")
    if not os.path.exists(models.PATHS[name]):
        print("model file missing", models.PATHS[name])
        open(out, "w").close()
        return
    m = models.load(name)
    gf = open(gt_out, "w", encoding="utf-8") if gt_out else None
    cnt = 0
    with open(out, "w", encoding="utf-8") as f:
        if "dln" in sets:
            half = n // 2
            fi = k % 2
            for ri, pid, img, gt, cat, w, h in dln.rows(fi, lambda r: r % half == k // 2):
                t = time.perf_counter()
                dets = m(img)
                f.write(det_line("dln", pid, w, h, dets, time.perf_counter() - t, {"doc_category": cat}) + "\n")
                if gf:
                    gf.write(json.dumps(dict(ds="dln", id=pid, W=w, H=h, doc_category=cat, gt=[[g[0]] + [round(v, 1) for v in g[1:]] for g in gt])) + "\n")
                cnt += 1
        import datasets
        for ds in ("odb", "drdoc"):
            if ds not in sets:
                continue
            for p in datasets.shard(datasets.pages(ds), k, n):
                img = Image.open(p["img"]).convert("RGB")
                iw, ih = img.size
                t = time.perf_counter()
                dets = m(img)
                sec = time.perf_counter() - t
                sx, sy = p["W"] / iw, p["H"] / ih
                dets = [(d[0], d[1], d[2] * sx, d[3] * sy, d[4] * sx, d[5] * sy) + tuple(d[6:]) for d in dets]
                f.write(det_line(ds, p["id"], p["W"], p["H"], dets, sec) + "\n")
                cnt += 1
    if gf:
        gf.close()
    print(name, k, n, cnt, "pages, peak RSS MB", round(rss_mb()))


def bench(name, n, out):
    import datasets
    r0 = rss_mb()
    t = time.perf_counter()
    m = models.load(name)
    load = time.perf_counter() - t
    r1 = rss_mb()
    ps = datasets.pages("odb")[::25][:n]
    imgs = [Image.open(p["img"]).convert("RGB") for p in ps]
    m(imgs[0])
    ts = []
    for im in imgs:
        t = time.perf_counter()
        m(im)
        ts.append(time.perf_counter() - t)
    ts.sort()
    res = dict(model=name, file_mb=os.path.getsize(models.PATHS[name]) / 1e6, load_s=load, rss_before_mb=r0, rss_after_load_mb=r1,
               peak_rss_mb=rss_mb(), median_s=ts[len(ts) // 2], mean_s=sum(ts) / len(ts), p95_s=ts[int(0.95 * (len(ts) - 1))], n=len(ts),
               threads=os.environ.get("ORT_THREADS", "0"))
    json.dump(res, open(out, "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
