"""Pages for test 6: ground truth (OmniDocBench English + DrDocBench dev) joined with the rects/decorations
of run 44483268299 (ro-*-logic-*.jsonl) and the heron int8 detections of run 44477997398."""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "layout"))
import datasets  # noqa: E402

ART = os.environ.get("T6_ART", "art")


def jl(pattern):
    for f in sorted(glob.glob(os.path.join(ART, "**", pattern), recursive=True)):
        for l in open(f, encoding="utf-8"):
            try:
                yield json.loads(l)
            except ValueError:
                pass


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def dedupe(dets):
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(inter(d["box"], k["box"]) < 0.8 * min(area(d["box"]), area(k["box"])) for k in keep):
            keep.append(d)
    return keep


def pages():
    import models
    ro = {}
    for r in jl("ro-*-logic-*.jsonl"):
        ro[r["id"]] = dict(rects=r.get("rects", []), deco=r.get("deco", []))
    h8 = {}
    for r in jl("layout-heron_int8-*.jsonl"):
        if r.get("ds") in ("odb", "drdoc"):
            h8[r["id"]] = r
    out = []
    for ds in ("odb", "drdoc"):
        for p in datasets.pages(ds):
            x = ro.get(p["id"], {})
            p["rects"] = x.get("rects", [])
            p["deco"] = x.get("deco", [])
            p["have_ro"] = p["id"] in ro
            tags = list(p["tags"])
            if any(sum(1 for b in p["boxes"] if _fin(b, r) >= 0.8) >= 2 and (r[2] - r[0]) < 0.8 * p["W"] for r in p["rects"]):
                tags.append("boxed_or_shaded_region")
            if any(c in ("figure_caption", "table_caption") for c in p["cats"]):
                tags.append("captions")
            p["tags"] = tags
            r = h8.get(p["id"])
            dets = []
            if r:
                for d in r["dets"]:
                    if d[1] < 0.5:
                        continue
                    u = models.unify("heron_int8", d[0], d[3], d[5], p["H"])
                    if not u:
                        continue
                    dets.append(dict(label=models.TO_RO[u], score=d[1], box=d[2:6], furniture=u in ("Page-header", "Page-footer")))
                dets = dedupe(dets)
            p["h8"] = dets
            p["have_h8"] = r is not None
            out.append(p)
    return out


def _fin(b, r):
    a = area(b)
    return inter(b, r) / a if a > 0 else 0.0


def match(gtb, regs):
    """GT body region -> detected region covering it (as in aggregate_ro.py)."""
    m = {}
    for g, b in enumerate(gtb):
        ga = max(1e-6, area(b))
        best, bi = 0, None
        for k, r in enumerate(regs):
            i = inter(b, r)
            if (i / ga >= 0.5 or i / max(1e-6, area(r)) >= 0.8) and i > best:
                best, bi = i, k
        if bi is not None:
            m[g] = bi
    return m


def det_ranks(gtb, regs):
    """Detected region -> GT rank (smallest GT index among GT regions matched to it), for training."""
    m = match(gtb, regs)
    rank = {}
    for g, k in m.items():
        rank[k] = min(rank.get(k, g), g)
    return rank
