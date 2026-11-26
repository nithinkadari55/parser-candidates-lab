"""Round 9 candidate runner. Every candidate gets the bundle's crop (docling's run-time crop), turned by the
harness's orientation rule (row["turn"]), and the bundle's words (turned with it); its cells are turned back.

  infer MODEL SHARD PART NPARTS BUNDLE OUT [CORE_SHARE]   (model environment; light imports)
        -> OUT/<model>/cells_s<k>p<p>.jsonl.gz (cells, text, timings; private) + OUT/<model>/run_s<k>p<p>.json (speed, memory)
  score MODEL SHARD PART BUNDLE OUT                         (scoring environment)
        -> OUT/<model>/s<k>p<p>.jsonl (ids and numbers only; public)
  ref BUNDLE OUT SHARD                                      TATR on the shard's first 30 crops (machine speed reference)
  synth OUT                                                 a small synthetic bundle (smoke test with known answers)"""
import os, sys, json, gzip, time, zlib, resource, platform, importlib, traceback, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo9 import turn_box, unturn_box, turn_image

REG = {
    "tatr": ("ort", "Tatr", None), "rapid_en": ("ort", "Rapid", "rapid"), "rapid_zh": ("ort", "Rapid", "rapid"), "rapid_plus": ("ort", "Rapid", "rapid"),
    "paddle_slanet": ("ort", "Rapid", "paddle_onnx"), "paddle_slanet_plus": ("ort", "Rapid", "paddle_onnx"), "ae3": ("ort", "Ae3", "ae3"),
    "mineru": ("mineru", "MinerU", "mineru"), "pptable": ("pptable", "PPTable", "pptable"),
    "tablemaster": ("torchm", "TableMaster", "tablemaster"), "mutab_ptn": ("torchm", "MuTab", "mutab_ptn"), "mutab_ftn": ("torchm", "MuTab", "mutab_ftn"),
    "lore_wireless": ("torchm", "Lore", "lore_wireless"), "lore_ptn": ("torchm", "Lore", "lore_ptn"), "lore_wtw": ("torchm", "Lore", "lore_wtw"),
    "trace_base": ("torchm", "Trace", "trace_base"), "trace_wtw": ("torchm", "Trace", "trace_wtw"),
}
W = os.environ.get("R9_W", os.path.join(os.path.expanduser("~"), "r9w"))


def rss_kb():
    try:
        return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS"))
    except Exception:
        return None


def cpu_name():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


def load_rows(bundle, keep):
    out = []
    with gzip.open(os.path.join(bundle, "rows.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if keep(r["id"]): out.append(r)
    return out


def core(tid, share):
    return zlib.crc32(("r9core|" + tid).encode()) % 10000 < int(round(share * 10000))


def infer(model, shard, part, nparts, bundle, out, share=None):
    from PIL import Image
    mod, cls, wgroup = REG[model]
    keep = lambda t: zlib.crc32(("part|" + t).encode()) % nparts == part and (share is None or core(t, share))
    rows = load_rows(bundle, keep)
    lim = int(os.environ.get("R9_LIMIT") or 0)
    if lim: rows = rows[:lim]
    odir = os.path.join(out, model); os.makedirs(odir, exist_ok=True)
    r0 = rss_kb(); t0 = time.perf_counter()
    M = getattr(importlib.import_module("models9." + mod), cls)(model, os.path.join(W, wgroup) if wgroup else None)
    load_s = time.perf_counter() - t0; r1 = rss_kb()
    wbytes = sum(os.path.getsize(p) for p in getattr(M, "weights", []) if os.path.exists(p))
    n_err = 0; times = []
    with gzip.open(os.path.join(odir, f"cells_s{shard}p{part}.jsonl.gz"), "wt", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            rec = dict(id=r["id"], set=r["set"])
            try:
                im = Image.open(os.path.join(bundle, "img", r["id"] + ".png")).convert("RGB"); w, h = im.size; turn = r.get("turn") or 0
                imt = turn_image(im, turn)
                wt = [dict(text=x["text"], bbox=turn_box(x["bbox"], turn, w, h) if turn else list(x["bbox"])) for x in r["words"]]
                t1 = time.perf_counter(); cells, ms_model, info = M(imt, wt); ms_total = (time.perf_counter() - t1) * 1000
                for c in cells:
                    if turn: c["bbox"] = unturn_box(c["bbox"], turn, w, h)
                rec.update(cells=[dict(r0=int(c["r0"]), r1=int(c["r1"]), c0=int(c["c0"]), c1=int(c["c1"]), bbox=[float(v) for v in c["bbox"]],
                                       header=bool(c.get("header")), **({"text": str(c["text"])} if "text" in c else {})) for c in cells],
                           ms_model=float(ms_model), ms_total=ms_total, info=info)
                times.append(ms_total)
                if os.environ.get("R9_VERBOSE"): print(model, r["id"], "cells", len(cells), "ms", round(ms_total), flush=True)
            except Exception as e:
                n_err += 1; rec.update(cells=[], err=repr(e)[:300])
                if n_err <= 5: traceback.print_exc()
            f.write(json.dumps(rec) + "\n")
            if i % 200 == 0: print(model, "progress", i, len(rows), "errors", n_err, flush=True)
    run = dict(model=model, shard=shard, part=part, n=len(rows), errors=n_err, load_s=load_s, rss_before_kb=r0, rss_loaded_kb=r1,
               peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, weights_bytes=wbytes, cpu=cpu_name(), nproc=os.cpu_count(),
               ms_total_median=sorted(times)[len(times) // 2] if times else None, core_share=share)
    json.dump(run, open(os.path.join(odir, f"run_s{shard}p{part}.json"), "w"))
    print("done", json.dumps(run), flush=True)
    if rows and n_err > 0.05 * len(rows): sys.exit(f"too many errors: {n_err} of {len(rows)}")


def _score_job(job):
    import score8 as S8
    tid, g, words, cells, own = job
    # the common words fill the cells by box; a model without usable boxes (all zero) is scored on its own text only
    s = S8.score_cells(g, words, cells) if (not own or not cells or any(sum(c["bbox"]) != 0 for c in cells)) else None
    so = None
    if own:
        # the model's own cell text: one pseudo-word per cell on a synthetic grid box, so each text lands in its cell
        cs = [dict(c, bbox=[c["c0"] * 10.0 + 1, c["r0"] * 10.0 + 1, c["c1"] * 10.0 - 1, c["r1"] * 10.0 - 1]) for c in cells]
        pw = [dict(text=c["text"], bbox=list(c["bbox"])) for c in cs if c.get("text", "").strip()]
        so = S8.score_cells(g, pw, cs)
    return tid, s, so


def score(model, shard, part, bundle, out):
    from multiprocessing import Pool
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r8"))
    import common9  # noqa: F401  (paths for score8)
    rows = {r["id"]: r for r in load_rows(bundle, lambda t: True)}
    recs = []
    with gzip.open(os.path.join(out, model, f"cells_s{shard}p{part}.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f: recs.append(json.loads(line))
    jobs = []
    for rec in recs:
        g = json.load(open(os.path.join(bundle, "gt", rec["id"] + ".json"), encoding="utf-8"))
        own = any("text" in c for c in rec["cells"])
        jobs.append((rec["id"], g, rows[rec["id"]]["words"], rec["cells"], own))
    jobs.sort(key=lambda j: -len(j[3]))
    res = {}
    with Pool(4) as p:
        for tid, s, so in p.imap_unordered(_score_job, jobs, chunksize=1): res[tid] = (s, so)
    with open(os.path.join(out, model, f"s{shard}p{part}.jsonl"), "w") as f:
        for rec in recs:
            s, so = res.get(rec["id"], (None, None))
            f.write(json.dumps(dict(id=rec["id"], set=rec["set"], s=s, so=so, ms_total=rec.get("ms_total"), ms_model=rec.get("ms_model"),
                                    n_cells=len(rec["cells"]), info=rec.get("info"), err=rec.get("err"))) + "\n")
    v = [x[0]["teds_s"] for x in res.values() if x[0] and x[0].get("teds_s") is not None]
    print(model, "scored", len(recs), "mean TEDS-S", round(sum(v) / max(1, len(v)), 4), flush=True)


def ref(bundle, out, shard):
    sys.path.insert(0, os.path.join(HERE, "models9"))
    from PIL import Image
    from models9.ort import Tatr
    rows = load_rows(bundle, lambda t: True)[:30]
    M = Tatr("tatr", None); ts = []
    for r in rows:
        im = Image.open(os.path.join(bundle, "img", r["id"] + ".png")).convert("RGB")
        t = time.perf_counter(); M(im, r["words"]); ts.append((time.perf_counter() - t) * 1000)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(shard=shard, cpu=cpu_name(), tatr_ms=ts), open(os.path.join(out, f"ref_s{shard}.json"), "w"))
    print("tatr reference ms median", sorted(ts)[len(ts) // 2] if ts else None, cpu_name(), flush=True)


def strip(odir, shard, part):
    """cells without text (geometry and grid positions only) for the public artifact"""
    src = os.path.join(odir, f"cells_s{shard}p{part}.jsonl.gz")
    with gzip.open(src, "rt", encoding="utf-8") as f, gzip.open(os.path.join(odir, f"geom_s{shard}p{part}.jsonl.gz"), "wt", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            g.write(json.dumps(dict(id=r["id"], cells=[{k: v for k, v in c.items() if k != "text"} for c in r.get("cells", [])])) + "\n")


def synth(out):
    """four drawn tables with known structure: ruled 4x3, borderless 5x4, ruled with a column span and a row span,
    three-line 6x5; rendered at 144 dpi; words from the drawing positions"""
    import io
    import pypdfium2 as pdfium
    from reportlab.pdfgen import canvas
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r8"))
    import common9  # noqa: F401
    from prep8 import grid_html, tags_of
    os.makedirs(os.path.join(out, "img"), exist_ok=True); os.makedirs(os.path.join(out, "gt"), exist_ok=True)
    specs = [("ruled", 4, 3, [], "grid"), ("borderless", 5, 4, [], "none"), ("spans", 4, 4, [(0, 1, 1, 3), (1, 0, 2, 1)], "grid"), ("threeline", 6, 5, [], "three")]
    rows_out = []
    for name, R, C, spans, lines in specs:
        cw, rh, x0, y0 = 90.0, 22.0, 20.0, 20.0
        PW, PH = x0 * 2 + cw * C, y0 * 2 + rh * R
        buf = io.BytesIO(); cv = canvas.Canvas(buf, pagesize=(PW, PH)); cv.setFont("Helvetica", 10)
        covered = {}; cells = []
        for (r, c, rs, cs) in spans:
            for a in range(r, r + rs):
                for b in range(c, c + cs): covered[(a, b)] = (r, c)
        for r in range(R):
            for c in range(C):
                if (r, c) in covered and covered[(r, c)] != (r, c): continue
                rs, cs = next(((s[2], s[3]) for s in spans if (s[0], s[1]) == (r, c)), (1, 1))
                text = ("Head %d" % c) if r == 0 else ("%d.%d" % (r * 7 + c, c))
                bx = [x0 + c * cw, y0 + r * rh, x0 + (c + cs) * cw, y0 + (r + rs) * rh]
                tw = cv.stringWidth(text, "Helvetica", 10)
                tx = (bx[0] + bx[2]) / 2 - tw / 2; ty = (bx[1] + bx[3]) / 2 + 3.5
                cv.drawString(tx, PH - ty, text)
                cells.append(dict(r=r, c=c, rs=rs, cs=cs, text=text, bbox=bx, word=[tx, ty - 9, tx + tw, ty + 2]))
                if lines == "grid": cv.rect(bx[0], PH - bx[3], bx[2] - bx[0], bx[3] - bx[1])
        if lines == "three":
            for yy in (y0, y0 + rh, y0 + R * rh): cv.line(x0, PH - yy, x0 + C * cw, PH - yy)
        cv.showPage(); cv.save()
        s = 2.0
        im = pdfium.PdfDocument(buf.getvalue())[0].render(scale=s).to_pil().convert("RGB")
        toks, gcells = grid_html([dict(r=x["r"], c=x["c"], rs=x["rs"], cs=x["cs"], text=x["text"], bbox=[v * s for v in x["bbox"]]) for x in cells], 1)
        tid = "synth_" + name
        g = dict(id=tid, doc=tid, html=toks, cells=gcells, tags=tags_of(toks, gcells, None), kind="synth")
        json.dump(g, open(os.path.join(out, "gt", tid + ".json"), "w")); im.save(os.path.join(out, "img", tid + ".png"))
        words = [dict(text=x["text"], bbox=[v * s for v in x["word"]]) for x in cells]
        rows_out.append(dict(id=tid, set="synth", kind="synth", W=im.size[0], H=im.size[1], scale=s, turn=0, words=words, cells={}))
    with gzip.open(os.path.join(out, "rows.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in rows_out: f.write(json.dumps(r) + "\n")
    print("synthetic bundle", len(rows_out), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "infer": infer(a[2], int(a[3]), int(a[4]), int(a[5]), a[6], a[7], float(a[8]) if len(a) > 8 and a[8] else None)
    elif a[1] == "score": score(a[2], int(a[3]), int(a[4]), a[5], a[6])
    elif a[1] == "ref": ref(a[2], a[3], int(a[4]))
    elif a[1] == "synth": synth(a[2])
    elif a[1] == "strip": strip(a[2], int(a[3]), int(a[4]))
