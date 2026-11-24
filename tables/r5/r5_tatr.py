"""Round 5 (harness port): TATR v1.1 fp16 on docling.rs crops and docling.rs word cells.

usage:
  r5_tatr.py jobs SHARD NSHARDS JOBS.jsonl     job list for the r5words tool (same shards as r4)
  r5_tatr.py run SHARD NSHARDS WORDS_DIR       TATR with the docling words -> pred/tatr16r5_<set>_s<k>.jsonl
                                               shard 0 also writes the port fixtures (r5fix/)
The crops and words come from tables/r5/words (docling-pdf 2.84.0 word_cells and page render at 2x),
the same source the harness reads at run time."""
import os, sys, json, time, copy, zlib, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r4"))
from r4_common import *
import numpy as np
from PIL import Image

FIX_PER_SET = 3
RAW_FIX = 2
E2E_FIX = 2
FIX_MAX_WORDS = 100


def set_ids(s):
    p = os.path.join(SETS, s, "index.json")
    if not os.path.exists(p): return []
    ids = json.load(open(p)); lim = int(os.environ.get("TB_LIMIT") or 0)
    return ids[:lim] if lim else ids


def mine(key, shard, n): return zlib.crc32(key.encode()) % n == shard


def load_gt(s, tid): return json.load(open(os.path.join(SETS, s, tid + ".json"), encoding="utf-8"))


def jobs(shard, n, out):
    k = 0
    with open(out, "w", encoding="utf-8") as f:
        for s in SEL_SETS:
            for tid in set_ids(s):
                if not mine(tid, shard, n): continue
                g = load_gt(s, tid)
                f.write(json.dumps(dict(set=s, id=tid, pdf=os.path.join(SETS, s, g["pdf"]), page=0, crop_pt=g["crop_pt"])) + "\n"); k += 1
    print("jobs", k, flush=True)


def tatr_call(M, im, toks):
    """bench_models.TATR.__call__, keeping the objects and raw outputs for the port fixtures"""
    x = M.prep(im)
    t = time.perf_counter()
    lg, bx = M.sess.run(None, {"pixel_values": x.astype(np.float16) if M.f16 else x})
    tm = time.perf_counter() - t
    lg, bx = lg.astype(np.float32), bx.astype(np.float32)
    objs = M.TL.outputs_to_objects(lg[0], bx[0], im.size)
    objs0 = copy.deepcopy(objs); toks0 = copy.deepcopy(toks)
    raw, _ = M.TL.run_post(objs, copy.deepcopy(toks))
    out = []
    for c in raw:
        rs, cs = c["row_nums"], c["column_nums"]
        if not rs or not cs: continue
        out.append(dict(r0=min(rs), r1=max(rs) + 1, c0=min(cs), c1=max(cs) + 1, bbox=list(map(float, c["bbox"])), header=bool(c.get("column header")), text=c.get("cell text", "")))
    keep = [dict(row_nums=sorted(int(v) for v in c["row_nums"]), column_nums=sorted(int(v) for v in c["column_nums"]), bbox=[float(v) for v in c["bbox"]],
                 column_header=bool(c.get("column header")), projected_row_header=bool(c.get("projected row header")), text=c.get("cell text", "")) for c in raw]
    return out, tm, dict(x=x, lg=lg[0], bx=bx[0], objs=objs0, toks=toks0, raw=keep)


def run(shard, n, wdir):
    import bench_models as BM
    M = BM.TATR16(0)
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    fixdir = os.path.join(OUT, "r5fix"); os.makedirs(fixdir, exist_ok=True)
    post_fix, raw_fix, e2e_fix = [], [], []
    for s in SEL_SETS:
        of = os.path.join(OUT, "pred", f"tatr16r5_{s}_s{shard}.jsonl")
        ov_new, ov_old, nw, nfix = [], [], [], 0
        with open(of, "w", encoding="utf-8") as f:
            for tid in set_ids(s):
                if not mine(tid, shard, n): continue
                rec = dict(id=tid)
                wp = os.path.join(wdir, s, tid + ".json")
                try:
                    w = json.load(open(wp, encoding="utf-8"))
                    if "error" in w: raise RuntimeError(w["error"])
                    g = load_gt(s, tid)
                    gtok = set().union(*[tokens(c["text"]) for c in g["cells"]]) if g["cells"] else set()
                    ov_new.append(jaccard(set().union(*[tokens(x["text"]) for x in w["words"]]) if w["words"] else set(), gtok))
                    ov_old.append(g.get("text_overlap", 0.0)); nw.append(len(w["words"]))
                    im = Image.open(os.path.join(wdir, s, tid + ".png")).convert("RGB")
                    t = time.perf_counter()
                    im2, (ox, oy) = BM.prep_image(im, "pad", 20)
                    toks = BM.chunks(dict(words=w["words"]), 20, ox, oy)
                    cells, tm, cap = tatr_call(M, im2, toks)
                    rec["ms_total"] = (time.perf_counter() - t) * 1000; rec["ms_model"] = tm * 1000
                    for c in cells: c["bbox"] = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
                    rec["cells"] = cells
                    if shard == 0 and nfix < FIX_PER_SET and cap["raw"] and len(w["words"]) <= FIX_MAX_WORDS:
                        nfix += 1
                        post_fix.append(dict(id=tid, set=s, img_size=list(im2.size), objects=cap["objs"], tokens=cap["toks"], cells=cap["raw"]))
                        if len(raw_fix) < RAW_FIX:
                            raw_fix.append(dict(id=tid, img_size=list(im2.size), logits=cap["lg"].tolist(), boxes=cap["bx"].tolist(), objects=cap["objs"]))
                        if len(e2e_fix) < E2E_FIX and 15 <= len(w["words"]) <= 120:
                            x = cap["x"][0]
                            im.save(os.path.join(fixdir, f"e2e_{len(e2e_fix)}.png"))
                            e2e_fix.append(dict(id=tid, png=f"e2e_{len(e2e_fix)}.png", words=w["words"], pad=20, input_shape=list(cap["x"].shape),
                                                channel_mean=[float(x[c].astype(np.float64).mean()) for c in range(3)], cells=cap["raw"],
                                                n_objects=len(cap["objs"])))
                except Exception as e:
                    rec["cells"] = []; rec["err"] = repr(e)[:300]
                f.write(json.dumps(rec) + "\n")
        if ov_new:
            print(f"{s}: tables {len(ov_new)}  words/table median {statistics.median(nw)}  text overlap with the answer key: docling words {statistics.mean(ov_new):.4f}  r4 pdfplumber {statistics.mean(ov_old):.4f}", flush=True)
        print("done tatr16r5", s, shard, flush=True)
    if shard == 0:
        json.dump(dict(source="tables/r5/r5_tatr.py (TATR v1.1 fp16, Xenova onnx/model_fp16.onnx; postprocess.py + inference.py at microsoft/table-transformer 16d124f, fitz.Rect shim)",
                       thresholds=dict(table=0.5, column=0.5, row=0.5, column_header=0.5, projected_row_header=0.5, spanning_cell=0.5),
                       post=post_fix, raw=raw_fix, e2e=e2e_fix), open(os.path.join(fixdir, "tatr_fixtures.json"), "w"))
        print("fixtures", len(post_fix), len(raw_fix), len(e2e_fix), flush=True)


if __name__ == "__main__":
    cmd, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if cmd == "jobs": jobs(k, n, sys.argv[4])
    else: run(k, n, sys.argv[4])
