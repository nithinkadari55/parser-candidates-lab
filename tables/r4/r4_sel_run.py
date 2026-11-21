"""Gap 15: TableFormer (docling-rs 2.84.0, full PDF pipeline on the real source page) and TATR v1.1 fp16 (same crop,
our pdfplumber words) on the r4 sets.  usage: r4_sel_run.py {tableformer,tatr16} SHARD NSHARDS"""
import os, sys, json, time, traceback, zlib
from r4_common import *
import numpy as np
import pypdfium2 as pdfium
from PIL import Image


def set_ids(s):
    p = os.path.join(SETS, s, "index.json")
    if not os.path.exists(p): return []
    ids = json.load(open(p)); lim = int(os.environ.get("TB_LIMIT") or 0)
    return ids[:lim] if lim else ids


def load_gt(s, tid): return json.load(open(os.path.join(SETS, s, tid + ".json"), encoding="utf-8"))


def mine(key, shard, n): return zlib.crc32(key.encode()) % n == shard


def run_tf(shard, n):
    import docling_rs as d
    from docling_rs.options import PdfFormatOption, PdfPipelineOptions, InputFormat
    from run_tableformer import tl, cells_of, page_heights
    po = PdfPipelineOptions(do_ocr=False)
    conv = d.DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}, do_ocr=False)
    tmpd = os.path.join(OUT, "tfcrop"); os.makedirs(tmpd, exist_ok=True)

    def convert(path):
        t = time.perf_counter(); r = conv.convert(path); dt = (time.perf_counter() - t) * 1000
        dd = r.document.export_to_dict(); ph = page_heights(dd)
        tabs = []
        for t_ in dd["tables"]:
            if not t_["prov"]: continue
            p = t_["prov"][0]; h = ph.get(p["page_no"], list(ph.values())[0] if ph else 0)
            tabs.append((tl(p["bbox"], h), t_, h))
        return tabs, dt

    for s in SEL_SETS:
        ids = set_ids(s); bypdf = {}
        for tid in ids:
            g = load_gt(s, tid); bypdf.setdefault(g["pdf"], []).append(g)
        of = os.path.join(OUT, "pred", f"tableformer_{s}_s{shard}.jsonl")
        with open(of, "w", encoding="utf-8") as f:
            for fn in sorted(bypdf):
                if not mine(fn, shard, n): continue
                path = os.path.join(SETS, s, fn)
                try:
                    tabs, dt = convert(path)
                except Exception as e:
                    tabs, dt = [], 0.0; print("convert failed", fn, repr(e)[:200], flush=True)
                for g in bypdf[fn]:
                    L, T, R, B = g["crop_pt"]
                    rec = dict(id=g["id"], ms_total=dt / len(bypdf[fn]), n_tables_on_page=len(tabs))
                    best = max(tabs, key=lambda x: iou(x[0], [L, T, R, B]), default=None)
                    bv = iou(best[0], [L, T, R, B]) if best else 0.0
                    rec["match_iou"] = bv
                    if best is not None and bv >= 0.3:
                        rec["cells"] = cells_of(best[1], best[2], L, T, S); rec["tf_mode"] = "page"
                    else:
                        try:
                            src = pdfium.PdfDocument(path); new = pdfium.PdfDocument.new(); new.import_pages(src, [0]); pg = new[0]
                            cl, cb, cr, ct = pg.get_cropbox()
                            l, r_ = max(cl, cl + L - 24), min(cr, cl + R + 24); t, b = min(ct, ct - (T - 24)), max(cb, ct - (B + 24))
                            pg.set_cropbox(l, b, r_, t)
                            tmp = os.path.join(tmpd, "crop.pdf"); new.save(tmp); new.close(); src.close()
                            dx, dy = l - cl, ct - t
                            tabs2, dt2 = convert(tmp)
                            rec["ms_total"] += dt2
                            gc = [L - dx, T - dy, R - dx, B - dy]
                            best2 = max(tabs2, key=lambda x: iou(x[0], gc), default=None)
                            bv2 = iou(best2[0], gc) if best2 else 0.0
                            rec["match_iou_crop"] = bv2
                            if best2 is not None and bv2 > 0.1:
                                rec["cells"] = cells_of(best2[1], best2[2], L - dx, T - dy, S); rec["tf_mode"] = "crop"
                            else:
                                rec["cells"] = []; rec["tf_mode"] = "none"; rec["err"] = "no table detected"
                        except Exception as e:
                            rec["cells"] = []; rec["tf_mode"] = "none"; rec["err"] = repr(e)[:200]
                    f.write(json.dumps(rec) + "\n")
                f.flush()
        print("done tableformer", s, shard, flush=True)


def run_tatr(shard, n):
    import bench_models as BM
    M = BM.TATR16(0)
    for s in SEL_SETS:
        of = os.path.join(OUT, "pred", f"tatr16_{s}_s{shard}.jsonl")
        cache = {}
        with open(of, "w", encoding="utf-8") as f:
            for tid in set_ids(s):
                if not mine(tid, shard, n): continue
                g = load_gt(s, tid)
                rec = dict(id=tid)
                try:
                    if g["pdf"] not in cache:
                        cache.clear(); doc = pdfium.PdfDocument(os.path.join(SETS, s, g["pdf"]))
                        cache[g["pdf"]] = np.asarray(doc[0].render(scale=S).to_pil().convert("RGB")); doc.close()
                    a = cache[g["pdf"]]
                    X0, Y0, X1, Y1 = [int(round(v * S)) for v in g["crop_pt"]]
                    im = Image.fromarray(a[Y0:Y1, X0:X1])
                    t = time.perf_counter()
                    im2, (ox, oy) = BM.prep_image(im, "pad", 20)
                    toks = BM.chunks(g, 20, ox, oy)
                    cells, tm = M(im2, toks)
                    rec["ms_total"] = (time.perf_counter() - t) * 1000; rec["ms_model"] = tm * 1000
                    for c in cells: c["bbox"] = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
                    rec["cells"] = cells
                except Exception as e:
                    rec["cells"] = []; rec["err"] = repr(e)[:300]; traceback.print_exc()
                f.write(json.dumps(rec) + "\n")
        print("done tatr16", s, shard, flush=True)


if __name__ == "__main__":
    m, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    (run_tf if m == "tableformer" else run_tatr)(k, n)
