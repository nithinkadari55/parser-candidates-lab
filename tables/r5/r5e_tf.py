"""Round 5e: the harness's crop rule. TableFormer through docling-rs 2.84.0 with the fp32 heron layout (as the harness
runs it) on the round-4 source pages; for each answer-key table, the docling table whose layout box overlaps the
answer-key crop most (IoU >= 0.3, as round 4) gives the run-time crop: its layout box + 4 pt, clamped to the page.
Tables docling does not find on the page are left out (the harness has nothing to run TATR on then).

usage: r5e_tf.py SHARD NSHARDS  -> pred/tfrt_<set>_s<k>.jsonl with cells in the run-time crop's pixels and crop_pt_rt"""
import os, sys, json, time, zlib
os.environ.setdefault("DOCLING_RS_FP32", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r4"))
from r4_common import *

MARGIN_PT = 4.0


def set_ids(s):
    p = os.path.join(SETS, s, "index.json")
    if not os.path.exists(p): return []
    ids = json.load(open(p)); lim = int(os.environ.get("TB_LIMIT") or 0)
    return ids[:lim] if lim else ids


def mine(key, shard, n): return zlib.crc32(key.encode()) % n == shard


def main():
    shard, n = int(sys.argv[1]), int(sys.argv[2])
    import docling_rs as d
    from docling_rs.options import PdfFormatOption, PdfPipelineOptions, InputFormat
    from run_tableformer import tl, cells_of
    po = PdfPipelineOptions(do_ocr=False)
    conv = d.DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}, do_ocr=False)
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    for s in SEL_SETS:
        bypdf = {}
        for tid in set_ids(s):
            g = json.load(open(os.path.join(SETS, s, tid + ".json"), encoding="utf-8"))
            bypdf.setdefault(g["pdf"], []).append(g)
        found = total = 0
        with open(os.path.join(OUT, "pred", f"tfrt_{s}_s{shard}.jsonl"), "w", encoding="utf-8") as f:
            for fn in sorted(bypdf):
                if not mine(fn, shard, n): continue
                t0 = time.perf_counter()
                try:
                    dd = conv.convert(os.path.join(SETS, s, fn)).document.export_to_dict()
                except Exception as e:
                    dd = None; print("convert failed", fn, repr(e)[:200], flush=True)
                dt = (time.perf_counter() - t0) * 1000
                tabs = []
                if dd:
                    sizes = {int(k): v["size"] for k, v in dd["pages"].items()}
                    for t in dd["tables"]:
                        if t.get("label") != "table" or len(t.get("prov") or []) != 1: continue
                        p = t["prov"][0]; size = sizes.get(p["page_no"]) or list(sizes.values())[0]
                        tabs.append((tl(p["bbox"], size["height"]), t, size))
                for g in bypdf[fn]:
                    total += 1
                    rec = dict(id=g["id"], ms_total=dt / len(bypdf[fn]), n_tables_on_page=len(tabs))
                    best = max(tabs, key=lambda x: iou(x[0], g["crop_pt"]), default=None)
                    bv = iou(best[0], g["crop_pt"]) if best else 0.0
                    rec["match_iou"] = bv
                    if best is not None and bv >= 0.3:
                        b, t, size = best
                        crop = [max(0.0, b[0] - MARGIN_PT), max(0.0, b[1] - MARGIN_PT),
                                min(size["width"], b[2] + MARGIN_PT), min(size["height"], b[3] + MARGIN_PT)]
                        rec["crop_pt_rt"] = crop; rec["layout_box"] = b
                        rec["cells"] = cells_of(t, size["height"], crop[0], crop[1], S); rec["tf_mode"] = "page"
                        found += 1
                    else:
                        rec["cells"] = []; rec["tf_mode"] = "none"; rec["err"] = "no matching table"
                    f.write(json.dumps(rec) + "\n")
                f.flush()
        print(f"{s} shard {shard}: docling found {found} of {total} tables", flush=True)


if __name__ == "__main__":
    main()
