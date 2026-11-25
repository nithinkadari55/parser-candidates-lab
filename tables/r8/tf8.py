"""Round 8 arm 1: TableFormer through docling-rs 2.84.0 (fp32 heron layout, as the harness runs it) on each set's page
PDFs. For every answer-key table, the docling table whose layout box overlaps the answer-key crop most (IoU >= 0.3,
as rounds 4-5) is the table docling found; its layout box + 4 pt (clamped to the page) is the run-time crop every arm
uses. Tables docling does not find are recorded with found=False (the harness has nothing to run TATR on then).

Born-digital and text-layer pages: do_ocr=False (rounds 4-5). Scans and photos (image-only pages): the harness's OCR
settings (PP-OCRv6 small recognizer, ppocr engine, English, ocr_scale 3, text-line orientation).
TableFormer time: DOCLING_RS_TIMING's "tableformer" stage per conversion (stderr), split evenly over docling's tables.

usage: tf8.py SHARD NSHARDS [SETS]  -> $TB_WORK/out/r8/tf/tf_<set>_s<k>.jsonl (cells in page points, top-left)"""
import os, sys, json, time, re, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *

MARGIN_PT = 4.0
DL = os.environ.get("DLRS_CACHE", "/home/runner/dlrs")
M = os.path.join(DL, "models")
ENV = {
    "DOCLING_RS_CACHE_DIR": DL,
    "DOCLING_RS_MODELS_DIR": M,
    "DOCLING_RS_FP32": "1",
    "DOCLING_OCR_REC_ONNX": os.path.join(M, "ocr_rec_v6.onnx"),
    "DOCLING_OCR_DICT": os.path.join(M, "ocr_rec_v6_dict.txt"),
    "DOCLING_RS_PDF_WORKERS": "1",
    "DOCLING_RS_PDF_INTRA": "4",
    "DOCLING_RS_TF_INTRA": "4",
    "DOCLING_RS_OCR_SESSIONS": "2",
    "DOCLING_RS_OCR_ENGINE": "ppocr",
    "DOCLING_RS_OCR_LANG": "en",
    "DOCLING_RS_TIMING": "1",
}
STAGE = re.compile(r"^\s+(\S+)\s+([0-9.]+) ms\s+[0-9.]+%\s+\((\d+) calls\)")


class Timing:
    """docling.rs prints its cumulative per-stage totals to stderr after each conversion; fd 2 goes to a file."""

    def __init__(self, path):
        self.path = path
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 2); os.close(fd)
        self.pos = 0; self.last = {}

    def delta(self):
        sys.stderr.flush()
        with open(self.path, encoding="utf-8", errors="replace") as f:
            f.seek(self.pos); txt = f.read(); self.pos = f.tell()
        cur = None
        for line in txt.splitlines():
            if "DOCLING_RS timing" in line: cur = {}
            elif cur is not None:
                m = STAGE.match(line)
                if m: cur[m.group(1)] = float(m.group(2))
        if cur is None: return {}
        out = {k: v - self.last.get(k, 0.0) for k, v in cur.items()}
        self.last = cur
        return out


def tl(b, h):
    if b.get("coord_origin") == "BOTTOMLEFT": return [b["l"], h - b["t"], b["r"], h - b["b"]]
    return [b["l"], b["t"], b["r"], b["b"]]


def cells_pt(t, h):
    out = []
    for c in t["data"]["table_cells"]:
        b = c.get("bbox")
        if not b: continue
        out.append(dict(r0=c["start_row_offset_idx"], r1=c["end_row_offset_idx"], c0=c["start_col_offset_idx"], c1=c["end_col_offset_idx"],
                        bbox=tl(b, h), header=bool(c.get("column_header"))))
    return out


def converter(scan):
    for k, v in ENV.items(): os.environ.setdefault(k, v)
    import docling_rs
    if scan:
        return docling_rs.DocumentConverter(do_ocr=True, do_table_structure=True, ocr_engine="ppocr", ocr_lang="en", ocr_scale=3.0,
                                            ocr_textline_orientation=True)
    return docling_rs.DocumentConverter(do_ocr=False, do_table_structure=True)


def main():
    shard, n = int(sys.argv[1]), int(sys.argv[2])
    sets = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else ALL_SETS
    odir = os.path.join(R8OUT, "tf"); os.makedirs(odir, exist_ok=True)
    timing = Timing(os.path.join(odir, f"stderr_s{shard}.log"))
    convs = {}
    for s in sets:
        ids = set_ids(s)
        if not ids: print("no set", s, flush=True); continue
        scan = set_kind(s) == "scan"
        if scan not in convs: convs[scan] = converter(scan)
        conv = convs[scan]
        bypage = {}
        for tid in ids:
            g = load_gt(s, tid); bypage.setdefault((g["pdf"], 0), []).append(g)
        found = total = errs = 0
        with open(os.path.join(odir, f"tf_{s}_s{shard}.jsonl"), "w", encoding="utf-8") as f:
            for (fn, pi) in sorted(bypage):
                if not mine(f"{fn}|{pi}", shard, n): continue
                t0 = time.perf_counter(); err = None
                try:
                    dd = conv.convert(os.path.join(SETS, s, fn), page_range=(pi + 1, pi + 1)).document.export_to_dict()
                except Exception as e:
                    dd = None; err = repr(e)[:300]; errs += 1; print("convert failed", s, fn, err, flush=True)
                ms = (time.perf_counter() - t0) * 1000
                st = timing.delta()
                tabs = []
                if dd:
                    sizes = {int(k): v["size"] for k, v in dd["pages"].items()}
                    for t in dd["tables"]:
                        if t.get("label") != "table" or len(t.get("prov") or []) != 1: continue
                        p = t["prov"][0]; size = sizes.get(p["page_no"]) or list(sizes.values())[0]
                        tabs.append((tl(p["bbox"], size["height"]), t, size))
                ntab = max(1, len(tabs))
                for g in bypage[(fn, pi)]:
                    total += 1
                    rec = dict(id=g["id"], set=s, ms_convert=ms / len(bypage[(fn, pi)]), ms_tf=st.get("tableformer", 0.0) / ntab if st else None,
                               ms_layout_page=st.get("layout.predict"), ms_ocr_page=sum(v for k, v in st.items() if k.startswith("ocr")) if st else None,
                               n_tables_on_page=len(tabs), err=err)
                    best = max(tabs, key=lambda x: iou(x[0], g["crop_pt"]), default=None)
                    bv = iou(best[0], g["crop_pt"]) if best else 0.0
                    rec["match_iou"] = bv
                    if best is not None and bv >= 0.3:
                        b, t, size = best
                        rec["crop_pt_rt"] = [max(0.0, b[0] - MARGIN_PT), max(0.0, b[1] - MARGIN_PT), min(size["width"], b[2] + MARGIN_PT), min(size["height"], b[3] + MARGIN_PT)]
                        rec["layout_box"] = b; rec["page_size"] = [size["width"], size["height"]]
                        rec["cells_pt"] = cells_pt(t, size["height"]); rec["found"] = True; found += 1
                    else:
                        rec["cells_pt"] = []; rec["found"] = False
                    f.write(json.dumps(rec) + "\n")
                f.flush()
        print(f"{s} shard {shard}: docling found {found} of {total} tables; conversion errors {errs}", flush=True)
    sizes = {}
    for p in glob.glob(os.path.join(M, "tableformer", "*")) + glob.glob(os.path.join(M, "layout_heron*.onnx")):
        sizes[os.path.relpath(p, M)] = os.path.getsize(p)
    json.dump(sizes, open(os.path.join(odir, "sizes.json"), "w"))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); raise
