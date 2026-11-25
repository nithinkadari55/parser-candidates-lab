"""Round 8 arms 2 and 5, and scoring of every arm. For each table docling found (tf8.py), on its run-time crop:

  words   born-digital and text-layer pages: docling-pdf 2.84.0 PdfPage::word_cells (tables/r5/words, r5words), the
          harness's run-time source; scans: PP-OCRv6 small det + rec (ppocr.py, docling.rs's preprocessing) on the crop
  turn    text pages: the round-3 glyph-angle vote over the PDF characters in the crop (the harness's rotation fix);
          scans: none (no glyphs) or PP-LCNet_x1_0_doc_ori (official ONNX), turn = its label (counter-clockwise)
  TATR    v1.1 fp16 (Xenova onnx/model_fp16.onnx), 20 px white pad, the round-3/5 preprocessing and postprocessing,
          on the turned crop; cells turned back to the crop
  scans   also on the answer-key crop of every scan table (found by docling or not): OCR + TATR, PP-LCNet, and for
          half of them (fixed by id) the crop turned 90/180/270 (lossless), with and without PP-LCNet
Every arm is scored here (score.py: TEDS, TEDS-Struct, GriTS-Top/Con, exact rows and columns) with the run-time words
as the text that fills the cells.

usage: tatr8.py SHARD NSHARDS TF_DIR WORDS_BIN
writes $TB_WORK/out/r8/pub/s<k>/ (scores and metadata, no table text: uploaded)
       $TB_WORK/out/r8/priv/s<k>/ (cells with text and words: kept in an Actions cache for the analysis job)"""
import os, sys, json, glob, time, copy, math, gzip, zlib, subprocess, collections, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import numpy as np
from PIL import Image
import bench_models as BM
import score as SC
import score8 as S8
import ppocr
from multiprocessing import Pool

PAD = 20
MODELS_DIR = os.environ.get("R8_MODELS", os.path.join(WORK, "models"))
DL = os.environ.get("DLRS_CACHE", "/home/runner/dlrs")
LCNET = os.path.join(MODELS_DIR, "pplcnet", "inference.onnx")
ROTSCAN_SHARE = 2  # of 4: scan tables that also get a rotated copy


# ---------- geometry (harness tables/rotate.rs) ----------

def _corners(b, f):
    pts = [f(b[0], b[1]), f(b[2], b[3]), f(b[0], b[3]), f(b[2], b[1])]
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def turn_box(b, turn, w, h):
    f = {90: lambda x, y: (y, w - x), 180: lambda x, y: (w - x, h - y), 270: lambda x, y: (h - y, x)}.get(turn, lambda x, y: (x, y))
    return _corners(b, f)


def unturn_box(b, turn, w, h):
    f = {90: lambda x, y: (w - y, x), 180: lambda x, y: (w - x, h - y), 270: lambda x, y: (y, h - x)}.get(turn, lambda x, y: (x, y))
    return _corners(b, f)


def turn_image(im, turn):
    return im.rotate(turn, expand=True) if turn else im


# ---------- glyph-angle vote (r3_rotate.chars_words / text_angle, without the render) ----------

def glyph_turn(pdf_path, page_index, region):
    import pypdfium2 as pdfium
    import pypdfium2.raw as c
    doc = pdfium.PdfDocument(pdf_path); page = doc[page_index]
    W, H = page.get_size(); sx, sy = int(round(W)), int(round(H))
    tp = page.get_textpage(); n = tp.count_chars()
    dx, dy = c.c_int(), c.c_int(); angs = []
    x0, y0, x1, y1 = region
    for i in range(n):
        ch = tp.get_text_range(i, 1)
        if not ch.strip(): continue
        l, b, r, t = tp.get_charbox(i)
        pts = []
        for x, y in ((l, b), (r, t), (l, t), (r, b)):
            c.FPDF_PageToDevice(page.raw, 0, 0, sx, sy, 0, x, y, dx, dy); pts.append((dx.value, dy.value))
        cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2; cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1: angs.append(c.FPDFText_GetCharAngle(tp.raw, i))
    rot = page.get_rotation()
    if not angs: return 0, 0
    votes = collections.Counter((round(math.degrees(a) / 90.0) * 90) % 360 for a in angs)
    d = max(votes, key=votes.get)
    corr = (-d - rot) % 360
    return (360 - corr) % 360, len(angs)


# ---------- models ----------

class Lcnet:
    """PP-LCNet_x1_0_doc_ori, official ONNX (PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx), round-3 preprocessing"""

    def __init__(self):
        import onnxruntime as ort
        self.s = ort.InferenceSession(LCNET, providers=["CPUExecutionProvider"]); self.i = self.s.get_inputs()[0].name

    def __call__(self, im):
        w, h = im.size; s = 256 / min(w, h)
        r = im.resize((max(224, int(round(w * s))), max(224, int(round(h * s)))), Image.BILINEAR)
        W, H = r.size; l, t = (W - 224) // 2, (H - 224) // 2
        a = np.asarray(r.crop((l, t, l + 224, t + 224))).astype(np.float32)[:, :, ::-1] / 255.0
        a = ((a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)[None].astype(np.float32)
        p = self.s.run(None, {self.i: np.ascontiguousarray(a)})[0][0]
        return [0, 90, 180, 270][int(np.argmax(p))], float(np.max(p))


class Ocr:
    def __init__(self):
        m = os.path.join(DL, "models")
        self.det = ppocr.Detector(os.path.join(m, "ocr_det.onnx"), threads=4)
        self.rec = ppocr.Recognizer(os.path.join(m, "ocr_rec_v6.onnx"), ppocr.load_dict(os.path.join(m, "ocr_rec_v6_dict.txt")), threads=4)

    def __call__(self, im):
        rgb = np.ascontiguousarray(np.asarray(im.convert("RGB")))
        boxes = self.det(rgb, region=True); out = []
        crops = [ppocr.crop(rgb, b["quad"]) for b in boxes]
        for b, cr in zip(boxes, crops):
            if cr is None: continue
            t, conf = self.rec.one(cr)
            if t.strip() and conf >= ppocr.TEXT_SCORE: out.append(dict(text=t, bbox=[float(v) for v in b["box"]]))
        return out


def tatr(M, im, words, turn):
    """TATR on the crop turned by `turn`; cells (with text) back on the crop, model ms, total ms"""
    W, H = im.size; t0 = time.perf_counter()
    imt = turn_image(im, turn)
    wt = [dict(text=w["text"], bbox=turn_box(w["bbox"], turn, W, H) if turn else list(w["bbox"])) for w in words]
    im2, (ox, oy) = BM.prep_image(imt, "pad", PAD)
    toks = BM.chunks(dict(words=wt), PAD, ox, oy)
    cells, tm = M(im2, toks)
    for c in cells:
        b = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
        c["bbox"] = unturn_box(b, turn, W, H) if turn else b
    return cells, tm * 1000, (time.perf_counter() - t0) * 1000


def unturn_words(words, turn, w, h):
    """words read on the image turned by `turn` (size of the turned image), back on the w x h crop"""
    return [dict(text=x["text"], bbox=unturn_box(x["bbox"], turn, w, h)) for x in words] if turn else words


# ---------- scoring ----------

def gt_crop(M, lc, ocr, s, tid, g, wdir, meta, score_jobs):
    """scan tables, every one (found by docling or not): OCR + TATR on the answer-key crop, PP-LCNet on it, and for
    half of them (fixed by id) the crop turned 90/180/270 degrees with and without PP-LCNet"""
    wj = json.load(open(os.path.join(wdir, s, tid + "__gt.json"), encoding="utf-8"))
    if "error" in wj: raise RuntimeError("r5words: " + str(wj["error"]))
    im = Image.open(os.path.join(wdir, s, tid + "__gt.png")).convert("RGB")
    words = ocr(im); cells, _, _ = tatr(M, im, words, 0)
    lab, conf = lc(im)
    meta.update(gt_lcnet=lab, gt_lcnet_conf=conf)
    score_jobs.append(((tid, "gt_tatr"), g, words, cells))
    if lab:
        imL = turn_image(im, lab); wL = ocr(imL); cl, _, _ = tatr(M, imL, wL, 0)
        score_jobs.append(((tid, "gt_tatr_lcnet"), g, wL, cl))
    else:
        score_jobs.append(((tid, "gt_tatr_lcnet"), g, words, cells))
    if zlib.crc32(("rotscan|" + tid).encode()) % 4 < ROTSCAN_SHARE:
        theta = [90, 180, 270][zlib.crc32(("theta|" + tid).encode()) % 3]
        imR = turn_image(im, theta); wR = ocr(imR); cR, _, _ = tatr(M, imR, wR, 0)
        labR, confR = lc(imR)
        meta.update(rs_theta=theta, rs_need=(360 - theta) % 360, rs_lcnet=labR, rs_lcnet_conf=confR)
        score_jobs.append(((tid, "rs_none"), g, wR, cR))
        if labR == (360 - theta) % 360:
            score_jobs.append(((tid, "rs_lcnet"), g, words, cells)); meta["rs_lcnet_same_as_upright"] = True
        elif labR == 0:
            score_jobs.append(((tid, "rs_lcnet"), g, wR, cR))
        else:
            imRL = turn_image(imR, labR); wRL = ocr(imRL); cRL, _, _ = tatr(M, imRL, wRL, 0)
            score_jobs.append(((tid, "rs_lcnet"), g, wRL, cRL))


def _score(job):
    key, gt, words, cells = job
    return key, S8.score_cells(gt, words, cells)


def main():
    shard, n, tf_dir, words_bin = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    pub = os.path.join(R8OUT, "pub", f"s{shard}"); priv = os.path.join(R8OUT, "priv", f"s{shard}")
    os.makedirs(pub, exist_ok=True); os.makedirs(priv, exist_ok=True)
    tf = {}
    for p in glob.glob(os.path.join(tf_dir, "**", "tf_*_s*.jsonl"), recursive=True):
        for r in jl(p): tf[(r["set"], r["id"])] = r
    print("tf records", len(tf), flush=True)
    mine_keys = sorted(k for k in tf if mine(k[1], shard, n))
    jobs = os.path.join(priv, "jobs.jsonl"); wdir = os.path.join(WORK, "r8words", f"s{shard}")
    with open(jobs, "w", encoding="utf-8") as f:
        for s, tid in mine_keys:
            r = tf[(s, tid)]; g = load_gt(s, tid)
            if set_kind(s) == "scan":
                f.write(json.dumps(dict(set=s, id=tid + "__gt", pdf=os.path.join(SETS, s, g["pdf"]), page=0, crop_pt=g["crop_pt"])) + "\n")
            if not r.get("found"): continue
            f.write(json.dumps(dict(set=s, id=tid, pdf=os.path.join(SETS, s, g["pdf"]), page=0, crop_pt=r["crop_pt_rt"])) + "\n")
    t0 = time.time()
    subprocess.run([words_bin, jobs, wdir], check=True)
    print("r5words s", round(time.time() - t0), flush=True)
    M = BM.TATR16(0); M.keep_text = True
    lc = Lcnet(); ocr = Ocr()
    meta_rows, priv_rows, score_jobs = [], [], []
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
    nerr = 0
    for k, (s, tid) in enumerate(mine_keys):
        r = tf[(s, tid)]; g = load_gt(s, tid)
        doc = doc_of(s, g)
        meta = dict(id=tid, set=s, type=set_type(s), doc=doc, split=split_of(doc), bucket=g["tags"]["bucket"],
                    tags={t: bool(g["tags"].get(t)) for t in SC.TAGS + ["scitsr_comp", "words_aligned"] if t in g["tags"]},
                    gt_rows=g["tags"].get("rows"), gt_cols=g["tags"].get("cols"), n_span=g["tags"].get("n_span"),
                    found=bool(r.get("found")), match_iou=r.get("match_iou"), ms_tf=r.get("ms_tf"), ms_convert=r.get("ms_convert"),
                    ms_ocr_page_tf=r.get("ms_ocr_page"), tf_err=r.get("err"), has_gt_text=not g["tags"].get("no_gt_text"), cpu=cpu)
        if set_kind(s) == "scan":
            try:
                gt_crop(M, lc, ocr, s, tid, g, wdir, meta, score_jobs)
            except Exception as e:
                meta["gt_err"] = repr(e)[:300]; print("answer-key crop error", s, tid, meta["gt_err"], flush=True)
        if not r.get("found"):
            meta_rows.append(meta); continue
        try:
            wp = os.path.join(wdir, s, tid + ".json")
            wj = json.load(open(wp, encoding="utf-8"))
            if "error" in wj: raise RuntimeError("r5words: " + str(wj["error"]))
            im = Image.open(os.path.join(wdir, s, tid + ".png")).convert("RGB")
            W, H = im.size; scale = wj["scale"]; cx, cy = r["crop_pt_rt"][0], r["crop_pt_rt"][1]
            meta.update(crop_w=W, crop_h=H, scale=scale, ms_words_page=wj.get("ms_page"))
            tf_cells = [dict(c, bbox=[(c["bbox"][0] - cx) * scale, (c["bbox"][1] - cy) * scale, (c["bbox"][2] - cx) * scale, (c["bbox"][3] - cy) * scale])
                        for c in r["cells_pt"]]
            t1 = time.perf_counter(); lab, conf = lc(im); meta["ms_lcnet"] = (time.perf_counter() - t1) * 1000
            meta.update(lcnet=lab, lcnet_conf=conf)
            arms = {"tf": tf_cells}; wset = {}
            if set_kind(s) == "scan":
                t1 = time.perf_counter(); words = ocr(im); meta["ms_ocr"] = (time.perf_counter() - t1) * 1000
                meta["turn"] = 0
                cells, ms_m, ms_t = tatr(M, im, words, 0)
                arms["tatr"] = cells; meta.update(ms_tatr_model=ms_m, ms_tatr=ms_t)
                if lab:
                    imL = turn_image(im, lab); wL = ocr(imL)
                    cl, _, _ = tatr(M, imL, wL, 0)
                    arms["tatr_lcnet"] = [dict(c, bbox=unturn_box(c["bbox"], lab, W, H)) for c in cl]
                    wset["tatr_lcnet"] = unturn_words(wL, lab, W, H)
                else:
                    arms["tatr_lcnet"] = cells
            else:
                words = [dict(text=x["text"], bbox=x["bbox"]) for x in wj["words"]]
                t1 = time.perf_counter()
                turn, nglyph = glyph_turn(os.path.join(SETS, s, g["pdf"]), 0, r["crop_pt_rt"])
                meta.update(turn=turn, n_glyphs=nglyph, ms_vote=(time.perf_counter() - t1) * 1000)
                if g.get("rot"): meta.update(truth_turn=g["rot"]["truth_turn"], rot_mode=g["rot"]["mode"], rot_theta=g["rot"]["theta"])
                cells, ms_m, ms_t = tatr(M, im, words, turn)
                arms["tatr"] = cells; meta.update(ms_tatr_model=ms_m, ms_tatr=ms_t)
                if g.get("rot") and turn != g["rot"]["truth_turn"]:
                    arms["tatr_oracle_turn"], _, _ = tatr(M, im, words, g["rot"]["truth_turn"])
            meta["n_words"] = len(words); meta["n_tatr_cells"] = len(arms["tatr"]); meta["n_tf_cells"] = len(tf_cells)
            meta["arms"] = sorted(arms)
            for a, cs in arms.items():
                score_jobs.append(((tid, a), g, wset.get(a, words), cs))
            priv_rows.append(dict(id=tid, set=s, W=W, H=H, scale=scale, words=words, words_alt=wset,
                                  cells={a: [dict(r0=c["r0"], r1=c["r1"], c0=c["c0"], c1=c["c1"], bbox=c["bbox"], header=bool(c.get("header")), text=c.get("text", "")) for c in cs] for a, cs in arms.items()}))
        except Exception as e:
            nerr += 1; meta["err"] = repr(e)[:300]; print("error", s, tid, meta["err"], flush=True)
        meta_rows.append(meta)
        if k % 200 == 0: print("progress", k, len(mine_keys), flush=True)
    print("inference done; tables", len(meta_rows), "errors", nerr, "score jobs", len(score_jobs), flush=True)
    score_jobs.sort(key=lambda j: -len(j[3]))
    scores = collections.defaultdict(dict)
    with Pool(4) as p:
        for i, (key, res) in enumerate(p.imap_unordered(_score, score_jobs, chunksize=1)):
            scores[key[0]][key[1]] = res
            if i % 1000 == 0: print("scored", i, len(score_jobs), flush=True)
    for m in meta_rows: m["scores"] = scores.get(m["id"], {})
    dump_jsonl(os.path.join(pub, "meta.jsonl"), meta_rows)
    with gzip.open(os.path.join(priv, "priv.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in priv_rows: f.write(json.dumps(r) + "\n")
    json.dump(dict(shard=shard, tables=len(meta_rows), found=sum(m["found"] for m in meta_rows), errors=nerr, cpu=cpu,
                   sizes={k: os.path.getsize(v) for k, v in [("tatr_fp16", os.path.join(MODELS_DIR, "tatr", "model_fp16.onnx")), ("pplcnet", LCNET),
                                                             ("ocr_det", os.path.join(DL, "models", "ocr_det.onnx")), ("ocr_rec_v6", os.path.join(DL, "models", "ocr_rec_v6.onnx"))] if os.path.exists(v)}),
              open(os.path.join(pub, "summary.json"), "w"))
    print("shard", shard, "tables", len(meta_rows), "errors", nerr, flush=True)
    if meta_rows and nerr > 0.02 * max(1, sum(m["found"] for m in meta_rows)):
        sys.exit(f"too many errors: {nerr}")


if __name__ == "__main__":
    main()
