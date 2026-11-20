import os, sys, io, json, math, random, time, copy
import numpy as np
from PIL import Image
import paths
from paths import SETS, OUT, MODELS
import bench_models as BM

R3 = os.path.join(OUT, "r3rot")
DORI = os.path.join(MODELS, "orient")


def rotate_item(im, words, theta):
    W, H = im.size
    if theta == 90: f = lambda x, y: (y, W - x)
    elif theta == 180: f = lambda x, y: (W - x, H - y)
    elif theta == 270: f = lambda x, y: (H - y, x)
    else: return im, words
    def rb(b):
        pts = [f(b[0], b[1]), f(b[2], b[3]), f(b[0], b[3]), f(b[2], b[1])]
        return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
    return im.rotate(theta, expand=True), [dict(w, bbox=rb(w["bbox"])) for w in words]


def chars_words(pdf_bytes, pageno, scale, crop_dev):
    import pypdfium2 as pdfium
    import pypdfium2.raw as c
    doc = pdfium.PdfDocument(pdf_bytes); page = doc[pageno]
    W, H = page.get_size()
    sx, sy = int(round(W * scale)), int(round(H * scale))
    tp = page.get_textpage()
    n = tp.count_chars()
    dx, dy = c.c_int(), c.c_int()
    chars = []
    for i in range(n):
        ch = tp.get_text_range(i, 1)
        l, b, r, t = tp.get_charbox(i)
        ang = c.FPDFText_GetCharAngle(tp.raw, i)
        pts = []
        for x, y in ((l, b), (r, t), (l, t), (r, b)):
            c.FPDF_PageToDevice(page.raw, 0, 0, sx, sy, 0, x, y, dx, dy); pts.append((dx.value, dy.value))
        box = [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
        chars.append((ch, box, ang))
    rot = page.get_rotation()
    x0, y0, x1, y1 = crop_dev
    words, cur, angs = [], [], []
    for ch, box, ang in chars:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = x0 <= cx <= x1 and y0 <= cy <= y1
        if inside and ch.strip(): angs.append(ang)
        if not ch.strip() or not inside:
            if cur: words.append(cur); cur = []
            continue
        if cur:
            pb = cur[-1][1]; size = max(pb[2] - pb[0], pb[3] - pb[1], 1)
            if abs((pb[0] + pb[2]) / 2 - cx) + abs((pb[1] + pb[3]) / 2 - cy) > 2.2 * size:
                words.append(cur); cur = []
        cur.append((ch, box))
    if cur: words.append(cur)
    out = []
    for w in words:
        bs = [b for _, b in w]
        out.append(dict(text="".join(ch for ch, _ in w), bbox=[min(b[0] for b in bs) - x0, min(b[1] for b in bs) - y0, max(b[2] for b in bs) - x0, max(b[3] for b in bs) - y0], gi=None))
    img = page.render(scale=scale).to_pil().convert("RGB").crop((int(x0), int(y0), int(x1), int(y1)))
    return img, out, angs, rot


def text_angle(angs, page_rot):
    if not angs: return None
    votes = {}
    for a in angs:
        d = (round(math.degrees(a) / 90.0) * 90) % 360
        votes[d] = votes.get(d, 0) + 1
    d = max(votes, key=votes.get)
    return (-d - page_rot) % 360


def build_pdf_rotated(src_pdf, pageno, theta, mode):
    from pypdf import PdfReader, PdfWriter, Transformation
    rd = PdfReader(io.BytesIO(src_pdf)); wr = PdfWriter()
    pg = rd.pages[pageno]
    if mode == "attr":
        pg.rotate(theta)
        wr.add_page(pg)
        T = None
    else:
        mb = pg.mediabox; l, b, r, t = float(mb.left), float(mb.bottom), float(mb.right), float(mb.top)
        tr = Transformation().rotate(theta)
        pts = [tr.apply_on((x, y)) for x, y in ((l, b), (r, t), (l, t), (r, b))]
        mx, my = min(p[0] for p in pts), min(p[1] for p in pts)
        tr = tr.translate(-mx, -my)
        pg.add_transformation(tr)
        nw, nh = max(p[0] for p in pts) - mx, max(p[1] for p in pts) - my
        from pypdf.generic import RectangleObject
        pg.mediabox = RectangleObject([0, 0, nw, nh]); pg.cropbox = RectangleObject([0, 0, nw, nh])
        wr.add_page(pg)
        T = tr
    bio = io.BytesIO(); wr.write(bio)
    return bio.getvalue(), T


def region_device(pdf_bytes, pageno, scale, user_box, T):
    import pypdfium2 as pdfium
    import pypdfium2.raw as c
    doc = pdfium.PdfDocument(pdf_bytes); page = doc[pageno]
    W, H = page.get_size(); sx, sy = int(round(W * scale)), int(round(H * scale))
    dx, dy = c.c_int(), c.c_int()
    l, b, r, t = user_box; pts = []
    for x, y in ((l, b), (r, t), (l, t), (r, b)):
        if T is not None: x, y = T.apply_on((x, y))
        c.FPDF_PageToDevice(page.raw, 0, 0, sx, sy, 0, x, y, dx, dy); pts.append((dx.value, dy.value))
    return [max(0, min(p[0] for p in pts)), max(0, min(p[1] for p in pts)), min(sx, max(p[0] for p in pts)), min(sy, max(p[1] for p in pts))]


def items():
    out = []
    rng = random.Random(11)
    for tid in json.load(open(os.path.join(SETS, "rot", "index.json"))):
        out.append(("rot", tid, None, None))
    for tid in json.load(open(os.path.join(SETS, "icdar", "index.json"))):
        g = json.load(open(os.path.join(SETS, "icdar", tid + ".json"), encoding="utf-8"))
        if g["tags"].get("rotated_page"):
            out.append(("icdar_real", tid, 0, "asis"))
            continue
        th = rng.choice([90, 180, 270])
        out += [("icdar_bd", tid, th, "attr"), ("icdar_bd", tid, th, "content")]
    for tid in json.load(open(os.path.join(SETS, "scitsr", "index.json"))):
        th = rng.choice([90, 180, 270])
        out += [("scitsr_bd", tid, th, "attr"), ("scitsr_bd", tid, th, "content")]
    return out


def load_item(kind, tid, theta, mode):
    if kind == "rot":
        g = json.load(open(os.path.join(SETS, "rot", tid + ".json"), encoding="utf-8"))
        im = Image.open(os.path.join(SETS, "rot", tid + ".png")).convert("RGB")
        words = BM.chunks(g, 0)
        for w in words: w["bbox"] = [w["bbox"][0], w["bbox"][1], w["bbox"][2], w["bbox"][3]]
        from make_pdfs import make_multi
        p = os.path.join(R3, "tmp_" + tid + ".pdf"); make_multi("rot", [tid], p, "tight")
        ox, oy = 20.0, 20.0
        _, _, angs, rot = chars_words(open(p, "rb").read(), 0, 1.0, [ox, oy, ox + im.size[0], oy + im.size[1]])
        os.remove(p)
        return g, im, words, text_angle(angs, rot), 90
    s = "icdar" if kind.startswith("icdar") else "scitsr"
    g = json.load(open(os.path.join(SETS, s, tid + ".json"), encoding="utf-8"))
    if s == "icdar":
        src = open(os.path.join(SETS, "icdar_pdf", g["filename"]), "rb").read(); pageno = g["page"] - 1; S = g["scale"]
        import pypdfium2 as pdfium
        pg = pdfium.PdfDocument(src)[pageno]; ml, mb, mr, mt = pg.get_mediabox()
        L, T, Rr, B = g["crop_pt"]
        user = [L + ml, mt - B, Rr + ml, mt - T]
    else:
        src = open(os.path.join(SETS, "scitsr", tid + ".pdf"), "rb").read(); pageno = 0; S = g["pdf_scale"][0]
        import pypdfium2 as pdfium
        pg = pdfium.PdfDocument(src)[0]; cl, cb, cr, ct = pg.get_cropbox()
        im0 = Image.open(os.path.join(SETS, "scitsr", tid + ".png")); cx, cy = g.get("crop_px", [0, 0])
        user = [cx / S + cl, ct - (cy + im0.size[1]) / S, (cx + im0.size[0]) / S + cl, ct - cy / S]
    if mode == "asis":
        pdfb, T = src, None
    else:
        pdfb, T = build_pdf_rotated(src, pageno, theta, mode)
        pageno = 0
    box = region_device(pdfb, pageno, S, user, T)
    im, words, angs, rot = chars_words(pdfb, pageno, S, box)
    truth = None if mode == "asis" else (360 - theta) % 360 if mode == "attr" else theta
    return g, im, words, text_angle(angs, rot), truth


class Orient:
    def __init__(self):
        import onnxruntime as ort
        self.pp = ort.InferenceSession(os.path.join(DORI, "pp-lcnet_x1_0_doc_ori.onnx"), providers=["CPUExecutionProvider"])
        self.ppi = self.pp.get_inputs()[0].name
        from onnxtr.models import classification, crop_orientation_predictor
        self.ox = crop_orientation_predictor(arch=classification.mobilenet_v3_small_crop_orientation(model_path=os.path.join(DORI, "crop_ori.onnx")))

    def pplcnet(self, im):
        w, h = im.size; s = 256 / min(w, h)
        r = im.resize((max(224, int(round(w * s))), max(224, int(round(h * s)))), Image.BILINEAR)
        W, H = r.size; l, t = (W - 224) // 2, (H - 224) // 2
        a = np.asarray(r.crop((l, t, l + 224, t + 224))).astype(np.float32)[:, :, ::-1] / 255.0
        a = ((a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)[None].astype(np.float32)
        p = self.pp.run(None, {self.ppi: a})[0][0]
        return [0, 90, 180, 270][int(np.argmax(p))], float(np.max(p))

    def onnxtr(self, im):
        res = self.ox([np.asarray(im)])
        return int(res[1][0]) % 360, float(res[2][0])


def run_models(g, im, words, tf_conv, tatr):
    out = {}
    pad = 20
    imp, (ox, oy) = BM.prep_image(im, "pad", pad)
    toks = [dict(w, bbox=[w["bbox"][0] - ox, w["bbox"][1] - oy, w["bbox"][2] - ox, w["bbox"][3] - oy]) for w in words]
    toks.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for i, t in enumerate(toks): t["span_num"] = i; t["line_num"] = 0; t["block_num"] = 0
    try:
        cells, _ = tatr(imp, copy.deepcopy(toks))
        for c in cells: c["bbox"] = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
    except Exception as e:
        cells = []
    out["tatr16"] = cells
    out["tableformer"] = tf_cells(g, im, words, tf_conv)
    return out


def tf_cells(g, im, words, conv):
    from make_pdfs import make_multi
    import run_tableformer as RT
    d = os.path.join(SETS, "_r3tmp"); os.makedirs(d, exist_ok=True)
    tid = "r3item"
    im.save(os.path.join(d, tid + ".png"))
    gg = dict(g); gg["words"] = words; gg.pop("rot", None)
    json.dump(gg, open(os.path.join(d, tid + ".json"), "w", encoding="utf-8"))
    p = os.path.join(R3, "tf.pdf"); offs = make_multi("_r3tmp", [tid], p, "letter")
    dd = conv.convert(p).document.export_to_dict(); ph = RT.page_heights(dd)
    tabs = [t for t in dd["tables"] if t["prov"]]
    if not tabs: return []
    b = max(tabs, key=lambda t: abs((t["prov"][0]["bbox"]["r"] - t["prov"][0]["bbox"]["l"]) * (t["prov"][0]["bbox"]["t"] - t["prov"][0]["bbox"]["b"])))
    ox, oy = offs[0]
    return RT.cells_of(b, ph[b["prov"][0]["page_no"]], ox, oy, 1.0)


def main():
    k, n = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(R3, exist_ok=True)
    import docling_rs as d
    from docling_rs.options import PdfFormatOption, PdfPipelineOptions, InputFormat
    po = PdfPipelineOptions(do_ocr=False)
    conv = d.DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}, do_ocr=False)
    tatr = BM.TATR16(0)
    orient = Orient()
    import score
    lim = int(os.environ.get("TB_LIMIT") or 0)
    allit = items()
    if lim:
        keep, seen = [], {}
        for x in allit:
            key = (x[0], x[3])
            if seen.get(key, 0) < lim: keep.append(x); seen[key] = seen.get(key, 0) + 1
        allit = keep
    its = [x for i, x in enumerate(allit) if i % n == k]
    of = open(os.path.join(R3, f"rot-{k}.jsonl"), "w", encoding="utf-8")
    for kind, tid, theta, mode in its:
        try:
            g, im, words, t_text, truth = load_item(kind, tid, theta, mode)
        except Exception as e:
            print("skip", kind, tid, mode, repr(e)[:200], flush=True); continue
        t0 = time.perf_counter(); pp, ppc = orient.pplcnet(im); t1 = time.perf_counter(); oxa, oxc = orient.onnxtr(im); t2 = time.perf_counter()
        preds = dict(none=0, text=t_text if t_text is not None else 0, pplcnet=(360 - pp) % 360, pplcnet_raw=pp, onnxtr=oxa, onnxtr_inv=(360 - oxa) % 360)
        if truth is not None: preds["oracle"] = truth
        res = {}
        for corr in sorted(set(preds.values())):
            imc, wc = rotate_item(im, words, (360 - corr) % 360) if corr else (im, words)
            gg = dict(g); gg["words"] = wc; gg["id"] = tid
            cells = run_models(gg, imc, wc, conv, tatr)
            for m, cl in cells.items():
                sc = score.score_one((gg, dict(id=tid, cells=cl)))
                res[f"{m}@{corr}"] = sc["teds_s"] if sc["teds_s"] is not None else sc["grits_top"]
        rec = dict(kind=kind, id=tid, mode=mode, theta=theta, truth=truth, preds=preds, conf=dict(pplcnet=ppc, onnxtr=oxc),
                   ms=dict(pplcnet=1000 * (t1 - t0), onnxtr=1000 * (t2 - t1)), scores=res, n_words=len(words), bucket=g["tags"]["bucket"])
        of.write(json.dumps(rec) + "\n"); of.flush()
    of.close()


if __name__ == "__main__":
    main()
