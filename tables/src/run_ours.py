import os, sys, json, re, copy, time, argparse
import numpy as np
import paths
from paths import RAW, OUT, TABLES as TDIR
import pypdfium2 as pdfium, pdfplumber

SRC = os.path.join(RAW, "ours")
TABLES = {
    "2305_p9": ("2305.03393v1.pdf", 9, (139.85, 337.22, 474.54, 470.25)),
    "2305_p10": ("2305.03393v1.pdf", 10, (143.44, 156.23, 470.95, 262.97)),
    "2206_p4": ("2206.01062.pdf", 4, (99.21, 137.67, 512.79, 293.91)),
    "2206_p6": ("2206.01062.pdf", 6, (62.16, 194.91, 285.68, 351.14)),
    "2206_p7a": ("2206.01062.pdf", 7, (80.09, 150.05, 267.75, 295.45)),
    "2206_p7b": ("2206.01062.pdf", 7, (352.62, 150.05, 523.55, 306.28)),
    "demo": ("docling-rs-demotion-repro.pdf", 1, (56.18, 402.19, 510.4, 672.89)),
}
PAD = 10.0; SCALE = 3.0
_ocr = None


def render(pdf, page, scale):
    doc = pdfium.PdfDocument(os.path.join(SRC, pdf)); pg = doc[page - 1]
    W, H = pg.get_size()
    return pg.render(scale=scale).to_pil().convert("RGB"), (W, H)


def words(pdf, page):
    with pdfplumber.open(os.path.join(SRC, pdf)) as d:
        p = d.pages[page - 1]
        return [dict(text=w["text"], x0=w["x0"], top=w["top"], x1=w["x1"], bottom=w["bottom"]) for w in p.extract_words(keep_blank_chars=False, use_text_flow=False)]


def crop(key):
    pdf, pg, (l, t, r, b) = TABLES[key]
    img, (W, H) = render(pdf, pg, SCALE)
    box = (max(0, l - PAD), max(0, t - PAD), min(W, r + PAD), min(H, b + PAD))
    return img.crop(tuple(int(round(v * SCALE)) for v in box)), box


def tokens(key, im, box):
    global _ocr
    pdf, pg, _ = TABLES[key]
    ws = words(pdf, pg); toks = []
    if ws:
        for w in ws:
            cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                toks.append(dict(text=w["text"], bbox=[(w["x0"] - box[0]) * SCALE, (w["top"] - box[1]) * SCALE, (w["x1"] - box[0]) * SCALE, (w["bottom"] - box[1]) * SCALE]))
    else:
        if _ocr is None:
            from rapidocr import RapidOCR; _ocr = RapidOCR()
        r = _ocr(np.asarray(im)[:, :, ::-1])
        for poly, txt in zip(r.boxes, r.txts):
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            toks.append(dict(text=txt, bbox=[min(xs), min(ys), max(xs), max(ys)]))
    toks.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    line = 0; last = None
    for i, t in enumerate(toks):
        cy = (t["bbox"][1] + t["bbox"][3]) / 2
        if last is not None and abs(cy - last) > 4 * SCALE / 3: line += 1
        last = cy; t["block_num"] = 0; t["line_num"] = line; t["span_num"] = i
    toks.sort(key=lambda t: (t["line_num"], t["bbox"][0]))
    for i, t in enumerate(toks): t["span_num"] = i
    return toks


def fill(cells, toks):
    for c in cells: c["toks"] = []
    for t in toks:
        tb = t["bbox"]; ta = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1])); best = None; bv = 0
        for c in cells:
            b = c["bbox"]; ix = max(0, min(b[2], tb[2]) - max(b[0], tb[0])); iy = max(0, min(b[3], tb[3]) - max(b[1], tb[1]))
            v = ix * iy / ta
            if v > bv: bv = v; best = c
        if best is not None and bv > 0.0: best["toks"].append(t)
    for c in cells:
        c["toks"].sort(key=lambda t: (t["line_num"], t["bbox"][0])); c["text"] = " ".join(t["text"] for t in c["toks"])
    return cells


def grid(cells):
    if not cells: return [], set()
    R = max(max(c["rows"]) for c in cells) + 1; C = max(max(c["cols"]) for c in cells) + 1
    g = [[""] * C for _ in range(R)]; hdr = set()
    for c in cells:
        for r in c["rows"]:
            for k in c["cols"]: g[r][k] = c.get("text", "")
            if c.get("header"): hdr.add(r)
    return g, hdr


N = lambda s: re.sub(r"\s+", "", s.lower().replace("–", "-").replace("—", "-").replace("’", "'"))


def score(key, cells):
    T = json.load(open(os.path.join(TDIR, "truth.json"), encoding="utf-8"))[key]
    tg = T["grid"]; g, hdr = grid(cells)
    R, C = len(g), (len(g[0]) if g else 0)
    pos = len(tg) * len(tg[0]); ok = 0
    for r in range(len(tg)):
        for c in range(len(tg[0])):
            p = g[r][c] if r < R and c < C else None
            if p is not None and N(p) == N(tg[r][c]): ok += 1
    trows = [tuple(N(x) for x in row) for row in tg[T["header_rows"]:]]
    prows = set(tuple(N(x) for x in row) for row in g)
    return dict(rows=R, cols=C, exp_rows=len(tg), exp_cols=len(tg[0]), header_rows=sorted(hdr), exp_header_rows=list(range(T["header_rows"])),
                cell_acc=round(ok / pos, 3), body_rows_exact=f"{sum(1 for t in trows if t in prows)}/{len(trows)}")


def show(g, hdr, maxw=14):
    return "\n".join(("H " if i in hdr else "  ") + "| " + " | ".join(x[:maxw] for x in row) for i, row in enumerate(g))


def run_model(name):
    from bench_models import TATR, SLANet
    m = {"tatr": TATR, "slanet1m": SLANet}[name](0)
    if name == "tatr": m.keep_text = True
    res = {}
    for key in TABLES:
        im, box = crop(key); toks = tokens(key, im, box)
        cells, _ = m(im, copy.deepcopy(toks))
        cc = [dict(rows=list(range(c["r0"], c["r1"])), cols=list(range(c["c0"], c["c1"])), bbox=c["bbox"], header=c["header"], text=c.get("text", "")) for c in cells]
        if name == "slanet1m": fill(cc, copy.deepcopy(toks))
        res[key] = (score(key, cc), grid(cc))
    return res


def run_tableformer():
    import docling_rs as d
    conv = d.DocumentConverter(do_ocr=True)
    docs = {}
    res = {}
    for key, (pdf, pg, (l, t, r, b)) in TABLES.items():
        if pdf not in docs: docs[pdf] = conv.convert(os.path.join(SRC, pdf)).document.export_to_dict()
        dd = docs[pdf]; ph = {int(k): v["size"]["height"] for k, v in dd["pages"].items()}
        best = None; bv = 0
        for tb in dd["tables"]:
            p = tb["prov"][0]
            if p["page_no"] != pg: continue
            bb = p["bbox"]
            x0, y0, x1, y1 = (bb["l"], ph[pg] - bb["t"], bb["r"], ph[pg] - bb["b"]) if bb.get("coord_origin") == "BOTTOMLEFT" else (bb["l"], bb["t"], bb["r"], bb["b"])
            ix = max(0, min(x1, r) - max(x0, l)); iy = max(0, min(y1, b) - max(y0, t))
            v = ix * iy / ((x1 - x0) * (y1 - y0) + (r - l) * (b - t) - ix * iy)
            if v > bv: bv = v; best = tb
        cc = []
        if best is not None:
            cc = [dict(rows=list(range(c["start_row_offset_idx"], c["end_row_offset_idx"])), cols=list(range(c["start_col_offset_idx"], c["end_col_offset_idx"])),
                       header=c["column_header"], text=c["text"]) for c in best["data"]["table_cells"]]
        res[key] = (score(key, cc), grid(cc))
    return res


def main():
    name = sys.argv[1]
    res = run_tableformer() if name == "tableformer" else run_model(name)
    out = {}
    for key, (sc, (g, hdr)) in res.items():
        print(f"### {name} {key} rows={sc['rows']}/{sc['exp_rows']} cols={sc['cols']}/{sc['exp_cols']} hdr={sc['header_rows']} (exp {sc['exp_header_rows']}) cell_acc={sc['cell_acc']} body_rows_exact={sc['body_rows_exact']}")
        print(show(g, hdr)); out[key] = sc
    v = [s["cell_acc"] for s in out.values()]
    print(f"MEAN cell_acc {name} {sum(v) / len(v):.4f} {v}")
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "pred", f"ours_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
