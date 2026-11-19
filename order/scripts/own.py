import json, os, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ro
import numpy as np, pdfplumber, pypdfium2 as pdfium

ORDER = os.path.dirname(HERE)
PDFDIR = os.environ.get("PDF_DIR", os.path.join(os.path.expanduser("~"), "pdfs"))
CASES = json.load(open(os.path.join(ORDER, "data", "own_expected.json"), encoding="utf-8"))


def norm(s):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (s or "").lower())).strip() + " "


def score_text(text, chains):
    t = norm(text)
    pos, found, total = {}, 0, 0
    for ci, ch in enumerate(chains):
        for si, sn in enumerate(ch):
            total += 1
            q = norm(sn).strip()
            k = t.find(" " + q + " ")
            if k < 0:
                k = t.find(q)
            if k >= 0:
                pos[(ci, si)] = k
                found += 1
    good = pairs = 0
    for ci, ch in enumerate(chains):
        for a in range(len(ch)):
            for b in range(a + 1, len(ch)):
                if (ci, a) in pos and (ci, b) in pos:
                    pairs += 1
                    good += pos[(ci, a)] < pos[(ci, b)]
    inter = 0
    for c1 in range(len(chains)):
        for c2 in range(c1 + 1, len(chains)):
            p1 = [v for (c, s), v in pos.items() if c == c1]
            p2 = [v for (c, s), v in pos.items() if c == c2]
            if p1 and p2 and not (max(p1) < min(p2) or max(p2) < min(p1)):
                inter = 1
    return dict(found=found, total=total, pair_acc=good / pairs if pairs else 0.0,
                exact=int(found == total and good == pairs and not inter), interleave=inter)


def words(pdf, page):
    with pdfplumber.open(os.path.join(PDFDIR, pdf)) as d:
        p = d.pages[page - 1]
        return [dict(text=w["text"], x0=w["x0"], top=w["top"], x1=w["x1"], bottom=w["bottom"]) for w in p.extract_words()], (float(p.width), float(p.height))


def text_in(box, ws, frac=0.5):
    x0, y0, x1, y1 = box
    out = []
    for w in ws:
        ix = max(0, min(x1, w["x1"]) - max(x0, w["x0"]))
        iy = max(0, min(y1, w["bottom"]) - max(y0, w["top"]))
        a = (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) + 1e-6
        if ix * iy / a >= frac:
            out.append(w)
    out.sort(key=lambda w: (round(w["top"] / 3), w["x0"]))
    return " ".join(w["text"] for w in out)


def docling_items(conv, pdf, page):
    import run_heron
    doc = conv.convert(os.path.join(PDFDIR, pdf), page_range=(page, page)).document.export_to_dict()
    H = doc["pages"][str(page)]["size"]["height"]
    out = []
    for it in run_heron.items(doc):
        if it["page_no"] != page or it["furniture"] or it["label"] in ("page_header", "page_footer"):
            continue
        it["box"] = run_heron.to_topleft(it["bbox"], H)
        out.append(it)
    full = {}

    def res(ref):
        _, kind, idx = ref.split("/")
        return doc[kind][int(idx)]
    texts = []

    def walk(node, inside):
        for ch in node.get("children", []):
            it = res(ch["$ref"])
            lab = it.get("label")
            pr = it.get("prov") or []
            if pr and pr[0]["page_no"] == page and (not inside or lab in ("caption", "footnote")) and lab not in ("page_header", "page_footer"):
                t = it.get("text") or ""
                if lab == "table" and it.get("data"):
                    t = " ".join(c.get("text", "") for c in it["data"].get("table_cells", []))
                texts.append(t)
            walk(it, inside or lab in ("picture", "table", "chart"))
    walk(doc["body"], False)
    return out, " ".join(texts)


def struct_text(pdf, page):
    import pikepdf
    with pikepdf.open(os.path.join(PDFDIR, pdf)) as p:
        if "/StructTreeRoot" not in p.Root:
            return None
    with pdfplumber.open(os.path.join(PDFDIR, pdf)) as d:
        pg = d.pages[page - 1]
        tree = pg.structure_tree
        bymcid = {}
        for c in pg.chars:
            if c.get("mcid") is not None:
                bymcid.setdefault(c["mcid"], []).append(c["text"])
        seq = []

        def walk(el):
            seq.extend(el.get("mcids") or [])
            for ch in el.get("children") or []:
                walk(ch)
        for el in tree:
            walk(el)
        return " ".join("".join(bymcid.get(m, [])) for m in seq)


def safe(fn):
    try:
        return fn()
    except Exception as e:
        return dict(error=(type(e).__name__ + ": " + str(e))[:200])


def main(out):
    import run_pp, run_heron
    import docling_rs as d
    run_heron.dm.ensure_env(os.environ.get("DLRS_CACHE", os.path.join(os.path.expanduser("~"), "dlrs")))
    os.environ.setdefault("DOCLING_RS_FP32", "1")
    conv = d.DocumentConverter(replace_broken_text_layer=True, heading_hierarchy=True, do_picture_classification=True,
                               do_picture_ocr=True, ocr_textline_orientation=True, use_embedded_picture_images=True)
    edge_bin = os.environ.get("EDGEPARSE_BIN")
    res = []
    for c in CASES:
        pdf, page, chains = c["pdf"], c["page"], c["chains"]
        path = os.path.join(PDFDIR, pdf)
        ws, (W, H) = words(pdf, page)
        r = dict(case=f"{pdf}#p{page}")
        body, dtext = docling_items(conv, pdf, page)
        r["docling_rs_text"] = score_text(dtext, chains)
        boxes = [i["box"] for i in body]
        labs = [i["label"] for i in body]
        bt = [text_in(b, ws) for b in boxes]
        by = lambda order, t: " ".join(t[i] for i in order)
        r["heron:docling_rs_order"] = score_text(by(range(len(boxes)), bt), chains)
        r["heron:docling_rb_py"] = safe(lambda: score_text(by(ro.docling_rb(boxes, labs, W, H), bt), chains))
        r["heron:xycut_y"] = score_text(by(ro.xycut(boxes, W, H, "y"), bt), chains)
        r["heron:xycut_x"] = score_text(by(ro.xycut(boxes, W, H, "x"), bt), chains)
        r["heron:xycutpp"] = score_text(by(ro.xycutpp(boxes, W, H), bt), chains)
        r["heron:xycutpp_lab"] = score_text(by(ro.xycutpp_lab(boxes, labs, W, H), bt), chains)
        img = pdfium.PdfDocument(path)[page - 1].render(scale=2.0).to_pil().convert("RGB")
        s = img.size[0] / W
        ppi = [dict(label=i["label"], box=[v / s for v in i["box"]]) for i in run_pp.items(np.asarray(img))]
        ppi = [i for i in ppi if i["label"] not in ("header", "footer", "number", "header_image", "footer_image")]
        pb = [i["box"] for i in ppi]
        pt = [text_in(b, ws) for b in pb]
        r["pp:ppdlv3_order"] = score_text(by(range(len(pb)), pt), chains)
        r["pp:xycutpp"] = score_text(by(ro.xycutpp(pb, W, H), pt), chains)
        r["pp:xycut_x"] = score_text(by(ro.xycut(pb, W, H, "x"), pt), chains)
        r["pp:docling_rb_py"] = safe(lambda: score_text(by(ro.docling_rb(pb, ["text"] * len(pb), W, H), pt), chains))

        def edge_pip():
            import edgeparse
            return score_text(edgeparse.convert(path, format="text", pages=str(page)), chains)
        r["edgeparse_pip"] = safe(edge_pip)

        def edge_src():
            if not edge_bin or not os.path.exists(edge_bin):
                raise RuntimeError("source build not available")
            with tempfile.TemporaryDirectory() as td:
                subprocess.run([edge_bin, path, "--format", "text", "--pages", str(page), "--output-dir", td], check=True, capture_output=True, timeout=300)
                txt = " ".join(open(os.path.join(td, f), encoding="utf-8", errors="ignore").read() for f in os.listdir(td) if f.endswith((".txt", ".text", ".md")))
            return score_text(txt, chains)
        r["edgeparse_src"] = safe(edge_src)

        def oxide():
            import pdf_oxide
            return score_text(pdf_oxide.PdfDocument(path).extract_text(page - 1), chains)
        r["pdf_oxide"] = safe(oxide)

        def odl(st):
            import opendataloader_pdf as o
            return score_text(o.convert(path, format="text", pages=str(page), to_stdout=True, quiet=True, use_struct_tree=st), chains)
        r["opendataloader"] = safe(lambda: odl(False))
        r["opendataloader_structtree"] = safe(lambda: odl(True))

        def tagged():
            t = struct_text(pdf, page)
            if t is None:
                raise RuntimeError("untagged")
            return score_text(t, chains)
        r["tagged_struct_tree"] = safe(tagged)
        res.append(r)
        print(r["case"], {k: (v.get("exact"), round(v.get("pair_acc", 0), 2)) for k, v in r.items() if isinstance(v, dict) and "exact" in v}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
