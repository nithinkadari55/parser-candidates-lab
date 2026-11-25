"""Round 8 data: the new sets as page PDFs with answer keys (common8.py describes the format).

  ptn     PubTabNet val (docling-project/PubTabNet_OTSL): table image on a page, answer-key cell text as the text layer
  pt2     PubTables-v2 Cropped-Tables test (kensho/PubTables-v2): table image + the PDF words as the text layer
  ftnc    FinTabNet.c test (bsmock/FinTabNet.c): table image + the PDF words; structure from the PDF annotations
  synfin  SynFinTabs test (ethanbradley/synfintabs): table image + the generator's word boxes
  tabrec  TabRecSet English (figshare 40332520): the whole photo/scan as an image-only page
  wtw     WTW test (ModelScope yichao666/WTW_test repack: table crops + HTML) as image-only pages
  ctdar   ICDAR-2019 cTDaR (GitHub cndplab-founder/ICDAR2019_cTDaR): modern -> ctdarm, historical -> ctdarh, image-only
  rotpdf  pages of the round-4 PDF sets turned 90/180/270 degrees, by /Rotate or by transforming the content (round 3)
Text-layer pages scale the image so the median word box is 9 pt high (a body-text table); image-only pages fit
the image into 842 pt. Samples are seeded and stratified as in rounds 3-5 (prep_sets.stratify).
Downloads stay in the job's scratch directory; only the derived sets are kept, in an Actions cache.
usage: prep8.py NAME"""
import os, sys, io, re, json, glob, math, random, shutil, tarfile, zipfile, subprocess, collections, urllib.request, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import numpy as np
from PIL import Image, ImageOps
from prep_sets import tag, stratify, parse_html, place

Image.MAX_IMAGE_PIXELS = None
LIMIT = int(os.environ.get("TB_LIMIT") or 0)
SCRATCH = os.environ.get("R8_SCRATCH") or os.path.join(RAW, "r8")
TARGET = dict(ptn=2500, pt2=1500, ftnc=2500, synfin=1500, tabrec=600, wtw=2000, rotpdf=700)
MARGIN = 4.0
WORD_PT = 9.0
PAGE_MARGIN = 36.0


def target(name):
    t = int(os.environ.get("R8_TARGET_" + name.upper()) or TARGET[name])
    return min(t, max(4, LIMIT)) if LIMIT else t


def sh(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode
    if r: raise RuntimeError(f"command failed ({r}): {cmd}")


def hf(repo, fn, rev=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=os.path.join(SCRATCH, repo.replace("/", "_")))


def get(url, dst, tries=4):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "table-bench/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            os.replace(dst + ".tmp", dst); return dst
        except Exception as e:
            print("download retry", k, repr(e)[:160], flush=True); time.sleep(5 + 10 * k)
    raise RuntimeError("download failed: " + url)


# ---------- answer keys ----------

def grid_html(cells, head_rows=0):
    """cells: dict(r, c, rs, cs, text, bbox) -> (PubTabNet tokens, [dict(text, bbox)] in token order). Holes become
    empty cells; a cell overlapping an earlier one is dropped (prep_tabrec.build)."""
    R = max(x["r"] + x["rs"] for x in cells); C = max(x["c"] + x["cs"] for x in cells)
    occ = {}; anchors = {}
    for k, x in enumerate(sorted(cells, key=lambda x: (x["r"], x["c"]))):
        if any((a, b) in occ for a in range(x["r"], x["r"] + x["rs"]) for b in range(x["c"], x["c"] + x["cs"])): continue
        for a in range(x["r"], x["r"] + x["rs"]):
            for b in range(x["c"], x["c"] + x["cs"]): occ[(a, b)] = 1
        anchors[(x["r"], x["c"])] = x
    toks, out = [], []
    head_rows = min(head_rows, R)
    for a in range(R):
        if a == 0 and head_rows: toks.append("<thead>")
        if a == head_rows and head_rows: toks.append("<tbody>")
        toks.append("<tr>")
        for b in range(C):
            if (a, b) in anchors:
                x = anchors[(a, b)]
                if x["rs"] > 1 or x["cs"] > 1:
                    toks.append("<td")
                    if x["cs"] > 1: toks.append(f' colspan="{x["cs"]}"')
                    if x["rs"] > 1: toks.append(f' rowspan="{x["rs"]}"')
                    toks.append(">")
                else:
                    toks.append("<td>")
                toks.append("</td>"); out.append(dict(text=x.get("text") or "", bbox=list(x.get("bbox") or [0, 0, 0, 0])))
            elif (a, b) not in occ:
                toks += ["<td>", "</td>"]; out.append(dict(text="", bbox=[0, 0, 0, 0]))
        toks.append("</tr>")
        if a == head_rows - 1 and head_rows: toks.append("</thead>")
    if head_rows and head_rows < R: toks.append("</tbody>")
    return toks, out


def tags_of(toks, cells, im, extra=None):
    return tag(dict(html=toks, cells=[[dict(tokens=list(c["text"]), bbox=c["bbox"]) for c in cells]]), im, extra)


def check_html(toks):
    rows, _ = parse_html(toks); _, R, C = place(rows)
    if R == 0 or C == 0: raise ValueError("empty table")
    return R, C


# ---------- pages ----------

def word_scale(words, default=1.0):
    hs = [w["bbox"][3] - w["bbox"][1] for w in words if w["bbox"][3] > w["bbox"][1]]
    if not hs: return default
    return float(min(1.5, max(0.25, WORD_PT / float(np.median(hs)))))


def clean(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    return s.encode("latin-1", "replace").decode("latin-1")


def text_page(im, words, s, out_pdf, img_fmt="PNG"):
    """the image at s pt per pixel on a page with a margin, the words drawn under it (round-3 make_pdfs: Helvetica
    scaled into each box, multi-line boxes split into lines). Returns the image's top-left on the page (pt)."""
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.lib.utils import ImageReader
    W, H = im.size
    PW, PH = W * s + 2 * PAGE_MARGIN, H * s + 2 * PAGE_MARGIN
    ox, oy = PAGE_MARGIN, PAGE_MARGIN
    c = canvas.Canvas(out_pdf, pagesize=(PW, PH), pageCompression=1)
    src = [w for w in words if clean(w["text"]).strip() and w["bbox"][2] > w["bbox"][0] and w["bbox"][3] > w["bbox"][1]]
    hs = sorted((w["bbox"][3] - w["bbox"][1]) * s for w in src)
    lh = max(3.0, hs[len(hs) // 2] if hs else 9.0)
    for w in src:
        t = " ".join(clean(w["text"]).split())
        x0, y0, x1, y1 = [v * s for v in w["bbox"]]
        x0 = max(x0, 0); bw = max(0.5, x1 - x0); bh = max(0.5, y1 - y0)
        n = max(1, int(round(bh / (lh * 1.15))))
        parts = t.split()
        if n == 1 or len(parts) < 2: lines = [t]
        else:
            per = max(1, -(-len(parts) // n)); lines = [" ".join(parts[i:i + per]) for i in range(0, len(parts), per)]
        lhh = bh / len(lines); fs = max(1.5, min(lhh * 0.85, lh * 0.85))
        for k, ln in enumerate(lines):
            sw = stringWidth(ln, "Helvetica", fs) or 1.0
            to = c.beginText(); to.setFont("Helvetica", fs); to.setHorizScale(100.0 * bw / sw)
            to.setTextOrigin(ox + x0, PH - (oy + y0 + (k + 1) * lhh) + 0.2 * lhh); to.textLine(ln); c.drawText(to)
    buf = io.BytesIO(); im.save(buf, format=img_fmt, **({"quality": 90} if img_fmt == "JPEG" else {})); buf.seek(0)
    c.drawImage(ImageReader(buf), ox, PH - oy - H * s, width=W * s, height=H * s)
    c.showPage(); c.save()
    return ox, oy


def image_page(im, out_pdf, margin=0.0, max_pt=842.0):
    """an image-only page (a scan): the image fitted into max_pt, JPEG. Returns (s, ox, oy)."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    W, H = im.size
    s = min(1.0, max_pt / max(W, H))
    PW, PH = W * s + 2 * margin, H * s + 2 * margin
    c = canvas.Canvas(out_pdf, pagesize=(PW, PH), pageCompression=1)
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=88); buf.seek(0)
    c.drawImage(ImageReader(buf), margin, margin, width=W * s, height=H * s)
    c.showPage(); c.save()
    return s, margin, margin


def cap(im, side=2200):
    W, H = im.size
    if max(W, H) <= side: return im, 1.0
    f = side / max(W, H)
    return im.resize((max(1, int(round(W * f))), max(1, int(round(H * f)))), Image.LANCZOS), f


def crop_pt(box_px, s, ox, oy, page=None):
    x0, y0, x1, y1 = box_px
    b = [ox + x0 * s - MARGIN, oy + y0 * s - MARGIN, ox + x1 * s + MARGIN, oy + y1 * s + MARGIN]
    b[0] = max(0.0, b[0]); b[1] = max(0.0, b[1])
    if page: b[2] = min(page[0], b[2]); b[3] = min(page[1], b[3])
    return b


class Writer:
    def __init__(self, name):
        self.name = name; self.dir = os.path.join(SETS, name)
        shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir)
        self.ids = []; self.stats = collections.Counter()

    def add(self, gt):
        json.dump(gt, open(os.path.join(self.dir, gt["id"] + ".json"), "w", encoding="utf-8"))
        self.ids.append(gt["id"]); self.stats["kept"] += 1

    def close(self):
        json.dump(sorted(self.ids), open(os.path.join(self.dir, "index.json"), "w"))
        json.dump(dict(self.stats), open(os.path.join(self.dir, "prep_stats.json"), "w"))
        b = collections.Counter(json.load(open(os.path.join(self.dir, t + ".json"), encoding="utf-8"))["tags"]["bucket"] for t in self.ids)
        print(self.name, "kept", len(self.ids), "buckets", dict(b), "stats", dict(self.stats), flush=True)


def gt_record(name, tid, doc, pdf, box_pt, toks, cells, T, kind, **extra):
    d = dict(id=tid, doc=doc, filename=doc, pdf=pdf, page_index=0, crop_pt=box_pt, html=toks, cells=cells, tags=T, kind=kind)
    d.update(extra)
    return d


def synth_table(w, tid, doc, im, table_px, toks, cells, words, extra_tags=None, fmt="PNG"):
    """a text-layer page: table image (its own crop) + words (image pixels)"""
    s = word_scale(words)
    fn = tid + ".pdf"
    ox, oy = text_page(im, words, s, os.path.join(w.dir, fn), fmt)
    tb = im.crop(tuple(int(round(v)) for v in table_px)) if table_px else im
    T = tags_of(toks, cells, tb, extra_tags)
    w.add(gt_record(w.name, tid, doc, fn, crop_pt(table_px or [0, 0, im.size[0], im.size[1]], s, ox, oy), toks, cells, T, "synth",
                    pt_per_px=s, n_words=len(words)))


# ---------- PubTabNet ----------

def prep_ptn():
    import pyarrow.parquet as pq
    from prep_sets import crop_rec
    p = hf("docling-project/PubTabNet_OTSL", "data/val-00000-of-00001-2d685be412ba1e61.parquet")
    recs = pq.read_table(p).to_pylist(); print("ptn val", len(recs), flush=True)
    pool = []
    for i, r in enumerate(recs):
        try:
            im, cells = crop_rec(r)
            pool.append((i, tag(dict(html=r["html"], cells=[[dict(tokens=list(c["text"]), bbox=c["bbox"]) for c in cells]]), im)))
        except Exception as e:
            print("skip", i, repr(e)[:120])
    chosen = stratify(pool, target("ptn"), random.Random(42))
    w = Writer("ptn")
    for i, T in chosen:
        r = recs[i]; im, cells = crop_rec(r)
        words = [dict(text=c["text"], bbox=c["bbox"]) for c in cells if c["text"].strip()]
        doc = (re.search(r"(PMC\d+)", r["filename"]) or [r["filename"]])[0]
        tid = f"ptn_{i:05d}"
        try:
            check_html(r["html"])
            synth_table(w, tid, doc, im, None, r["html"], cells, words)
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


# ---------- PubTables-v2 cropped ----------

def voc_boxes(xml_bytes):
    import xml.etree.ElementTree as ET
    r = ET.fromstring(xml_bytes); out = collections.defaultdict(list)
    for o in r.iter("object"):
        b = o.find("bndbox"); out[o.find("name").text].append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def prep_pt2():
    sys.path.insert(0, os.path.join(TABLES, "src"))
    import prep_pt2 as P
    x = os.path.join(SCRATCH, "pt2x"); os.makedirs(x, exist_ok=True)
    for part in ("tables", "words", "xml-annotations", "images"):
        p = hf("kensho/PubTables-v2", f"PubTables-v2_Cropped-Tables_test_{part}.tar.gz")
        sh(f"tar -xzf '{p}' -C '{x}' && rm -f '{p}'")
    P.X = x
    dirs = {k: P.find(k) for k in ("tables", "words", "xml_annotations", "images")}
    names = sorted(f[:-5] for f in os.listdir(dirs["tables"]) if f.endswith(".json"))
    print("pt2 tables", len(names), flush=True)
    pool = []
    for n in names:
        try:
            toks, cells, ws, ok = P.load(n, dirs)
            pool.append((n, tags_of(toks, cells, None, dict(words_aligned=ok))))
        except Exception as e:
            print("skip", n, repr(e)[:160])
    chosen = stratify(pool, target("pt2"), random.Random(42))
    w = Writer("pt2")
    for n, _ in chosen:
        tid = "pt2_" + re.sub(r"[^A-Za-z0-9]", "", n)
        try:
            toks, cells, ws, ok = P.load(n, dirs)
            check_html(toks)
            im = Image.open(os.path.join(dirs["images"], n + ".jpg")).convert("RGB")
            tb = voc_boxes(open(os.path.join(dirs["xml_annotations"], n + ".xml"), "rb").read()).get("table")
            words = [dict(text=v["text"], bbox=v["bbox"]) for v in ws]
            doc = (re.search(r"(PMC\d+)", n) or [n])[0]
            synth_table(w, tid, doc, im, tb[0] if tb else None, toks, cells, words, dict(words_aligned=ok), fmt="JPEG")
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


# ---------- FinTabNet.c ----------

def ftnc_html(t):
    """FinTabNet.c PDF annotation table -> tokens, cells (text) ; header rows = leading column-header rows"""
    R = len(t["rows"]); C = len(t["columns"])
    hdr = [bool(t["rows"][str(i)]["is_column_header"]) if str(i) in t["rows"] else bool(t["rows"][i]["is_column_header"]) for i in range(R)]
    head = 0
    while head < R and hdr[head]: head += 1
    cells = []
    for c in t["cells"]:
        rs, cs = sorted(c["row_nums"]), sorted(c["column_nums"])
        if not rs or not cs: continue
        cells.append(dict(r=rs[0], c=cs[0], rs=rs[-1] - rs[0] + 1, cs=cs[-1] - cs[0] + 1, text=c.get("pdf_text_content") or c.get("json_text_content") or ""))
    if not cells: raise ValueError("no cells")
    if max(x["r"] + x["rs"] for x in cells) != R or max(x["c"] + x["cs"] for x in cells) != C: raise ValueError("cells do not fill the grid")
    return grid_html(cells, head)


def prep_ftnc():
    x = os.path.join(SCRATCH, "ftncx"); os.makedirs(x, exist_ok=True)
    ann = hf("bsmock/FinTabNet.c", "FinTabNet.c-PDF_Annotations.tar.gz")
    st = hf("bsmock/FinTabNet.c", "FinTabNet.c-Structure.tar.gz")
    sh(f"tar -xzf '{st}' -C '{x}' --wildcards 'FinTabNet.c-Structure/test/*'")
    tests = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(x, "FinTabNet.c-Structure", "test", "*.xml")))
    print("ftnc test xml", len(tests), flush=True)
    pages = collections.defaultdict(list)
    for n in tests:
        m = re.match(r"(.+)_table_(\d+)$", n)
        if m: pages[m.group(1)].append((int(m.group(2)), n))
    annot = {}; seen_ids = []
    with tarfile.open(ann, "r|*") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith("_tables.json"): continue
            page = os.path.basename(m.name)[:-len("_tables.json")]
            if page in pages: annot[page] = json.load(t.extractfile(m))
    os.remove(ann)
    print("ftnc annotated pages for test tables", len(annot), "of", len(pages), flush=True)
    stats = collections.Counter(); pool = []; keys = {}
    for page, lst in pages.items():
        tabs = annot.get(page)
        if tabs is None: stats["no_annotation"] += len(lst); continue
        for k, n in lst:
            try:
                vb = voc_boxes(open(os.path.join(x, "FinTabNet.c-Structure", "test", n + ".xml"), "rb").read())
                nr, nc = len(vb.get("table row", [])), len(vb.get("table column", []))
                cand = [t for t in tabs if len(t["rows"]) == nr and len(t["columns"]) == nc and not t.get("exclude_for_structure")]
                pick = tabs[k] if k < len(tabs) and tabs[k] in cand else (cand[0] if len(cand) == 1 else None)
                if pick is None: stats["no_matching_annotation"] += 1; continue
                if len(seen_ids) < 5: seen_ids.append((n, pick.get("structure_id"), pick.get("split")))
                toks, cells = ftnc_html(pick)
                pool.append((n, tags_of(toks, cells, None)))
                keys[n] = (toks, cells, vb)
            except Exception as e:
                stats["error"] += 1; print("skip", n, repr(e)[:160])
    print("ftnc pool", len(pool), "stats", dict(stats), "id check", seen_ids, flush=True)
    chosen = stratify(pool, target("ftnc"), random.Random(42))
    lst = os.path.join(SCRATCH, "ftnc_members.txt")
    with open(lst, "w") as f:
        for n, _ in chosen:
            f.write(f"FinTabNet.c-Structure/images/{n}.jpg\nFinTabNet.c-Structure/words/{n}_words.json\n")
    sh(f"tar -xzf '{st}' -C '{x}' -T '{lst}' && rm -f '{st}'")
    w = Writer("ftnc"); w.stats.update(stats)
    for n, _ in chosen:
        tid = "ftnc_" + re.sub(r"[^A-Za-z0-9]", "", n)
        try:
            toks, cells, vb = keys[n]
            check_html(toks)
            im = Image.open(os.path.join(x, "FinTabNet.c-Structure", "images", n + ".jpg")).convert("RGB")
            words = [dict(text=v["text"], bbox=v["bbox"]) for v in json.load(open(os.path.join(x, "FinTabNet.c-Structure", "words", n + "_words.json"), encoding="utf-8"))]
            doc = re.sub(r"_page_\d+$", "", re.sub(r"_table_\d+$", "", n))
            synth_table(w, tid, doc, im, vb["table"][0] if vb.get("table") else None, toks, cells, words, fmt="JPEG")
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


# ---------- SynFinTabs ----------

def fit_offset(gt_words, ocr_words, ocr_boxes):
    """image = gt * a + b per axis, by least squares over words whose text is unique in both lists"""
    cg = collections.Counter(w["text"] for w in gt_words); co = collections.Counter(ocr_words)
    pos = {t: b for t, b in zip(ocr_words, ocr_boxes) if co[t] == 1}
    pairs = [(w["bbox"], pos[w["text"]]) for w in gt_words if cg[w["text"]] == 1 and w["text"] in pos and len(w["text"]) >= 2]
    if len(pairs) < 4: return None
    out = []
    for i in (0, 1):
        g = np.array([[p[0][i], 1.0] for p in pairs] + [[p[0][i + 2], 1.0] for p in pairs]); o = np.array([p[1][i] for p in pairs] + [p[1][i + 2] for p in pairs])
        (a, b), res, *_ = np.linalg.lstsq(g, o, rcond=None)
        err = np.abs(g @ np.array([a, b]) - o)
        out.append((float(a), float(b), float(np.median(err))))
    return out, len(pairs)


def prep_synfin():
    import pyarrow.parquet as pq
    p = hf("ethanbradley/synfintabs", "data/test-00000-of-00002.parquet")
    f = pq.ParquetFile(p); pool = []; recs = {}
    for g in range(f.num_row_groups):
        for r in f.read_row_group(g, columns=["id", "bbox", "rows", "representations"]).to_pylist():
            try:
                toks, texts = html_to_tokens(r["representations"]["html"])
                check_html(toks)
                cells = [dict(text=t, bbox=[0, 0, 0, 0]) for t in texts]
                pool.append((r["id"], tags_of(toks, cells, None)))
                recs[r["id"]] = (g, toks, cells)
            except Exception as e:
                print("skip", r.get("id"), repr(e)[:120])
    print("synfin pool", len(pool), flush=True)
    chosen = stratify(pool, target("synfin"), random.Random(42))
    want = collections.defaultdict(set)
    for i, _ in chosen: want[recs[i][0]].add(i)
    w = Writer("synfin"); fits = []
    for g, ids in sorted(want.items()):
        for r in f.read_row_group(g).to_pylist():
            if r["id"] not in ids: continue
            tid = "synfin_" + re.sub(r"[^A-Za-z0-9]", "", r["id"])
            try:
                _, toks, cells = recs[r["id"]]
                words = [dict(text=x["text"], bbox=list(map(float, x["bbox"]))) for row in r["rows"] for c in row["cells"] for x in c["words"] if x["text"].strip()]
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                fo = fit_offset(words, r["ocr_results"]["words"], r["ocr_results"]["bboxes"])
                if fo is None: w.stats["no_offset_fit"] += 1; continue
                (ax, bx, ex), (ay, by, ey) = fo[0]
                fits.append((ax, ay, ex, ey))
                if not (0.3 < ax < 5) or abs(ax - ay) > 0.05 * ax or ex > 3 * ax or ey > 3 * ay: w.stats["bad_offset_fit"] += 1; continue
                tb = r["bbox"]; tbi = [tb[0] * ax + bx, tb[1] * ay + by, tb[2] * ax + bx, tb[3] * ay + by]
                X0, Y0 = max(0, int(tbi[0]) - 12), max(0, int(tbi[1]) - 12)
                X1, Y1 = min(im.size[0], int(math.ceil(tbi[2])) + 12), min(im.size[1], int(math.ceil(tbi[3])) + 12)
                crop = im.crop((X0, Y0, X1, Y1))
                ws = [dict(text=x["text"], bbox=[x["bbox"][0] * ax + bx - X0, x["bbox"][1] * ay + by - Y0, x["bbox"][2] * ax + bx - X0, x["bbox"][3] * ay + by - Y0]) for x in words]
                synth_table(w, tid, tid, crop, [tbi[0] - X0, tbi[1] - Y0, tbi[2] - X0, tbi[3] - Y0], toks, cells, ws, dict(theme=r["theme"]))
            except Exception as e:
                w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    if fits:
        a = np.array(fits); print("synfin offset fits: scale x/y median", np.median(a[:, 0]), np.median(a[:, 1]), "residual px median", np.median(a[:, 2]), np.median(a[:, 3]), flush=True)
    w.close()


# ---------- scans ----------

def scan_table(w, tid, doc, pdf, s, ox, oy, page_pt, table_px, toks, cells, tb_im, extra=None):
    T = tags_of(toks, cells, tb_im, extra)
    w.add(gt_record(w.name, tid, doc, pdf, crop_pt(table_px, s, ox, oy, page_pt), toks, cells, T, "scan", pt_per_px=s))


def prep_tabrec():
    sys.path.insert(0, os.path.join(TABLES, "src"))
    import prep_tabrec as P
    z = get("https://ndownloader.figshare.com/files/40332520", os.path.join(SCRATCH, "tabrec.zip"))
    zz = zipfile.ZipFile(z); names = zz.namelist()
    root = "TabRecSet (CurveTabSet)/"
    anns = {os.path.basename(n)[:-5]: n for n in names if n.startswith(root + "TSR_TCR_annotation/english/") and n.endswith(".json")}
    rng = random.Random(42); used = set(); w = Writer("tabrec"); per = target("tabrec")
    for kind in ("english_all-line", "english_no-line", "english_three-line"):
        ims = sorted(n for n in names if n.startswith(root + f"image/{kind}/") and not n.endswith("/"))
        rng.shuffle(ims); pool = []; recs = {}
        for n in ims:
            base = os.path.splitext(os.path.basename(n))[0]
            if base not in anns or base in used: continue
            try:
                cells = P.parse(json.loads(zz.read(anns[base])))
                if not cells: continue
                R = max(c["r"] + c["rs"] for c in cells); C = max(c["c"] + c["cs"] for c in cells)
                toks, out = grid_html([dict(r=c["r"], c=c["c"], rs=c["rs"], cs=c["cs"], text=c["text"], bbox=c["box"]) for c in cells])
                T = tags_of(toks, out, None, dict(line_type=kind.split("_")[1]))
                tid = "tabrec_" + re.sub(r"[^A-Za-z0-9]", "", base) + kind.split("_")[1].replace("-", "")
                recs[tid] = (n, cells, toks, out, base); pool.append((tid, T))
            except Exception as e:
                print("skip", n, repr(e)[:160]); continue
            if len(pool) >= per * 4: break
        pick = stratify(pool, per, rng)
        for tid, _ in pick:
            n, cells, toks, out, base = recs[tid]; used.add(base)
            try:
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(zz.read(n)))).convert("RGB")
                im, f = cap(im)
                box = [max(0, min(c["box"][0] for c in cells) * f - 10), max(0, min(c["box"][1] for c in cells) * f - 10),
                       min(im.size[0], max(c["box"][2] for c in cells) * f + 10), min(im.size[1], max(c["box"][3] for c in cells) * f + 10)]
                fn = tid + ".pdf"; s, ox, oy = image_page(im, os.path.join(w.dir, fn))
                out2 = [dict(text=c["text"], bbox=[v * f for v in c["bbox"]]) for c in out]
                scan_table(w, tid, base, fn, s, ox, oy, (im.size[0] * s, im.size[1] * s), box, toks, out2, im.crop(tuple(int(v) for v in box)),
                           dict(line_type=kind.split("_")[1]))
            except Exception as e:
                w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
        print(kind, "pool", len(pool), "picked", len(pick), flush=True)
    os.remove(z); w.close()


WTW = "https://www.modelscope.cn/api/v1/datasets/yichao666/WTW_test/repo"


def prep_wtw():
    from concurrent.futures import ThreadPoolExecutor
    names = []; page = 1
    while True:
        d = json.load(urllib.request.urlopen(f"{WTW}/tree?Revision=master&Root=html&Recursive=false&PageNumber={page}&PageSize=500", timeout=120))["Data"]
        fs = [x["Path"] for x in (d.get("Files") or []) if x["Path"].endswith(".html")]
        names += fs
        if len(d.get("Files") or []) < 500: break
        page += 1
    names = sorted(os.path.basename(p)[:-5] for p in names)
    print("wtw html files", len(names), flush=True)
    hd = os.path.join(SCRATCH, "wtw"); os.makedirs(hd, exist_ok=True)

    def fetch(rel):
        dst = os.path.join(hd, rel.replace("/", "_"))
        if not os.path.exists(dst): get(f"{WTW}?Revision=master&FilePath={urllib.parse.quote(rel)}", dst)
        return dst
    import urllib.parse
    with ThreadPoolExecutor(16) as ex:
        htmls = dict(zip(names, ex.map(lambda n: fetch(f"html/{n}.html"), names)))
    pool = []; recs = {}
    for n in names:
        try:
            toks, texts = html_to_tokens(open(htmls[n], encoding="utf-8", errors="replace").read())
            check_html(toks)
            cells = [dict(text=t, bbox=[0, 0, 0, 0]) for t in texts]
            pool.append((n, tags_of(toks, cells, None))); recs[n] = (toks, cells)
        except Exception as e:
            print("skip", n, repr(e)[:120])
    chosen = stratify(pool, target("wtw"), random.Random(42))
    with ThreadPoolExecutor(16) as ex:
        imgs = dict(zip([n for n, _ in chosen], ex.map(lambda n: fetch(f"images/{n}.jpg"), [n for n, _ in chosen])))
    w = Writer("wtw")
    for n, _ in chosen:
        tid = "wtw_" + re.sub(r"[^A-Za-z0-9]", "", n)
        try:
            toks, cells = recs[n]
            im = ImageOps.exif_transpose(Image.open(imgs[n])).convert("RGB"); im, f = cap(im)
            fn = tid + ".pdf"; s, ox, oy = image_page(im, os.path.join(w.dir, fn), margin=PAGE_MARGIN)
            scan_table(w, tid, re.sub(r"_table\d+$", "", n), fn, s, ox, oy, (im.size[0] * s + 2 * PAGE_MARGIN, im.size[1] * s + 2 * PAGE_MARGIN),
                       [0, 0, im.size[0], im.size[1]], toks, cells, im)
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


def ctdar_tables(xml_path):
    import xml.etree.ElementTree as ET
    r = ET.parse(xml_path).getroot(); out = []
    pts = lambda e: [tuple(float(v) for v in p.split(",")) for p in e.find("Coords").get("points").split()]
    for t in r.iter("table"):
        poly = pts(t); cells = []
        for c in t.iter("cell"):
            sr, er, sc, ec = (int(c.get(k)) for k in ("start-row", "end-row", "start-col", "end-col"))
            cp = pts(c)
            cells.append(dict(r=sr, c=sc, rs=er - sr + 1, cs=ec - sc + 1, text="", bbox=[min(p[0] for p in cp), min(p[1] for p in cp), max(p[0] for p in cp), max(p[1] for p in cp)]))
        if not cells: continue
        r0 = min(c["r"] for c in cells); c0 = min(c["c"] for c in cells)
        for c in cells: c["r"] -= r0; c["c"] -= c0
        out.append(([min(p[0] for p in poly), min(p[1] for p in poly), max(p[0] for p in poly), max(p[1] for p in poly)], cells))
    return r.get("filename"), out


def prep_ctdar():
    d = os.path.join(SCRATCH, "ctdar")
    if not os.path.isdir(d): sh(f"git clone -q --depth 1 https://github.com/cndplab-founder/ICDAR2019_cTDaR.git '{d}' && git -C '{d}' log -1 --format=%H")
    imgs = {}
    for p in glob.glob(os.path.join(d, "**", "*.*"), recursive=True):
        if os.path.splitext(p)[1].lower() in (".jpg", ".png", ".tiff", ".tif"): imgs.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    xmls = {}
    for sub in ("test_ground_truth/TRACKB2", "test_ground_truth/TRACKB1", "training/TRACKB1/ground_truth"):
        for x in sorted(glob.glob(os.path.join(d, sub, "*.xml"))):
            xmls.setdefault(os.path.basename(x)[:-4], x)
    print("ctdar structure xml (deduplicated by image)", len(xmls), flush=True)
    ws = {"ctdarm": Writer("ctdarm"), "ctdarh": Writer("ctdarh")}
    keys = sorted(xmls)
    if LIMIT: keys = [k for k in keys if k.startswith("cTDaR_t1")][:LIMIT] + [k for k in keys if k.startswith("cTDaR_t0")][:LIMIT]
    for k in keys:
        w = ws["ctdarm" if k.startswith("cTDaR_t1") else "ctdarh"]
        try:
            fn_img, tabs = ctdar_tables(xmls[k])
            p = imgs.get(k) or imgs.get(os.path.splitext(fn_img or "")[0])
            if p is None: w.stats["no_image"] += len(tabs); continue
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im, f = cap(im)
            fn = re.sub(r"[^A-Za-z0-9_]", "", k) + ".pdf"; s, ox, oy = image_page(im, os.path.join(w.dir, fn))
            for ti, (tb, cells) in enumerate(tabs):
                try:
                    for c in cells: c["bbox"] = [v * f for v in c["bbox"]]
                    tb = [v * f for v in tb]
                    toks, out = grid_html(cells)
                    check_html(toks)
                    tid = f"{w.name}_{re.sub(r'[^A-Za-z0-9]', '', k)}_t{ti}"
                    rel = [dict(text="", bbox=[c["bbox"][0] - tb[0], c["bbox"][1] - tb[1], c["bbox"][2] - tb[0], c["bbox"][3] - tb[1]]) for c in out]
                    scan_table(w, tid, k, fn, s, ox, oy, (im.size[0] * s, im.size[1] * s), tb, toks, rel, im.crop(tuple(int(v) for v in tb)), dict(no_gt_text=True))
                except Exception as e:
                    w.stats["error"] += 1; print("skip", k, ti, repr(e)[:160])
        except Exception as e:
            w.stats["error"] += 1; print("skip", k, repr(e)[:160])
    for w in ws.values(): w.close()


# ---------- rotated text PDFs (round 3) ----------

def prep_rotpdf():
    sys.path.insert(0, os.path.join(TABLES, "src"))
    import r3_rotate as R3R
    import pypdfium2 as pdfium
    rng = random.Random(8); share = dict(r4scitsr=0.35, r4pt=0.35, r4brgm=0.15, r4icdar=0.15)
    n_all = target("rotpdf"); w = Writer("rotpdf")
    for s, f in share.items():
        ids = set_ids(s); rng.shuffle(ids); k = 0
        for tid in ids:
            if k >= max(1, int(round(n_all * f))): break
            g = load_gt(s, tid)
            try:
                src = open(os.path.join(SETS, s, g["pdf"]), "rb").read()
                pg = pdfium.PdfDocument(src)[g.get("page_index", 0)]
                if pg.get_rotation(): w.stats["already_rotated"] += 1; continue
                cl, cb, cr, ct = pg.get_cropbox()
                L, T, R, B = g["crop_pt"]
                user = [L + cl, ct - B, R + cl, ct - T]
                theta = rng.choice([90, 180, 270]); mode = ("attr", "content")[k % 2]
                pdfb, Tm = R3R.build_pdf_rotated(src, g.get("page_index", 0), theta, mode)
                box = R3R.region_device(pdfb, 0, 1.0, user, Tm)
                if box[2] - box[0] < 5 or box[3] - box[1] < 5: w.stats["empty_region"] += 1; continue
                nid = "rotpdf_" + tid; fn = nid + ".pdf"
                open(os.path.join(w.dir, fn), "wb").write(pdfb)
                d = dict(g); d.update(id=nid, pdf=fn, page_index=0, crop_pt=box, kind="pdf", src_set=s, src_id=tid,
                                      rot=dict(theta=theta, mode=mode, truth_turn=(theta if mode == "attr" else (360 - theta) % 360)))
                d.pop("words", None)
                d["tags"] = dict(g["tags"]); d["tags"]["rotated"] = True
                w.add(d); k += 1
            except Exception as e:
                w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    globals()["prep_" + sys.argv[1]]()
