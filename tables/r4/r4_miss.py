"""Gap 14: train-free table proposals where heron INT8 left no table.

SPARTAN (shlok-n/spartan @66ada02, MIT) is called as-is: its pdfium word extraction and detect_tables()
(boundary detector + text-alignment detector + unification) on the page rendered at 300 DPI.
The vector proposer ports the idea of firecrawl/pdf-inspector (MIT; src/tables/detect_lines.rs,
detect_rects.rs): ruling segments from PDF path operators, collinear joins (2 pt / 6 pt), intersecting
grids, span-matched horizontal rule bands (booktabs) and clusters of cell rectangles; dense chart grids
(>= 27 vertical edges) are dropped.

usage: r4_miss.py {dln,odb,sci} SHARD NSHARDS ART_DIR
ART_DIR holds the order run 44477997398 artifacts (layout-heron_int8-*, dln-gt-*)."""
import os, sys, io, json, glob, time, re, random, zipfile, urllib.request, collections
from r4_common import *
import numpy as np
import pypdfium2 as pdfium, pdfplumber

DPI = int(os.environ.get("R4_SPARTAN_DPI", "300"))
LIMIT = int(os.environ.get("TB_LIMIT") or 0)
TABLE_LABELS = ("Table", "Document Index")
SCI_URL = "https://zenodo.org/api/records/8165783/files/SCI-3000v1.1.zip/content"
SCI_PDFS = int(os.environ.get("R4_SCI_PDFS", "360"))


# ---------------------------------------------------------------- vector proposals (pdf-inspector idea)
def segments(pp, off):
    H, V, cells = [], [], []
    def add(x0, y0, x1, y1):
        x0 -= off[0]; x1 -= off[0]; y0 -= off[1]; y1 -= off[1]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx >= 10 and dy <= 0.035 * dx: H.append(((y0 + y1) / 2, min(x0, x1), max(x0, x1)))
        elif dy >= 10 and dx <= 0.035 * dy: V.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))
    for l in pp.lines: add(l["x0"], l["top"], l["x1"], l["bottom"])
    for r in pp.rects:
        w, h = r["width"], r["height"]
        if min(w, h) <= 2.5 and max(w, h) >= 10:
            if w >= h: add(r["x0"], (r["top"] + r["bottom"]) / 2, r["x1"], (r["top"] + r["bottom"]) / 2)
            else: add((r["x0"] + r["x1"]) / 2, r["top"], (r["x0"] + r["x1"]) / 2, r["bottom"])
        elif w >= 5 and h >= 5:
            if r.get("stroke"):
                for a in ((r["x0"], r["top"], r["x1"], r["top"]), (r["x0"], r["bottom"], r["x1"], r["bottom"]), (r["x0"], r["top"], r["x0"], r["bottom"]), (r["x1"], r["top"], r["x1"], r["bottom"])): add(*a)
            cells.append([r["x0"] - off[0], r["top"] - off[1], r["x1"] - off[0], r["bottom"] - off[1]])
    return join(H), join(V), cells


def join(rules, tol=2.0, gap=6.0):
    """(pos, lo, hi) rules: cluster by position (2 pt), then join intervals whose gap is <= 6 pt"""
    out, grp = [], []
    def flush():
        if not grp: return
        pos = float(np.mean([g[0] for g in grp]))
        cur = None
        for _, lo, hi in sorted(grp, key=lambda g: g[1]):
            if cur and lo <= cur[1] + gap: cur[1] = max(cur[1], hi)
            else:
                if cur: out.append((pos, cur[0], cur[1]))
                cur = [lo, hi]
        out.append((pos, cur[0], cur[1]))
    for r in sorted(rules):
        if grp and r[0] - grp[0][0] > tol:
            flush(); grp = []
        grp.append(r)
    flush()
    return out


def text_lines_in(lines, box):
    """lines: [(y0, y1, [(x0, x1)...])] in pt -> lines whose centre lies in box"""
    return [l for l in lines if box[1] <= (l[0] + l[1]) / 2 <= box[3] and any(box[0] <= (a + b) / 2 <= box[2] for a, b in l[2])]


def phrases(line, box):
    xs = sorted((a, b) for a, b in line[2] if box[0] <= (a + b) / 2 <= box[2])
    h = max(1.0, line[1] - line[0]); thr = max(9.6, 0.8 * h); n = 0; last = None
    for a, b in xs:
        if last is None or a - last > thr: n += 1
        last = b if last is None else max(last, b)
    return n


def vector_proposals(H, V, cells, lines, pw, ph):
    props = []
    page_area = pw * ph
    # grids: intersecting horizontal and vertical rules
    n = len(H) + len(V); parent = list(range(n))
    def f(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, (y, x0, x1) in enumerate(H):
        for j, (x, y0, y1) in enumerate(V):
            if x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3: parent[f(i)] = f(len(H) + j)
    comps = collections.defaultdict(list)
    for k in range(n): comps[f(k)].append(k)
    for ks in comps.values():
        hs = [H[k] for k in ks if k < len(H)]; vs = [V[k - len(H)] for k in ks if k >= len(H)]
        if len(hs) < 2 or len(vs) < 2: continue
        xs = sorted(set(round(v[0] / 3) for v in vs))
        if len(xs) >= 27: continue
        b = [min([h[1] for h in hs] + [v[0] for v in vs]), min([h[0] for h in hs] + [v[1] for v in vs]), max([h[2] for h in hs] + [v[0] for v in vs]), max([h[0] for h in hs] + [v[2] for v in vs])]
        if b[2] - b[0] < 30 or b[3] - b[1] < 15 or area(b) > 0.85 * page_area: continue
        if len(text_lines_in(lines, b)) < 2: continue
        props.append((b, "vec-grid"))
    # rule bands: horizontal rules with matching span (8 pt), stacked, with multi-column text rows between
    used = set(); Hs = sorted(H)
    for i, r in enumerate(Hs):
        if i in used or r[2] - r[1] < 80: continue
        chain = [i]
        for j in range(i + 1, len(Hs)):
            s = Hs[j]
            if abs(s[1] - r[1]) <= 8 and abs(s[2] - r[2]) <= 8 and s[0] - Hs[chain[-1]][0] <= 0.5 * ph: chain.append(j)
        if len(chain) < 2: continue
        b = [min(Hs[k][1] for k in chain), Hs[chain[0]][0], max(Hs[k][2] for k in chain), Hs[chain[-1]][0]]
        if b[3] - b[1] < 12: continue
        tl_ = text_lines_in(lines, b)
        multi = sum(1 for l in tl_ if phrases(l, b) >= 2)
        if len(tl_) >= 2 and multi >= 2 and multi >= 0.4 * len(tl_):
            used.update(chain); props.append((b, "vec-band"))
    # clusters of cell rectangles
    rs = [c for c in cells if area(c) < 0.5 * page_area]
    if len(rs) >= 6:
        mw = float(np.median([c[2] - c[0] for c in rs])); rs = [c for c in rs if c[2] - c[0] <= 10 * mw]
    if 4 <= len(rs) <= 3000:
        p2 = list(range(len(rs)))
        def g(a):
            while p2[a] != a: p2[a] = p2[p2[a]]; a = p2[a]
            return a
        order = sorted(range(len(rs)), key=lambda k: rs[k][0])
        for ii, a in enumerate(order):
            A = rs[a]
            for b_ in order[ii + 1:]:
                Bx = rs[b_]
                if Bx[0] > A[2] + 1.5: break
                if Bx[1] <= A[3] + 1.5 and A[1] <= Bx[3] + 1.5: p2[g(a)] = g(b_)
        cl = collections.defaultdict(list)
        for k in range(len(rs)): cl[g(k)].append(rs[k])
        for cs in cl.values():
            if len(cs) < 4 or len(set(round(c[1] / 2) for c in cs)) < 2: continue
            b = [min(c[0] for c in cs), min(c[1] for c in cs), max(c[2] for c in cs), max(c[3] for c in cs)]
            if area(b) > 0.85 * page_area or len(text_lines_in(lines, b)) < 2: continue
            props.append((b, "vec-rect"))
    kept = []
    for b, s in sorted(props, key=lambda p: -area(p[0])):
        if all(inter(b, k[0]) / max(1e-6, area(b)) <= 0.6 for k in kept): kept.append((b, s))
    return kept


# ---------------------------------------------------------------- per page
class Analyzer:
    def __init__(self):
        from spartan.config import PipelineConfig, TextSourceMode
        self.cfg = PipelineConfig(dpi=DPI, ocr_engine="none", text_source=TextSourceMode.PDF)
        from spartan.pre_processing import render_page
        from spartan.text_extraction import extract_pdf_words
        from spartan.region_segmentation import detect_tables
        from spartan.models import group_words_into_lines
        self.render, self.words, self.detect, self.lines = render_page, extract_pdf_words, detect_tables, group_words_into_lines

    def __call__(self, page, pp, W, Hh, gts, heron, gt_pics=(), cells_ref=None):
        """gts/heron in GT space (W x Hh). Returns record with proposals and per-table taxonomy."""
        import cv2
        k = DPI / 72.0
        t0 = time.perf_counter()
        bgr = self.render(page, DPI); Hp, Wp = bgr.shape[:2]
        words, _ = self.words(page, Wp, Hp)
        crop, tabs, _ = self.detect(bgr, words, self.cfg)
        t_sp = time.perf_counter() - t0
        sp = [([t.bbox.x0 + crop.x0, t.bbox.y0 + crop.y0, t.bbox.x1 + crop.x0, t.bbox.y1 + crop.y0], "spartan-" + t.method) for t in tabs]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # GT space <-> render pixels
        sx, sy, ox, oy = W / Wp, Hh / Hp, 0.0, 0.0
        mapping = "stretch"
        if cells_ref:
            best = None
            wc = [((w.bbox.x0 + w.bbox.x1) / 2, (w.bbox.y0 + w.bbox.y1) / 2) for w in words[:: max(1, len(words) // 200)]]
            for name, (a, b, c, d) in self._mappings(Wp, Hp, W, Hh).items():
                for fmt in ("xywh", "xyxy"):
                    boxes = [(r[0], r[1], r[0] + r[2], r[1] + r[3]) if fmt == "xywh" else tuple(r) for r in cells_ref]
                    hit = sum(1 for x, y in wc if any(bx[0] - 2 <= x * a + c <= bx[2] + 2 and bx[1] - 2 <= y * b + d <= bx[3] + 2 for bx in boxes))
                    if best is None or hit > best[0]: best = (hit, name, (a, b, c, d))
            if best and best[0] > 0: mapping = best[1]; sx, sy, ox, oy = best[2]
        tog = lambda bb: [bb[0] * sx + ox, bb[1] * sy + oy, bb[2] * sx + ox, bb[3] * sy + oy]
        top = lambda bb: [(bb[0] - ox) / sx, (bb[1] - oy) / sy, (bb[2] - ox) / sx, (bb[3] - oy) / sy]
        # vector proposals (pdfplumber paths, aligned to the render)
        t1 = time.perf_counter(); vec = []; H = V = []; vec_note = ""
        lines_pt = [(min(w.bbox.y0 for w in ln) / k, max(w.bbox.y1 for w in ln) / k, [(w.bbox.x0 / k, w.bbox.x1 / k) for w in ln]) for ln in self.lines([w for w in words if not w.vertical])]
        if page.get_rotation():
            vec_note = "rotated page: no vector proposals"
        else:
            try:
                pw_ = plumber_words(pp)
                off, _ = render_offset(pp, page, pw_, gray, k) if pw_ else ((0.0, 0.0), 0)
                H, V, cells = segments(pp, off)
                vec = [([b[0] * k, b[1] * k, b[2] * k, b[3] * k], s) for b, s in vector_proposals(H, V, cells, lines_pt, Wp / k, Hp / k)]
            except Exception as e:
                vec_note = "vector error " + repr(e)[:120]
        t_vec = time.perf_counter() - t1
        htab = [d[2:6] for d in heron if d[0] in TABLE_LABELS and d[1] >= 0.5]
        hpic = [d[2:6] for d in heron if d[0] == "Picture" and d[1] >= 0.5] + list(gt_pics)
        props = []
        for bb, src in sp + vec:
            g = tog(bb)
            if area(g) < 0.002 * W * Hh: continue
            props.append(dict(box=[round(v, 1) for v in g], src=src, cov_heron=round(cover(g, htab), 3), in_fig=cover(g, hpic) >= 0.5))
        # taxonomy for each GT table
        tax = []
        for g in gts:
            p = top(g); ppt = [v / k for v in p]
            ws = [w for w in words if p[0] <= w.bbox.cx <= p[2] and p[1] <= w.bbox.cy <= p[3]]
            nh = sum(1 for y, x0, x1 in H if ppt[1] - 2 <= y <= ppt[3] + 2 and min(x1, ppt[2]) - max(x0, ppt[0]) >= 0.5 * (ppt[2] - ppt[0]))
            nv = sum(1 for x, y0, y1 in V if ppt[0] - 2 <= x <= ppt[2] + 2 and min(y1, ppt[3]) - max(y0, ppt[1]) >= 0.5 * (ppt[3] - ppt[1]))
            lns = self.lines([w for w in ws if not w.vertical])
            cols = [phrases((min(w.bbox.y0 for w in ln) / k, max(w.bbox.y1 for w in ln) / k, [(w.bbox.x0 / k, w.bbox.x1 / k) for w in ln]), ppt) for ln in lns]
            X0, Y0, X1, Y1 = [int(max(0, v)) for v in p]
            patch = gray[Y0:min(Hp, Y1), X0:min(Wp, X1)]
            p1 = float(np.percentile(patch, 1)) if patch.size else 255.0; med = float(np.median(patch)) if patch.size else 255.0
            hi = max(htab, key=lambda h: iou(h, g), default=None); hv = iou(hi, g) if hi else 0.0
            other = [(iou(d[2:6], g), d[0]) for d in heron if d[0] not in TABLE_LABELS and d[1] >= 0.5]
            ob = max(other, default=(0, None))
            tax.append(dict(box=[round(v, 1) for v in g], heron_iou=round(hv, 3), heron_other=ob[1] if ob[0] >= 0.5 else None,
                            n_words=len(ws), rows=len(lns), med_cols=float(np.median(cols)) if cols else 0.0, h_rules=nh, v_rules=nv,
                            vertical_frac=(sum(1 for w in ws if w.vertical) / len(ws)) if ws else 0.0, ink_p1=round(p1, 1), bg_med=round(med, 1),
                            in_fig=cover(g, hpic) >= 0.5, area_frac=area(g) / (W * Hh)))
        return dict(W=W, H=Hh, mapping=mapping, sec_spartan=round(t_sp, 3), sec_vector=round(t_vec, 3), vec_note=vec_note,
                    n_words=len(words), gt=tax, props=props, heron_tables=[[round(v, 1) for v in h] for h in htab])

    @staticmethod
    def _mappings(Wp, Hp, W, Hh):
        s = min(W / Wp, Hh / Hp)
        return {"stretch": (W / Wp, Hh / Hp, 0.0, 0.0), "fit-center": (s, s, (W - Wp * s) / 2, (Hh - Hp * s) / 2), "fit-topleft": (s, s, 0.0, 0.0)}


# ---------------------------------------------------------------- datasets
def heron_index(art):
    det, gt = {}, {}
    for p in glob.glob(os.path.join(art, "**", "layout-heron_int8-*.jsonl"), recursive=True):
        for r in jl(p): det[r["id"]] = r
    for p in glob.glob(os.path.join(art, "**", "dln-gt-*.jsonl"), recursive=True):
        for r in jl(p): gt[r["id"]] = r
    return det, gt


def ds_dln(shard, n, art, an, f):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    det, gt = heron_index(art)
    byhash = {i.split(":")[2]: i for i in det if i.startswith("dln:")}
    fn = f"data/test-{shard:05d}-of-00006.parquet"
    p = hf_hub_download("docling-project/DocLayNet-v1.2", fn, repo_type="dataset", local_dir=os.path.expanduser("~/dln12"))
    cnt = 0; nojoin = 0
    for batch in pq.ParquetFile(p).iter_batches(batch_size=4, columns=["pdf", "metadata", "pdf_cells"]):
        d = batch.to_pydict()
        for k in range(len(d["pdf"])):
            md = d["metadata"][k] or {}; h = md.get("page_hash")
            pid = byhash.get(h)
            if pid is None or pid not in gt: nojoin += 1; continue
            g = gt[pid]; hr = det[pid]
            gts = [x[1:5] for x in g["gt"] if x[0] == "Table"]; gpics = []
            flat = [c for grp in (d["pdf_cells"][k] or []) for c in (grp if isinstance(grp, list) else [grp])]
            ref = [c["bbox"] for c in flat[:200] if isinstance(c, dict) and c.get("bbox")]
            yield pid, d["pdf"][k], 0, g["W"], g["H"], gts, hr["dets"], gpics, ref, dict(doc_category=g.get("doc_category"), heron_sec=hr.get("sec"))
            cnt += 1
            if LIMIT and cnt >= LIMIT: return
    print("dln shard", shard, "pages", cnt, "not joined", nojoin, flush=True)


def ds_odb(shard, n, art, an, f):
    from huggingface_hub import hf_hub_download, list_repo_files
    det, _ = heron_index(art)
    jp = hf_hub_download("opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset", local_dir=os.path.expanduser("~/odbj"))
    recs = {"odb:" + os.path.basename(r["page_info"]["image_path"]): r for r in json.load(open(jp, encoding="utf-8"))
            if r["page_info"]["page_attribute"].get("language") == "english"}
    have = set(x[len("ori_pdfs/"):-4] for x in list_repo_files("opendatalab/OmniDocBench", repo_type="dataset", revision="v1_0") if x.startswith("ori_pdfs/"))
    ids = sorted(i for i in recs if i in det and os.path.splitext(i[4:])[0] in have)
    print("odb english pages with heron detections", sum(1 for i in recs if i in det), "with original PDFs (v1_0 ori_pdfs)", len(ids), flush=True)
    ids = [i for j, i in enumerate(ids) if j % n == shard]
    if LIMIT: ids = ids[:LIMIT]
    for pid in ids:
        r = recs[pid]; pi = r["page_info"]; stem = os.path.splitext(pid[4:])[0]
        try:
            p = hf_hub_download("opendatalab/OmniDocBench", f"ori_pdfs/{stem}.pdf", repo_type="dataset", revision="v1_0", local_dir=os.path.expanduser("~/odbpdf"))
        except Exception as e:
            print("odb pdf missing", stem, repr(e)[:100]); continue
        gts, pics = [], []
        for e in r["layout_dets"]:
            if e.get("ignore"): continue
            pl = e["poly"]; xs, ys = pl[0::2], pl[1::2]; b = [min(xs), min(ys), max(xs), max(ys)]
            if e["category_type"] in ("table", "table_mask"): gts.append(b)
        yield pid, open(p, "rb").read(), 0, float(pi["width"]), float(pi["height"]), gts, det[pid]["dets"], pics, None, dict(heron_sec=det[pid].get("sec"))


class RangeFile(io.RawIOBase):
    def __init__(s, u):
        s.u = u; s.p = 0
        for a in range(6):
            try:
                r = urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=60); s.n = int(r.headers["Content-Length"]); break
            except Exception:
                time.sleep(10 * (a + 1))
    def seekable(s): return True
    def readable(s): return True
    def seek(s, o, w=0):
        s.p = o if w == 0 else (s.p + o if w == 1 else s.n + o); return s.p
    def tell(s): return s.p
    def readinto(s, b):
        k = len(b)
        if k == 0 or s.p >= s.n: return 0
        e = min(s.n, s.p + k) - 1
        for a in range(8):
            try:
                d = urllib.request.urlopen(urllib.request.Request(s.u, headers={"Range": f"bytes={s.p}-{e}", "User-Agent": "table-bench/1.0"}), timeout=120).read(); break
            except Exception:
                time.sleep(5 * (a + 1))
        else:
            raise IOError("range read failed")
        b[:len(d)] = d; s.p += len(d); return len(d)


def ds_sci(shard, n, art, an, f):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "order", "layout"))
    import models as LM
    heron = LM.load("heron_int8")
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(SCI_URL), 1 << 22))
    names = z.namelist()
    mism = set()
    try:
        for l in z.read([x for x in names if x.endswith("cropbox_mismatches.csv")][0]).decode("utf-8", "replace").splitlines()[1:]:
            if l.strip(): mism.add(l.split(",")[0].strip().replace(".pdf", ""))
    except Exception:
        pass
    pdfs = sorted(x for x in names if x.startswith("SCI-3000/PDFs/") and x.endswith(".pdf"))
    ann = collections.defaultdict(list)
    for x in names:
        m = re.match(r"SCI-3000/Annotations/([0-9a-f]+)-(\d+)\.json$", x)
        if m: ann[m.group(1)].append((int(m.group(2)), x))
    rng = random.Random(2026)
    pick = [p for p in pdfs if os.path.basename(p)[:-4] not in mism]
    rng.shuffle(pick); pick = pick[:SCI_PDFS]
    pick = [p for j, p in enumerate(pick) if j % n == shard]
    print("sci pdfs", len(pdfs), "cropbox mismatches", len(mism), "this shard", len(pick), flush=True)
    cnt = 0
    for pz in pick:
        pid0 = os.path.basename(pz)[:-4]
        try:
            data = z.read(pz)
        except Exception as e:
            print("sci read failed", pid0, repr(e)[:100]); continue
        pages = sorted(ann.get(pid0, []))
        if not pages: continue
        base = 0 if pages[0][0] == 0 else 1
        doc = pdfium.PdfDocument(data)
        for num, an_name in pages:
            pi = num - base
            if pi < 0 or pi >= len(doc): continue
            a = json.loads(z.read(an_name))
            Wc, Hc = float(a["canvasWidth"]), float(a["canvasHeight"])
            gts, pics = [], []
            for e in a.get("annotations", []):
                val = [b.get("value") for b in e.get("body", []) if b.get("purpose") == "img-cap-enum"]
                m = re.search(r"xywh=pixel:([\d.]+),([\d.]+),([\d.]+),([\d.]+)", e.get("target", {}).get("selector", {}).get("value", ""))
                if not val or not m: continue
                x, y, w, h = map(float, m.groups())
                if val[0] == "Table": gts.append([x, y, x + w, y + h])
                elif val[0] == "Figure": pics.append([x, y, x + w, y + h])
            pg = doc[pi]; wpt, hpt = pg.get_size()
            img = pg.render(scale=Wc / wpt).to_pil().convert("RGB")
            t = time.perf_counter(); dets = heron(img); hs = time.perf_counter() - t
            fx, fy = Wc / img.size[0], Hc / img.size[1]
            dets = [[d[0], d[1], d[2] * fx, d[3] * fy, d[4] * fx, d[5] * fy] for d in dets]
            yield f"sci:{pid0}:{num}", data, pi, Wc, Hc, gts, dets, [], None, dict(heron_sec=hs, gt_figures=pics)
            cnt += 1
        doc.close()
        if LIMIT and cnt >= LIMIT: break


def main():
    ds, shard, n, art = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    an = Analyzer()
    os.makedirs(R4OUT, exist_ok=True)
    of = os.path.join(R4OUT, f"miss-{ds}-{shard}.jsonl")
    gen = {"dln": ds_dln, "odb": ds_odb, "sci": ds_sci}[ds]
    cnt = 0
    with open(of, "w", encoding="utf-8") as f:
        for pid, data, pi, W, Hh, gts, dets, pics, ref, extra in gen(shard, n, art, an, f):
            try:
                doc = pdfium.PdfDocument(data); page = doc[pi]
                with pdfplumber.open(io.BytesIO(data)) as plb:
                    rec = an(page, plb.pages[pi], W, Hh, gts, dets, pics, ref)
                doc.close()
            except Exception as e:
                rec = dict(error=repr(e)[:200], W=W, H=Hh, gt=[dict(box=g) for g in gts], props=[])
                nerr = getattr(main, "nerr", 0) + 1; main.nerr = nerr
                if nerr <= 5:
                    import traceback; traceback.print_exc()
            rec.update(ds=ds, id=pid, **extra)
            f.write(json.dumps(rec) + "\n"); cnt += 1
            if cnt % 50 == 0: print(ds, shard, cnt, flush=True); f.flush()
    print("done", ds, shard, "pages", cnt, flush=True)


if __name__ == "__main__":
    main()
