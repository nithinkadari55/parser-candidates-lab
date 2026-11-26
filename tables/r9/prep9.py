"""Round 9 data: the new sets as page PDFs with answer keys (round-8 format, common8.py; extra keys in common9.py).

  pulse     PulseBench-Tab (HF pulse-ai/PulseBench-Tab): every table image, image-only page, GT html
  rdtb      RD-TableBench (HF reducto/rd-tablebench): every table image (_images/*.jpg), image-only page, GT html
  odb       OmniDocBench v1.5 (HF opendatalab/OmniDocBench): every table block with html, on its full page image
  tabreczh  TabRecSet Chinese part (figshare 40332520, the round-8 file): seeded sample per line type
  mustard   MUSTARD (HF badrivishalk/MUSTARD): every table image, OTSL -> grid (structure only)
  iflytab   iFLYTAB val (HF mirror gvl610/iFLYTAB): seeded sample, row/column polygons + cells -> grid (structure only)
  cisol     CISOL TD-TSR validation (HF bevaya/CISOL): every table on the full page, rows/columns/spanning cells -> grid
All are image-only pages (OCR words at run time). Downloads stay in the job's scratch directory; only the derived
sets are kept, in an Actions cache. Logs print counts and ids only, never table text.
usage: prep9.py NAME"""
import os, sys, io, re, json, glob, random, zipfile, collections, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common9 import *
import prep8 as P8
from prep8 import grid_html, tags_of, check_html, image_page, cap, Writer, scan_table, hf, get, sh, PAGE_MARGIN
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
LIMIT = int(os.environ.get("TB_LIMIT") or 0)
SCRATCH = P8.SCRATCH
TARGET = dict(pulse=0, rdtb=0, odb=0, tabreczh=1500, mustard=0, iflytab=1000, cisol=0)  # 0 = every table
REV = {"pulse-ai/PulseBench-Tab": "e0e1fd5c4b3f7daa0c6afed88177ed9242e6593d", "reducto/rd-tablebench": "7748503e2bd5f210d27aa2ef5fdf4b8aa13099bb",
       "opendatalab/OmniDocBench": "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec", "badrivishalk/MUSTARD": None,
       "gvl610/iFLYTAB": None, "bevaya/CISOL": "9904c5193b0d05d83437bad3fb3456efb9ba8c57"}


def hf_list(repo, path, rev="main"):
    """file paths under path (dataset repo), via the tree API"""
    import urllib.request, urllib.parse
    out = []; url = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}?recursive=true&limit=1000"
    while url:
        for k in range(8):
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "table-bench/1.0"}), timeout=120); break
            except Exception as e:
                print("list retry", k, repr(e)[:120], flush=True); time.sleep(30 * (k + 1))
        out += [x["path"] for x in json.load(r) if x.get("type") == "file"]
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = nxt.group(1) if nxt else None
    return out


def hf_files(repo, paths, root, rev="main"):
    """download dataset files one by one with pacing and back-off (anonymous runners get HTTP 429 from
    snapshot_download on repos with thousands of files)"""
    import urllib.parse
    for i, p in enumerate(paths):
        dst = os.path.join(root, p)
        if os.path.exists(dst): continue
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(p)}"
        for k in range(10):
            try:
                get(url, dst, tries=1); break
            except Exception as e:
                time.sleep(20 * (k + 1))
        else:
            raise RuntimeError("download failed: " + p)
        time.sleep(0.12)
        if i % 500 == 0: print("hf files", i, len(paths), flush=True)


def target(name):
    t = TARGET[name]
    if LIMIT: return min(t, LIMIT) if t else LIMIT
    return t


def take(seq, name):
    n = target(name)
    return seq[:n] if n else seq


def html_gt(html):
    toks, texts = html_to_tokens(html)
    check_html(toks)
    return toks, [dict(text=t, bbox=[0, 0, 0, 0]) for t in texts]


def crop_page(w, tid, doc, im, box_px, toks, cells, extra_tags=None, margin=PAGE_MARGIN, **extra):
    """image-only page (JPEG), the table box in image pixels -> answer key"""
    im, f = cap(im)
    box = [v * f for v in box_px]
    fn = tid + ".pdf"
    s, ox, oy = image_page(im, os.path.join(w.dir, fn), margin=margin)
    page = (im.size[0] * s + 2 * margin, im.size[1] * s + 2 * margin)
    tb = im.crop(tuple(int(round(v)) for v in box))
    scan_table(w, tid, doc, fn, s, ox, oy, page, box, toks, cells, tb, extra_tags)
    g = json.load(open(os.path.join(w.dir, tid + ".json"), encoding="utf-8"))
    g.update(extra)
    json.dump(g, open(os.path.join(w.dir, tid + ".json"), "w", encoding="utf-8"))


def safe(s): return re.sub(r"[^A-Za-z0-9]", "", str(s))


# ---------- PulseBench-Tab ----------

LANG = {"english": "en", "chinese": "zh", "spanish": "es", "russian": "ru", "french": "fr", "japanese": "ja", "arabic": "ar",
        "german": "de", "korean": "ko"}


def prep_pulse():
    import pyarrow.parquet as pq
    p = hf("pulse-ai/PulseBench-Tab", "data/train-00000-of-00001.parquet", REV["pulse-ai/PulseBench-Tab"])
    f = pq.ParquetFile(p); w = Writer("pulse"); n = 0; langs = collections.Counter()
    print("pulse columns", f.schema_arrow.names, flush=True)
    for g in range(f.num_row_groups):
        for r in f.read_row_group(g).to_pylist():
            if target("pulse") and n >= target("pulse"): break
            n += 1
            tid = "pulse_" + safe(r["sample_id"])
            try:
                toks, cells = html_gt(r["ground_truth_html"])
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                lang = LANG.get(str(r.get("language", "")).lower(), str(r.get("language", ""))[:8])
                langs[lang] += 1
                crop_page(w, tid, tid, im, [0, 0, im.size[0], im.size[1]], toks, cells, lang=lang)
            except Exception as e:
                w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    print("pulse languages", dict(langs), flush=True)
    w.close()


# ---------- RD-TableBench ----------

def prep_rdtb():
    from huggingface_hub import hf_hub_download
    z = hf_hub_download("reducto/rd-tablebench", "rd-tablebench.zip", repo_type="dataset", revision=REV["reducto/rd-tablebench"],
                        local_dir=os.path.join(SCRATCH, "rdtb"))
    zz = zipfile.ZipFile(z); names = zz.namelist()
    gts = {os.path.basename(n)[:-5]: n for n in names if "/groundtruth/" in n and n.endswith(".html")}
    ims = {os.path.splitext(os.path.basename(n))[0]: n for n in names if "/_images/" in n and n.lower().endswith((".jpg", ".png", ".jpeg"))}
    keys = take(sorted(k for k in gts if k in ims), "rdtb")
    print("rdtb gt", len(gts), "images", len(ims), "used", len(keys), flush=True)
    w = Writer("rdtb")
    for k in keys:
        tid = "rdtb_" + safe(k)[:60]
        try:
            toks, cells = html_gt(zz.read(gts[k]).decode("utf-8", "replace"))
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(zz.read(ims[k])))).convert("RGB")
            crop_page(w, tid, tid, im, [0, 0, im.size[0], im.size[1]], toks, cells)
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


# ---------- OmniDocBench ----------

def prep_odb():
    from huggingface_hub import hf_hub_download, snapshot_download
    rev = REV["opendatalab/OmniDocBench"]; root = os.path.join(SCRATCH, "odb")
    js = json.load(open(hf_hub_download("opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset", revision=rev, local_dir=root), encoding="utf-8"))
    tabs = []; attrs = collections.Counter(); keys_seen = collections.Counter()
    for page in js:
        pi = page.get("page_info", {}); pa = pi.get("page_attribute", {}) or {}
        for k, d in enumerate(page.get("layout_dets", [])):
            if d.get("category_type") != "table": continue
            keys_seen.update(d.keys())
            if d.get("ignore"): attrs["ignored"] += 1; continue
            html = d.get("html") or ""
            if "<t" not in html: attrs["no_html"] += 1; continue
            tabs.append((pi["image_path"], k, d, pa))
    print("odb pages", len(js), "table blocks kept", len(tabs), dict(attrs), "block keys", dict(keys_seen), flush=True)
    tabs = take(sorted(tabs, key=lambda x: (x[0], x[1])), "odb")
    need = sorted({t[0] for t in tabs})
    hf_files("opendatalab/OmniDocBench", ["images/" + n for n in need], root, rev)
    w = Writer("odb"); src = collections.Counter()
    for img, k, d, pa in tabs:
        tid = "odb_" + safe(os.path.splitext(img)[0])[:70] + f"_b{k}"
        try:
            toks, cells = html_gt(d["html"])
            im = ImageOps.exif_transpose(Image.open(os.path.join(root, "images", img))).convert("RGB")
            p = d["poly"]; xs, ys = p[0::2], p[1::2]
            box = [max(0, min(xs)), max(0, min(ys)), min(im.size[0], max(xs)), min(im.size[1], max(ys))]
            a = d.get("attribute") or {}
            ds = pa.get("data_source", "")
            src[ds] += 1
            LM = {"table_en": "en", "table_simplified_chinese": "zh", "table_en_ch_mixed": "zh/en", "english": "en", "simplified_chinese": "zh",
                  "en_ch_mixed": "zh/en", "table_traditional_chinese": "zh-tw"}
            raw = str(a.get("language") or pa.get("language") or "")
            lang = LM.get(raw, raw)
            meta = {k2: a.get(k2) for k2 in ("table_layout", "with_span", "line", "include_equation", "include_background", "include_photo") if k2 in a}
            meta["data_source"] = ds
            crop_page(w, tid, os.path.splitext(img)[0], im, box, toks, cells, r9type=ODB_TYPE.get(ds, "scientific"), lang=lang, src_meta=meta)
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    print("odb page sources", dict(src), flush=True)
    w.close()


# ---------- TabRecSet Chinese ----------

def prep_tabreczh():
    sys.path.insert(0, os.path.join(TABLES, "src"))
    import prep_tabrec as P
    z = get("https://ndownloader.figshare.com/files/40332520", os.path.join(SCRATCH, "tabrec.zip"))
    zz = zipfile.ZipFile(z); names = zz.namelist()
    root = "TabRecSet (CurveTabSet)/"
    adirs = collections.Counter(n.split("/")[2] for n in names if n.startswith(root + "TSR_TCR_annotation/") and n.count("/") >= 3)
    idirs = collections.Counter(n.split("/")[2] for n in names if n.startswith(root + "image/") and n.count("/") >= 3)
    print("tabrec annotation dirs", dict(adirs), "image dirs", dict(idirs), flush=True)
    anns = {os.path.basename(n)[:-5]: n for n in names if n.startswith(root + "TSR_TCR_annotation/chinese/") and n.endswith(".json")}
    kinds = sorted(k for k in idirs if k.startswith("chinese"))
    rng = random.Random(42); used = set(); w = Writer("tabreczh")
    per = max(1, target("tabreczh") // max(1, len(kinds)))
    for kind in kinds:
        ims = sorted(n for n in names if n.startswith(root + f"image/{kind}/") and not n.endswith("/"))
        rng.shuffle(ims); pool = []; recs = {}
        for n in ims:
            base = os.path.splitext(os.path.basename(n))[0]
            if base not in anns or base in used: continue
            try:
                cells = P.parse(json.loads(zz.read(anns[base])))
                if not cells: continue
                toks, out = grid_html([dict(r=c["r"], c=c["c"], rs=c["rs"], cs=c["cs"], text=c["text"], bbox=c["box"]) for c in cells])
                check_html(toks)
                lt = kind.split("_", 1)[1] if "_" in kind else kind
                T = tags_of(toks, out, None, dict(line_type=lt))
                tid = "tabreczh_" + safe(base) + safe(lt)
                recs[tid] = (n, cells, toks, out, base, lt); pool.append((tid, T))
            except Exception as e:
                print("skip", n, repr(e)[:160]); continue
            if len(pool) >= per * 4: break
        pick = P8.stratify(pool, per, rng)
        for tid, _ in pick:
            n, cells, toks, out, base, lt = recs[tid]; used.add(base)
            try:
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(zz.read(n)))).convert("RGB")
                box = [max(0, min(c["box"][0] for c in cells) - 10), max(0, min(c["box"][1] for c in cells) - 10),
                       min(im.size[0], max(c["box"][2] for c in cells) + 10), min(im.size[1], max(c["box"][3] for c in cells) + 10)]
                im2, f = cap(im)
                out2 = [dict(text=c["text"], bbox=[v * f for v in c["bbox"]]) for c in out]
                crop_page(w, tid, base, im, box, toks, out2, dict(line_type=lt), margin=0.0, lang="zh")
            except Exception as e:
                w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
        print(kind, "pool", len(pool), "picked", len(pick), flush=True)
    os.remove(z); w.close()


# ---------- MUSTARD (OTSL) ----------

def otsl_cells(seq):
    """OTSL string (F/E cell, L left-merge, U up-merge, X both, N row end) -> cells dict(r, c, rs, cs)"""
    rows, cur = [], []
    for ch in seq.strip():
        if ch == "N": rows.append(cur); cur = []
        elif ch in "FELUX": cur.append(ch)
    if cur: rows.append(cur)
    if not rows: raise ValueError("empty OTSL")
    C = max(len(r) for r in rows)
    grid = [r + ["E"] * (C - len(r)) for r in rows]
    cells = []
    for i, r in enumerate(grid):
        for j, t in enumerate(r):
            if t not in "FE": continue
            cs = 1
            while j + cs < C and grid[i][j + cs] in "LX": cs += 1
            rs = 1
            while i + rs < len(grid) and grid[i + rs][j] in "UX": rs += 1
            cells.append(dict(r=i, c=j, rs=rs, cs=cs, text="", bbox=[0, 0, 0, 0]))
    return cells


def prep_mustard():
    from huggingface_hub import snapshot_download
    root = os.path.join(SCRATCH, "mustard")
    hf_files("badrivishalk/MUSTARD", ["testset/merged.txt"], root)
    lab = {}
    for line in open(os.path.join(root, "testset", "merged.txt"), encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) >= 2: lab[parts[0]] = parts[-1]
    keys = take(sorted(lab), "mustard")
    hf_files("badrivishalk/MUSTARD", ["testset/" + k for k in keys], root)
    print("mustard labels", len(lab), "used", len(keys), "groups", dict(collections.Counter("/".join(k.split("/")[:2]) for k in keys)), flush=True)
    w = Writer("mustard")
    for k in keys:
        tid = "mustard_" + safe(k.replace(".png", "").replace(".jpg", ""))[:70]
        try:
            p = os.path.join(root, "testset", k)
            if not os.path.exists(p): w.stats["no_image"] += 1; continue
            toks, cells = grid_html(otsl_cells(lab[k]))
            check_html(toks)
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            seg = k.split("/")
            lang = seg[1] if seg[0] == "indic" and len(seg) > 2 else seg[0]
            crop_page(w, tid, tid, im, [0, 0, im.size[0], im.size[1]], toks, cells, dict(no_gt_text=True), lang=lang,
                      src_meta=dict(group=seg[0]))
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


# ---------- grids from row / column bands (iFLYTAB, CISOL) ----------

def bbox_of(poly):
    if poly and isinstance(poly[0], (list, tuple)):
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    else:
        xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def band_hits(lo, hi, bands, share=0.5):
    out = []
    for i, (a, b) in enumerate(bands):
        ov = min(hi, b) - max(lo, a)
        if ov > share * max(1e-6, b - a): out.append(i)
    return out


def bands_grid(rows, cols, spans, header_rows=0):
    """rows/cols: boxes; spans: boxes of merged cells -> cells dict(r, c, rs, cs, bbox) (TATR-style grid)"""
    rows = sorted(rows, key=lambda b: (b[1] + b[3]) / 2); cols = sorted(cols, key=lambda b: (b[0] + b[2]) / 2)
    if not rows or not cols: raise ValueError("no rows or columns")
    rb = [(b[1], b[3]) for b in rows]; cb = [(b[0], b[2]) for b in cols]
    taken = {}; cells = []
    for sp in spans:
        rr = band_hits(sp[1], sp[3], rb); cc = band_hits(sp[0], sp[2], cb)
        if not rr or not cc or (len(rr) == 1 and len(cc) == 1): continue
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if any((a, b) in taken for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)): continue
        for a in range(r0, r1 + 1):
            for b in range(c0, c1 + 1): taken[(a, b)] = 1
        cells.append(dict(r=r0, c=c0, rs=r1 - r0 + 1, cs=c1 - c0 + 1, text="", bbox=[cb[c0][0], rb[r0][0], cb[c1][1], rb[r1][1]]))
    for a in range(len(rows)):
        for b in range(len(cols)):
            if (a, b) not in taken: cells.append(dict(r=a, c=b, rs=1, cs=1, text="", bbox=[cb[b][0], rb[a][0], cb[b][1], rb[a][1]]))
    return grid_html(cells, header_rows)


def prep_iflytab():
    from huggingface_hub import snapshot_download
    root = os.path.join(SCRATCH, "iflytab")
    files = hf_list("gvl610/iFLYTAB", "original/val")
    js = sorted(p for p in files if p.endswith(".json"))
    rng = random.Random(42); rng.shuffle(js)
    pick0 = take(js, "iflytab"); fs = set(files)
    hf_files("gvl610/iFLYTAB", [p for j in pick0 for p in (j, j[:-5] + ".png", j[:-5] + ".jpg") if p in fs], root)
    pick = [os.path.join(root, p) for p in pick0]
    print("iflytab val labels", len(js), "used", len(pick), flush=True)
    w = Writer("iflytab"); shown = 0; kinds = collections.Counter()
    for jp in sorted(pick):
        base = os.path.splitext(os.path.basename(jp))[0]
        tid = "iflytab_" + safe(base)
        try:
            d = json.load(open(jp, encoding="utf-8"))
            if d.get("cell") and shown < 2:
                print("iflytab cell sample structure", type(d["cell"][0]).__name__, (list(d["cell"][0].keys()) if isinstance(d["cell"][0], dict) else str(d["cell"][0])[:80]), flush=True); shown += 1
            spans = []
            for c in d.get("cell") or []:
                poly = c.get("points") or c.get("poly") or c.get("bbox") if isinstance(c, dict) else c
                if poly: spans.append(bbox_of(poly))
            toks, cells = bands_grid([bbox_of(r) for r in d["row"]], [bbox_of(c) for c in d["col"]], spans)
            check_html(toks)
            img = next((p for p in (jp[:-5] + e for e in (".png", ".jpg", ".jpeg")) if os.path.exists(p)), None)
            if img is None: w.stats["no_image"] += 1; continue
            im = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
            wl = bool(d.get("is_wireless")); kinds["wireless" if wl else "wired"] += 1
            allb = [bbox_of(r) for r in d["row"]] + [bbox_of(c) for c in d["col"]]
            box = [max(0, min(b[0] for b in allb)), max(0, min(b[1] for b in allb)), min(im.size[0], max(b[2] for b in allb)), min(im.size[1], max(b[3] for b in allb))]
            crop_page(w, tid, tid, im, box, toks, cells, dict(no_gt_text=True, line_type="wireless" if wl else "wired"), src_meta=dict(wireless=wl))
        except Exception as e:
            w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    print("iflytab kinds", dict(kinds), flush=True)
    w.close()


def prep_cisol():
    import pyarrow.parquet as pq
    p = hf("bevaya/CISOL", "td_tsr/validation-00000-of-00001.parquet", REV["bevaya/CISOL"])
    t = pq.read_table(p); print("cisol columns", t.schema.names, "rows", t.num_rows, flush=True)
    recs = t.to_pylist(); w = Writer("cisol"); shown = False; n = 0
    for r in recs:
        objs = r.get("annotations") or r.get("objects") or {}
        if isinstance(objs, dict):
            cats = objs.get("category_name") or objs.get("category") or []; boxes = objs.get("bbox") or []
        else:
            cats = [o.get("category_name") or o.get("category") for o in objs]; boxes = [o.get("bbox") for o in objs]
        if not shown:
            print("cisol objects keys", list(objs.keys()) if isinstance(objs, dict) else "list", "categories", dict(collections.Counter(map(str, cats))), flush=True); shown = True
        xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
        by = collections.defaultdict(list)
        for c, b in zip(cats, xyxy): by[str(c)].append(b)
        im = None
        for ti, tb in enumerate(sorted(by.get("table", []), key=lambda b: (b[1], b[0]))):
            if target("cisol") and n >= target("cisol"): break
            inside = lambda b: (b[0] + b[2]) / 2 >= tb[0] - 2 and (b[0] + b[2]) / 2 <= tb[2] + 2 and (b[1] + b[3]) / 2 >= tb[1] - 2 and (b[1] + b[3]) / 2 <= tb[3] + 2
            rows = [b for b in by.get("row", []) if inside(b)]; cols = [b for b in by.get("column", []) if inside(b)]
            spans = [b for b in by.get("spanning_cell", []) + by.get("cell", []) if inside(b)]
            heads = [b for b in by.get("header", []) if inside(b)]
            tid = f"cisol_{safe(os.path.splitext(r.get('file_name') or str(r.get('image_id')))[0])[:60]}_t{ti}"
            try:
                rs = sorted(rows, key=lambda b: (b[1] + b[3]) / 2)
                hr = sum(1 for b in rs if any((b[1] + b[3]) / 2 >= h[1] and (b[1] + b[3]) / 2 <= h[3] for h in heads))
                toks, cells = bands_grid(rows, cols, spans, header_rows=hr)
                check_html(toks)
                if im is None: im = ImageOps.exif_transpose(Image.open(io.BytesIO(r["image"]["bytes"]))).convert("RGB")
                crop_page(w, tid, r.get("file_name") or tid, im, tb, toks, cells, dict(no_gt_text=True), margin=0.0,
                          src_meta=dict(type_tag=r.get("type_tag"), size_tag=r.get("size_tag")))
                n += 1
            except Exception as e:
                w.stats["error"] += 1; print("skip", tid, repr(e)[:160])
    w.close()


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    globals()["prep_" + sys.argv[1]]()
