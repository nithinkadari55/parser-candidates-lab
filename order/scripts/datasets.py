import glob, json, os, re

HOME = os.path.expanduser("~")
ODB = os.environ.get("ODB_DIR", os.path.join(HOME, "odb"))
DRD = os.environ.get("DRD_DIR", os.path.join(HOME, "drdoc"))

BODY_EXCL = {"header", "footer", "page_number", "page_footnote", "abandon", "equation_caption"}
FURNITURE = {"header", "footer", "page_number", "page_footnote", "abandon", "equation_caption"}
LMAP = {"title": "section_header", "text_block": "text", "figure": "picture", "table": "table",
        "figure_caption": "caption", "table_caption": "caption", "figure_footnote": "footnote",
        "table_footnote": "footnote", "equation_isolated": "formula", "reference": "list_item",
        "code_txt": "code", "code_algorithm": "code", "code_txt_caption": "caption", "equation_explanation": "text",
        "chart_mask": "picture", "table_mask": "table", "unknown_mask": "picture", "algorithm_mask": "code",
        "text_mask": "text", "need_mask": "text", "header": "page_header", "footer": "page_footer",
        "page_number": "page_footer", "page_footnote": "footnote", "abandon": "page_footer",
        "equation_caption": "formula_number", "chart": "picture", "chart_caption": "caption"}
EDGE_TAGS = ("fullwidth_interrupts_columns", "footnotes", "caption_beside_figure", "exam_numbered_items", "landscape_or_spread")


def box(e):
    p = e["poly"]
    xs, ys = p[0::2], p[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def split_of(key):
    return "dev" if sum(map(ord, key)) % 2 == 0 else "test"


def doc_key(ds, name, info):
    if ds == "drdoc":
        return "drdoc:" + str(info.get("page_name")), int(info.get("page_no") or 0)
    stem = os.path.splitext(name)[0]
    for pat in (r"^(.*)_page_0*(\d+)$", r"^(.*\.pdf)_(\d+)$", r"^(.*?)[_-](\d+)$"):
        m = re.match(pat, stem)
        if m and not stem.startswith("page-"):
            return "odb:" + m.group(1), int(m.group(2))
    return "odb:" + stem, 0


def tags_and_bucket(lay, src, cats, allcats, boxes, W, H):
    has_ft = bool(allcats & {"figure", "table", "chart_mask", "table_mask"})
    if lay in ("three_column", "1andmore_column", "other_layout") or src in ("newspaper", "magazine", "PPT2PDF"):
        bucket = "complex"
    elif lay == "double_column" or has_ft:
        bucket = "medium"
    else:
        bucket = "simple"
    tags = []
    for b in boxes:
        if b[2] - b[0] > 0.55 * W:
            ab = [c for c in boxes if c[2] - c[0] < 0.45 * W and c[3] <= b[1] + 2]
            bl = [c for c in boxes if c[2] - c[0] < 0.45 * W and c[1] >= b[3] - 2]
            if any(x[2] <= y[0] or y[2] <= x[0] for x in ab for y in ab) and any(x[2] <= y[0] or y[2] <= x[0] for x in bl for y in bl):
                tags.append("fullwidth_interrupts_columns")
                break
    if allcats & {"table_footnote", "figure_footnote", "page_footnote"}:
        tags.append("footnotes")
    done = False
    for i, c in enumerate(cats):
        if c in ("figure_caption", "table_caption") and not done:
            for j, c2 in enumerate(cats):
                if c2 in ("figure", "table"):
                    b, f = boxes[i], boxes[j]
                    iv = min(b[3], f[3]) - max(b[1], f[1])
                    if iv > 0.5 * (b[3] - b[1]) and (b[0] >= f[2] - 2 or b[2] <= f[0] + 2):
                        tags.append("caption_beside_figure")
                        done = True
                        break
    if src == "exam_paper":
        tags.append("exam_numbered_items")
    if W > 1.15 * H and src != "PPT2PDF":
        tags.append("landscape_or_spread")
    if src == "PPT2PDF":
        tags.append("slides")
    if src in ("newspaper", "magazine"):
        tags.append("newspaper_magazine")
    return bucket, tags


def _page(ds, rec, img_path):
    info = rec["page_info"]
    a = info.get("page_attribute", {})
    W, H = float(info["width"]), float(info["height"])
    name = info.get("image_path", "")
    pid = ds + ":" + (os.path.relpath(img_path, DRD).replace("\\", "/") if ds == "drdoc" else os.path.basename(name))
    dets = rec["layout_dets"]
    body = [e for e in dets if e.get("order") is not None and e["category_type"] not in BODY_EXCL and not e.get("ignore")]
    body.sort(key=lambda e: e["order"])
    full = [e for e in dets if not e.get("ignore") and e["category_type"] != "list_group"
            and (e.get("order") is not None or e["category_type"] in FURNITURE)]
    bid = {id(e): k for k, e in enumerate(body)}
    full_ids = [bid.get(id(e), -1) for e in full]
    cats = [e["category_type"] for e in body]
    allcats = {e["category_type"] for e in dets}
    boxes = [box(e) for e in body]
    bucket, tags = tags_and_bucket(a.get("layout"), a.get("data_source"), cats, allcats, boxes, W, H)
    dk, pno = doc_key(ds, os.path.basename(name), info)
    return dict(id=pid, ds=ds, img=img_path, W=W, H=H, layout=a.get("layout"), source=a.get("data_source"),
                bucket=bucket, tags=tags, split=split_of(os.path.basename(name) if ds == "odb" else pid), doc=dk, page_no=pno,
                boxes=boxes, cats=cats, labels=[LMAP.get(c, "text") for c in cats], texts=[(e.get("text") or "") for e in body],
                fboxes=[box(e) for e in full], fcats=[e["category_type"] for e in full],
                flabels=[LMAP.get(e["category_type"], "text") for e in full], ftexts=[(e.get("text") or "") for e in full],
                fbody=full_ids)


def odb_pages():
    d = json.load(open(os.path.join(ODB, "OmniDocBench.json"), encoding="utf-8"))
    out = []
    for rec in d:
        if rec["page_info"]["page_attribute"].get("language") != "english":
            continue
        out.append(_page("odb", rec, os.path.join(ODB, "images", rec["page_info"]["image_path"])))
    return out


def drdoc_pages():
    out = []
    for f in sorted(glob.glob(os.path.join(DRD, "dev", "*", "*", "json", "*.json"))):
        try:
            recs = json.load(open(f, encoding="utf-8"))
        except ValueError:
            continue
        if not recs:
            continue
        rec = recs[0] if isinstance(recs, list) else recs
        info = rec.get("page_info", {})
        if info.get("page_attribute", {}).get("language", "english") != "english":
            continue
        docdir = os.path.dirname(os.path.dirname(f))
        img = os.path.join(docdir, "images", os.path.basename(info.get("image_path", "")))
        if not os.path.exists(img):
            img = os.path.join(docdir, "images", "page_%s.jpg" % info.get("page_no"))
        if not os.path.exists(img):
            continue
        out.append(_page("drdoc", rec, img))
    return out


def pages(ds):
    ps = odb_pages() if ds == "odb" else drdoc_pages()
    return [p for p in ps if len(p["boxes"]) >= 2]


def shard(items, k, n):
    return [x for i, x in enumerate(items) if i % n == k]


def edge(tags):
    return any(t in tags for t in EDGE_TAGS)


if __name__ == "__main__":
    import collections, sys
    for ds in sys.argv[1:] or ["odb", "drdoc"]:
        ps = pages(ds)
        print(ds, len(ps), collections.Counter(p["bucket"] for p in ps), collections.Counter(p["split"] for p in ps))
        print(" tags", collections.Counter(t for p in ps for t in p["tags"]))
        print(" layouts", collections.Counter(p["layout"] for p in ps))
        print(" sources", collections.Counter(p["source"] for p in ps))
        print(" cats", collections.Counter(c for p in ps for c in p["fcats"]))
        print(" docs", len({p["doc"] for p in ps}))
