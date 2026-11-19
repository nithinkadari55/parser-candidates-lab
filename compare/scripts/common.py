import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
ODB = os.environ.get("ODB_DIR", os.path.join(HOME, "odb"))
CORPUS = os.environ.get("CORPUS_DIR", os.path.join(HOME, "corpus"))
DLRS_MODELS = os.environ.get("DLRS_MODELS", os.path.join(HOME, "dlrs"))
ODB_REPO, ODB_REV = "opendatalab/OmniDocBench", "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
DOCLINGRS_REV = "cfb8df247f555b61251f36925a6a6312c19a6513"
PER_BUCKET = int(os.environ.get("ODB_PER_BUCKET", "80"))
EDGE_ISSUES = {"fuzzy_scan", "watermark", "with_watermark", "colorful_backgroud", "geometric_deformation",
               "handwriting", "text_L-shaped_wrap", "transparent_pages"}
COMPLEX_LAYOUTS = {"three_column", "1andmore_column", "other_layout"}
COMPLEX_SOURCES = {"newspaper", "magazine", "PPT2PDF"}


def h(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def bucket_of(p):
    a = p["page_info"]["page_attribute"]
    cats = {e["category_type"] for e in p["layout_dets"]}
    W, H = p["page_info"]["width"], p["page_info"]["height"]
    issues = set(a.get("special_issue") or [])
    if issues & EDGE_ISSUES or a.get("subset") == "layout_hard" or (W > 1.15 * H and a.get("data_source") != "PPT2PDF"):
        return "edge"
    if a.get("layout") in COMPLEX_LAYOUTS or a.get("data_source") in COMPLEX_SOURCES:
        return "complex"
    if a.get("layout") == "single_column" and not cats & {"table", "table_mask"} and a.get("subset") == "v1.5":
        return "simple"
    return "medium"


def load_gt():
    return json.load(open(os.path.join(ODB, "OmniDocBench.json"), encoding="utf-8"))


def select(gt=None):
    gt = gt if gt is not None else load_gt()
    pools = {}
    for p in gt:
        if p["page_info"]["page_attribute"].get("language") != "english":
            continue
        cats = {e["category_type"] for e in p["layout_dets"]}
        b = bucket_of(p)
        kind = "table" if "table" in cats else "formula" if "equation_isolated" in cats else "other"
        pools.setdefault(b, {}).setdefault(kind, []).append(p["page_info"]["image_path"])
    out = []
    for b in ("simple", "medium", "complex", "edge"):
        subs = [sorted(pools.get(b, {}).get(k, []), key=h) for k in ("table", "formula", "other")]
        picked, i = [], 0
        while len(picked) < PER_BUCKET and any(i < len(s) for s in subs):
            for s in subs:
                if i < len(s) and len(picked) < PER_BUCKET:
                    picked.append(s[i])
            i += 1
        out += [dict(img=n, bucket=b) for n in picked]
    return out


def selection_index(gt=None):
    gt = gt if gt is not None else load_gt()
    sel = select(gt)
    by = {p["page_info"]["image_path"]: p for p in gt}
    for s in sel:
        cats = {e["category_type"] for e in by[s["img"]]["layout_dets"]}
        s["has_table"] = "table" in cats
        s["has_formula"] = "equation_isolated" in cats
    return sel


def filtered_gt(path_out, only=None):
    gt = load_gt()
    sel = selection_index(gt)
    keep = {s["img"]: s["bucket"] for s in sel if only is None or s["img"] in only}
    out = []
    for p in gt:
        n = p["page_info"]["image_path"]
        if n in keep:
            q = json.loads(json.dumps(p))
            q["page_info"]["page_attribute"]["bucket"] = keep[n]
            out.append(q)
    json.dump(out, open(path_out, "w", encoding="utf-8"), ensure_ascii=False)
    return len(out)


def shard(items, k, n):
    return [x for i, x in enumerate(items) if i % n == k]


def corpus_pdfs():
    out = []
    for sub in ("pdf", "scanned"):
        d = os.path.join(CORPUS, sub)
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".pdf")]
    return out


def page_count(path):
    import pypdfium2 as pdfium
    d = pdfium.PdfDocument(path)
    try:
        return len(d)
    finally:
        d.close()


def balanced(paths, k, n):
    sizes = sorted(((page_count(p), p) for p in paths), key=lambda x: (-x[0], x[1]))
    bins = [[0, []] for _ in range(n)]
    for c, p in sizes:
        b = min(bins, key=lambda x: x[0])
        b[0] += c
        b[1].append(p)
    return sorted(bins[k][1])


def peak_rss_mb():
    try:
        import resource
    except ImportError:
        return None
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(s, c) / 1024.0, 1)


def pdf_words(path):
    import pypdfium2 as pdfium
    d = pdfium.PdfDocument(path)
    try:
        words = []
        for i in range(len(d)):
            pg = d[i]
            tp = pg.get_textpage()
            words += norm_words(tp.get_text_range())
            tp.close()
            pg.close()
        return words
    finally:
        d.close()


def norm_words(s):
    return re.findall(r"[0-9a-z]+", (s or "").lower())


def bag_f1(pred, ref):
    from collections import Counter
    a, b = Counter(pred), Counter(ref)
    inter = sum((a & b).values())
    if not a or not b:
        return 0.0
    p, r = inter / sum(a.values()), inter / sum(b.values())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def strip_html(s):
    return re.sub(r"<[^>]+>", " ", s or "")


def write_jsonl(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()


if __name__ == "__main__":
    import collections
    sel = selection_index()
    print(len(sel), collections.Counter(s["bucket"] for s in sel))
    print(collections.Counter((s["bucket"], s["has_table"], s["has_formula"]) for s in sel))
    sys.exit(0)
