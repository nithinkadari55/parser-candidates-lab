import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.environ.get("ODB_DIR", os.path.join(os.path.expanduser("~"), "odb"))
EXCL = {"header", "footer", "page_number", "page_footnote", "abandon", "equation_caption"}
LMAP = {"title": "section_header", "text_block": "text", "figure": "picture", "table": "table",
        "figure_caption": "caption", "table_caption": "caption", "figure_footnote": "footnote",
        "table_footnote": "footnote", "equation_isolated": "formula", "reference": "list_item",
        "code_txt": "code", "code_txt_caption": "caption", "equation_explanation": "text",
        "chart_mask": "picture", "table_mask": "table", "unknown_mask": "picture", "algorithm_mask": "code",
        "text_mask": "text", "need_mask": "text"}


def box(e):
    p = e["poly"]
    xs, ys = p[0::2], p[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def pages():
    d = json.load(open(os.path.join(D, "OmniDocBench.json"), encoding="utf-8"))
    out = []
    for p in d:
        a = p["page_info"]["page_attribute"]
        if a.get("language") != "english":
            continue
        W, H = p["page_info"]["width"], p["page_info"]["height"]
        els = [e for e in p["layout_dets"] if e.get("order") is not None and e["category_type"] not in EXCL]
        els.sort(key=lambda e: e["order"])
        boxes = [box(e) for e in els]
        cats = [e["category_type"] for e in els]
        allcats = {e["category_type"] for e in p["layout_dets"]}
        lay, src = a.get("layout"), a.get("data_source")
        has_ft = bool(allcats & {"figure", "table", "chart_mask", "table_mask"})
        if lay in ("three_column", "1andmore_column", "other_layout") or src in ("newspaper", "magazine", "PPT2PDF"):
            bucket = "complex"
        elif lay == "double_column" or has_ft:
            bucket = "medium"
        else:
            bucket = "simple"
        tags = []
        for i, b in enumerate(boxes):
            if b[2] - b[0] > 0.55 * W:
                above = any(c[2] - c[0] < 0.45 * W and c[3] <= b[1] + 2 for c in boxes)
                below = any(c[2] - c[0] < 0.45 * W and c[1] >= b[3] - 2 for c in boxes)
                ab = [c for c in boxes if c[2] - c[0] < 0.45 * W and c[3] <= b[1] + 2]
                bl = [c for c in boxes if c[2] - c[0] < 0.45 * W and c[1] >= b[3] - 2]
                if any(x[2] <= y[0] or y[2] <= x[0] for x in ab for y in ab) and any(x[2] <= y[0] or y[2] <= x[0] for x in bl for y in bl):
                    tags.append("fullwidth_interrupts_columns")
                    break
        if allcats & {"table_footnote", "figure_footnote", "page_footnote"}:
            tags.append("footnotes")
        for i, c in enumerate(cats):
            if c in ("figure_caption", "table_caption"):
                for j, c2 in enumerate(cats):
                    if c2 in ("figure", "table"):
                        b, f = boxes[i], boxes[j]
                        iv = min(b[3], f[3]) - max(b[1], f[1])
                        if iv > 0.5 * (b[3] - b[1]) and (b[0] >= f[2] - 2 or b[2] <= f[0] + 2):
                            tags.append("caption_beside_figure")
                            break
                if "caption_beside_figure" in tags:
                    break
        if src == "exam_paper":
            tags.append("exam_numbered_items")
        if W > 1.15 * H and src != "PPT2PDF":
            tags.append("landscape_or_spread")
        if src == "PPT2PDF":
            tags.append("slides")
        if src in ("newspaper", "magazine"):
            tags.append("newspaper_magazine")
        out.append(dict(split="dev" if sum(map(ord, p["page_info"]["image_path"])) % 2 == 0 else "test", img=p["page_info"]["image_path"], W=W, H=H, layout=lay, source=src, bucket=bucket,
                        tags=tags, boxes=boxes, cats=cats, labels=[LMAP.get(c, "text") for c in cats],
                        texts=[(e.get("text") or "")[:60] for e in els]))
    return out


def edge(tags):
    return any(t in tags for t in ("fullwidth_interrupts_columns", "footnotes", "caption_beside_figure", "exam_numbered_items", "landscape_or_spread"))


def shard(items, k, n):
    return [x for i, x in enumerate(items) if i % n == k]


if __name__ == "__main__":
    import collections
    ps = pages()
    print(len(ps), collections.Counter(p["bucket"] for p in ps))
    print(collections.Counter(t for p in ps for t in p["tags"]))
    print(collections.Counter(len(p["boxes"]) < 2 for p in ps))
