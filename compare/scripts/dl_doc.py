import html
from collections import Counter

import common as C

PIC = ("picture", "chart")


def res(doc, ref):
    _, kind, idx = ref.split("/")
    return doc[kind][int(idx)]


def table_html(t):
    cells = (t.get("data") or {}).get("table_cells") or []
    rows = {}
    for c in cells:
        rows.setdefault(c.get("start_row_offset_idx", 0), []).append(c)
    out = ["<table>"]
    for r in sorted(rows):
        out.append("<tr>")
        for c in sorted(rows[r], key=lambda c: c.get("start_col_offset_idx", 0)):
            tag = "th" if c.get("column_header") else "td"
            rs = c.get("end_row_offset_idx", r + 1) - c.get("start_row_offset_idx", r)
            cs = c.get("end_col_offset_idx", 1) - c.get("start_col_offset_idx", 0)
            a = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            out.append(f"<{tag}{a}>{html.escape(c.get('text') or '')}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def table_text(t):
    cells = (t.get("data") or {}).get("table_cells") or []
    cells = sorted(cells, key=lambda c: (c.get("start_row_offset_idx", 0), c.get("start_col_offset_idx", 0)))
    return " ".join(c.get("text") or "" for c in cells)


def walk(doc, fn, node=None, inside=None, depth=0):
    node = node if node is not None else doc["body"]
    for ch in node.get("children", []):
        it = res(doc, ch["$ref"])
        lab = it.get("label")
        if inside and lab not in ("caption", "footnote"):
            continue
        fn(it, depth)
        walk(doc, fn, it, inside or (lab if lab in PIC + ("table",) else None), depth + 1)


def to_markdown(doc):
    parts = []

    def fn(it, depth):
        lab = it.get("label")
        t = (it.get("text") or "").strip()
        if lab == "title":
            parts.append("# " + t)
        elif lab == "section_header":
            parts.append("#" * (int(it.get("level") or 1) + 1) + " " + t)
        elif lab == "table":
            parts.append(table_html(it))
        elif lab == "formula":
            if t:
                parts.append("$$\n" + t + "\n$$")
        elif lab == "list_item":
            parts.append("- " + t)
        elif lab in PIC or it.get("self_ref", "").startswith("#/groups"):
            return
        elif t:
            parts.append(t)
    walk(doc, fn)
    return "\n\n".join(parts) + "\n"


def page_text(doc, page):
    parts = []

    def fn(it, depth):
        pr = it.get("prov") or []
        if not pr or pr[0].get("page_no") != page:
            return
        if it.get("label") == "table":
            parts.append(table_text(it))
        elif it.get("text"):
            parts.append(it["text"])
    walk(doc, fn)
    return " ".join(parts)


def features(doc):
    texts, tables, pics, groups = doc.get("texts", []), doc.get("tables", []), doc.get("pictures", []), doc.get("groups", [])
    lab = Counter((t.get("label"), t.get("content_layer", "body")) for t in texts)
    levels = Counter(int(t.get("level") or 1) for t in texts if t.get("label") == "section_header")
    items = texts + tables + pics
    prov = sum(1 for t in items if t.get("prov") and all("bbox" in p and "page_no" in p for p in t["prov"]))
    cells = [c for t in tables for c in (t.get("data") or {}).get("table_cells") or []]
    span_tables = sum(1 for t in tables if any(c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1 for c in (t.get("data") or {}).get("table_cells") or []))
    hdr_tables = sum(1 for t in tables if any(c.get("column_header") for c in (t.get("data") or {}).get("table_cells") or []))
    rowhdr_tables = sum(1 for t in tables if any(c.get("row_header") for c in (t.get("data") or {}).get("table_cells") or []))
    lists = [g for g in groups if g.get("label") in ("list", "ordered_list")]
    nested = sum(1 for g in lists if (g.get("parent") or {}).get("$ref", "").startswith(("#/texts", "#/groups")) and
                 res(doc, g["parent"]["$ref"]).get("label") in ("list_item", "list", "ordered_list"))
    classified = sum(1 for p in pics if any(a.get("kind") == "classification" for a in p.get("annotations") or []) or
                     ((p.get("meta") or {}).get("classification")))
    captioned = sum(1 for p in pics if p.get("captions"))
    formulas = [t for t in texts if t.get("label") == "formula"]
    words = []
    for t in texts:
        words += C.norm_words(t.get("text"))
    for t in tables:
        words += C.norm_words(table_text(t))
    return dict(
        n_texts=len(texts), labels={f"{a}|{b}": n for (a, b), n in lab.items()},
        titles=lab[("title", "body")], headings=sum(levels.values()), heading_levels=dict(sorted(levels.items())),
        list_items=sum(n for (a, _), n in lab.items() if a == "list_item"), list_groups=len(lists), nested_lists=nested,
        enumerated_items=sum(1 for t in texts if t.get("label") == "list_item" and t.get("enumerated")),
        tables=len(tables), table_cells=len(cells), tables_with_spans=span_tables, tables_with_col_header=hdr_tables,
        tables_with_row_header=rowhdr_tables, pictures=len(pics), pictures_captioned=captioned, pictures_classified=classified,
        footnotes=sum(n for (a, _), n in lab.items() if a == "footnote"),
        page_headers=sum(n for (a, l), n in lab.items() if a == "page_header"), page_footers=sum(n for (a, l), n in lab.items() if a == "page_footer"),
        furniture_items=sum(n for (a, l), n in lab.items() if l == "furniture"),
        formulas=len(formulas), formulas_with_text=sum(1 for t in formulas if (t.get("text") or "").strip()),
        items_with_prov=prov, items_total=len(items), pages=len(doc.get("pages", {}))), words


def strip_images(doc):
    for p in doc.get("pictures", []):
        p.pop("image", None)
    for pg in (doc.get("pages") or {}).values():
        pg.pop("image", None)
    return doc
