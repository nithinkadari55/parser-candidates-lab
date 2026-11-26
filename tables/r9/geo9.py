"""Round 9 helpers with no heavy imports (usable inside every model environment): crop turning (the harness's
tables/rotate.rs convention, as tables/r8/tatr8.py) and HTML-sequence -> grid cells."""
import re


def _corners(b, f):
    pts = [f(b[0], b[1]), f(b[2], b[3]), f(b[0], b[3]), f(b[2], b[1])]
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def turn_box(b, turn, w, h):
    f = {90: lambda x, y: (y, w - x), 180: lambda x, y: (w - x, h - y), 270: lambda x, y: (h - y, x)}.get(turn, lambda x, y: (x, y))
    return _corners(b, f)


def unturn_box(b, turn, w, h):
    f = {90: lambda x, y: (w - y, x), 180: lambda x, y: (w - x, h - y), 270: lambda x, y: (y, h - x)}.get(turn, lambda x, y: (x, y))
    return _corners(b, f)


def turn_image(im, turn):
    return im.rotate(turn, expand=True) if turn else im


def box4(b):
    """any of [x0, y0, x1, y1], 8 numbers, or [[x, y] x 4] -> [x0, y0, x1, y1]"""
    if b is None: return None
    b = list(b)
    if b and isinstance(b[0], (list, tuple)):
        xs = [p[0] for p in b]; ys = [p[1] for p in b]
    elif len(b) == 4:
        return [float(min(b[0], b[2])), float(min(b[1], b[3])), float(max(b[0], b[2])), float(max(b[1], b[3]))]
    else:
        xs, ys = b[0::2], b[1::2]
    if not xs: return None
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


SPAN = re.compile(r'(colspan|rowspan)\s*=\s*"?(\d+)')


def html_cells(tokens, boxes=None, texts=None):
    """PubTabNet-style structure tokens (<thead>, <tr>, <td>, <td, ' colspan="2"', >, </td>, ...; also whole
    '<td colspan="2"></td>' strings) and one box (and text) per cell in token order -> cells dict(r0, r1, c0, c1,
    bbox, header[, text]). Missing boxes become [0, 0, 0, 0] (the cell still counts for structure)."""
    cells = []; occ = set(); r = -1; c = 0; head = False; pend = None; k = 0
    toks = []
    for t in tokens:
        t = t.strip() if t.strip() else t
        if t.startswith("<td") and t.endswith("</td>") and t != "<td></td>":
            toks += ["<td", t[3:t.index(">")], ">", "</td>"]
        elif t.startswith("<td ") and t.endswith(">") and not t.endswith("</td>"):
            toks += ["<td", t[3:-1], ">"]
        else:
            toks.append(t)

    def place(rs, cs):
        nonlocal c, k
        while (r, c) in occ: c += 1
        for a in range(r, r + rs):
            for b in range(c, c + cs): occ.add((a, b))
        bb = boxes[k] if boxes is not None and k < len(boxes) else None
        cell = dict(r0=r, r1=r + rs, c0=c, c1=c + cs, bbox=box4(bb) or [0.0, 0.0, 0.0, 0.0], header=head)
        if texts is not None: cell["text"] = texts[k] if k < len(texts) else ""
        cells.append(cell); k += 1; c += cs
    for t in toks:
        if t == "<thead>": head = True
        elif t == "</thead>": head = False
        elif t == "<tr>": r += 1; c = 0
        elif t in ("<td>", "<td></td>", "<th>", "<th></th>"):
            if r < 0: r = 0
            place(1, 1)
        elif t in ("<td", "<th"):
            pend = {"colspan": 1, "rowspan": 1}
        elif pend is not None and t not in (">", "</td>", "</th>"):
            for m in SPAN.finditer(t): pend[m.group(1)] = max(1, int(m.group(2)))
        elif t == ">" and pend is not None:
            if r < 0: r = 0
            place(pend["rowspan"], pend["colspan"]); pend = None
    return cells


def logic_cells(boxes, logic, texts=None):
    """boxes + logic points [row_start, row_end, col_start, col_end] (inclusive) -> cells"""
    out = []
    for i, (b, lp) in enumerate(zip(boxes, logic)):
        r0, r1, c0, c1 = (int(v) for v in list(lp)[:4])
        if r1 < r0 or c1 < c0 or r0 < 0 or c0 < 0: continue
        d = dict(r0=r0, r1=r1 + 1, c0=c0, c1=c1 + 1, bbox=box4(b) or [0.0, 0.0, 0.0, 0.0], header=False)
        if texts is not None: d["text"] = texts[i] if i < len(texts) else ""
        out.append(d)
    return out
