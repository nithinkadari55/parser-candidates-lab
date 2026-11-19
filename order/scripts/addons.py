"""Add-ons around XY-Cut++ (ro.xycutpp). Each one is our own implementation of the idea it names.

fig   PDFFigures2-style grouping: a caption plus the graphic panels it describes (adjacent pictures not
      separated by body text, and short labels inside them) become one region, ordered internally.
deco  PdfPig DecorationTextBlockClassifier-style removal: blocks that repeat at the same place with the
      same text (digits ignored) on neighbouring pages of the same document, or that look like page
      numbers in the top/bottom margin, are taken out before the column cuts.
box   Burdoc-style boxed sections: drawn rectangles and shaded areas (found in the page image) that
      enclose two or more blocks become a section, ordered as one unit.
num   numbering as an order constraint: consecutive item numbers (27., 28., (3), Q5) must be read in
      increasing order; each item moves with the blocks that follow it in its column.
tiny  small side blocks (equation numbers, "(3 marks)", connecting words beside equations) are attached
      to the line they sit on; small blocks in the page margin are attached to the nearest block.
"""
import difflib, re

from ro import xycutpp, _xov

GRAPHIC = {"picture", "table", "chart"}


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def ubox(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def frac_in(b, r):
    a = area(b)
    return inter(b, r) / a if a > 0 else 0.0


# ---------------------------------------------------------------- fig
CAP_RE = re.compile(r"^\s*(fig\.?|figure|table|tab\.|chart|exhibit|plate|graph|scheme)\s*[\dIVXA-Z]", re.I)


def _gap(a, b):
    gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return gx, gy


def _between(a, b):
    return [min(a[2], b[2]) if a[2] < b[0] or b[2] < a[0] else max(a[0], b[0]),
            min(a[3], b[3]) if a[3] < b[1] or b[3] < a[1] else max(a[1], b[1]),
            max(a[0], b[0]) if a[2] < b[0] or b[2] < a[0] else min(a[2], b[2]),
            max(a[1], b[1]) if a[3] < b[1] or b[3] < a[1] else min(a[3], b[3])]


def _blocked(a, b, others, boxes):
    r = _between(a, b)
    if r[2] <= r[0] and r[3] <= r[1]:
        return False
    r = [min(r[0], r[2]) - 1, min(r[1], r[3]) - 1, max(r[0], r[2]) + 1, max(r[1], r[3]) + 1]
    return any(inter(boxes[t], r) > 0.3 * area(boxes[t]) for t in others)


def fig_groups(boxes, labels, texts, idx, W, H):
    caps = [i for i in idx if labels[i] == "caption" or (labels[i] not in GRAPHIC and CAP_RE.match(texts[i] or "") and len((texts[i] or "").split()) < 80 and labels[i] in ("text", "caption"))]
    graphics = [i for i in idx if labels[i] in GRAPHIC]
    if not caps or not graphics:
        return []
    textish = [i for i in idx if labels[i] not in GRAPHIC and i not in caps]
    used = set()
    groups = []
    for c in sorted(caps, key=lambda i: boxes[i][1]):
        cb = boxes[c]
        best, seed = None, None
        for g in graphics:
            if g in used:
                continue
            gb = boxes[g]
            gx, gy = _gap(cb, gb)
            xo = min(cb[2], gb[2]) - max(cb[0], gb[0])
            yo = min(cb[3], gb[3]) - max(cb[1], gb[1])
            if xo > 0 and gy <= 0.04 * H:
                d = gy
            elif yo > 0.5 * (cb[3] - cb[1]) and gx <= 0.04 * W:
                d = gx
            else:
                continue
            if _blocked(cb, gb, textish, boxes):
                continue
            if best is None or d < best:
                best, seed = d, g
        if seed is None:
            continue
        comp = [seed]
        if labels[seed] != "table":
            grown = True
            while grown:
                grown = False
                for g in graphics:
                    if g in used or g in comp or labels[g] == "table":
                        continue
                    for m in comp:
                        gx, gy = _gap(boxes[m], boxes[g])
                        if gx <= 0.02 * W and gy <= 0.02 * H and not _blocked(boxes[m], boxes[g], textish, boxes):
                            comp.append(g)
                            grown = True
                            break
        region = ubox([boxes[g] for g in comp])
        inside = [t for t in textish if t not in used and frac_in(boxes[t], region) >= 0.8]
        notes = [t for t in idx if labels[t] == "footnote" and t not in used and t not in inside
                 and _xov(boxes[t], region) and 0 <= boxes[t][1] - max(region[3], cb[3]) <= 0.03 * H]
        g = comp + inside + [c] + notes
        used.update(g)
        groups.append(dict(panels=comp + inside, caps=[c], notes=notes))
    return groups


def order_fig(gr, boxes, W, H):
    pan = gr["panels"]
    po = [pan[k] for k in xycutpp([boxes[i] for i in pan], W, H)] if len(pan) > 1 else list(pan)
    region = ubox([boxes[i] for i in pan])
    before = [c for c in gr["caps"] if (boxes[c][1] + boxes[c][3]) / 2 < region[1]]
    after = [c for c in gr["caps"] if c not in before]
    return before + po + after + gr["notes"]


# ---------------------------------------------------------------- box
def sections(rects, boxes, labels, idx):
    cands = []
    for r in rects:
        ins = [i for i in idx if frac_in(boxes[i], r) >= 0.8]
        cross = [i for i in idx if 0.2 < frac_in(boxes[i], r) < 0.8]
        if len([i for i in ins if labels[i] not in GRAPHIC]) >= 2 and len(ins) < 0.9 * len(idx) and not cross:
            cands.append((area(r), r, ins))
    acc, used = [], set()
    for a, r, ins in sorted(cands, key=lambda t: -t[0]):
        if set(ins) & used:
            continue
        acc.append((r, ins))
        used.update(ins)
    return acc


# ---------------------------------------------------------------- tiny
def tiny_map(boxes, labels, texts, idx, W, H):
    tiny = set()
    for t in idx:
        b = boxes[t]
        if labels[t] in GRAPHIC or labels[t] == "caption":
            continue
        txt = (texts[t] or "").strip()
        if b[2] - b[0] <= 0.15 * W and b[3] - b[1] <= 0.05 * H and len(txt.split()) <= 4:
            tiny.add(t)
    big = [i for i in idx if boxes[i][2] - boxes[i][0] > 0.3 * W]
    mx0 = min((boxes[i][0] for i in big), default=None)
    mx1 = max((boxes[i][2] for i in big), default=None)
    out = {}
    for t in tiny:
        b = boxes[t]
        h = b[3] - b[1]
        w = b[2] - b[0]
        flow = any(j != t and j not in tiny and _xov(b, boxes[j]) and min(abs(boxes[j][1] - b[3]), abs(b[1] - boxes[j][3])) <= 0.03 * H for j in idx)
        best, host = None, None
        if not flow:
            for j in idx:
                if j == t or j in tiny:
                    continue
                bj = boxes[j]
                if bj[2] - bj[0] < 2 * w:
                    continue
                yo = min(b[3], bj[3]) - max(b[1], bj[1])
                if yo < 0.5 * h:
                    continue
                if b[2] <= bj[0] + 1:
                    g = bj[0] - b[2]
                elif bj[2] <= b[0] + 1:
                    g = b[0] - bj[2]
                else:
                    continue
                if g <= 0.3 * W and (best is None or g < best):
                    best, host = g, j
        if host is not None:
            out[t] = (host, boxes[t][2] <= boxes[host][0] + 1)
            continue
        if big and (b[2] < mx0 or b[0] > mx1):
            cand = [j for j in idx if j not in tiny]
            if cand:
                cy = (b[1] + b[3]) / 2
                j = min(cand, key=lambda j: abs((boxes[j][1] + boxes[j][3]) / 2 - cy))
                out[t] = (j, cy < (boxes[j][1] + boxes[j][3]) / 2)
    return out


# ---------------------------------------------------------------- num
NUM_RE = re.compile(r"^\s*(?:q(?:uestion)?\s*)?\(?(\d{1,3})\s*[\.\):]\s*\S", re.I)


def numbering(seq, boxes, texts):
    num = {}
    for i in seq:
        m = NUM_RE.match(texts[i] or "")
        if m:
            num[i] = int(m.group(1))
    if len(num) < 3:
        return seq
    vals = list(num.values())
    if len(set(vals)) != len(vals) or max(vals) - min(vals) + 1 != len(vals):
        return seq
    if [num[i] for i in seq if i in num] == sorted(vals):
        return seq
    heads = [k for k, i in enumerate(seq) if i in num]
    groups = []
    for a, k in enumerate(heads):
        end = heads[a + 1] if a + 1 < len(heads) else len(seq)
        h = seq[k]
        g = [h]
        for m in range(k + 1, end):
            j = seq[m]
            if boxes[j][1] >= boxes[h][1] - 2 and _xov(boxes[j], boxes[h]):
                g.append(j)
            else:
                break
        groups.append(g)
    member = {x for g in groups for x in g}
    sg = sorted(groups, key=lambda g: num[g[0]])
    out, gi = [], 0
    for i in seq:
        if i in member:
            if i in num:
                out += sg[gi]
                gi += 1
        else:
            out.append(i)
    return out


# ---------------------------------------------------------------- deco
PAGE_NO_RE = re.compile(r"^\W*(page\s*)?[\divxlc]{1,5}(\s*(of|/)\s*\d+)?\W*$", re.I)


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", (t or "").lower())).strip()


def decorations(pages):
    """pages: list of dicts with W, H, fboxes, flabels, ftexts, doc, page_no, id. Returns {id: set(full indices)}."""
    bydoc = {}
    for p in pages:
        bydoc.setdefault(p["doc"], []).append(p)
    res = {}
    for doc, ps in bydoc.items():
        ps.sort(key=lambda p: p["page_no"])
        for k, p in enumerate(ps):
            W, H = p["W"], p["H"]
            marks = set()
            nb = [q for q in ps[max(0, k - 2):k] + ps[k + 1:k + 3]]
            for i, b in enumerate(p["fboxes"]):
                if p["flabels"][i] in GRAPHIC:
                    continue
                margin = b[3] <= 0.15 * H or b[1] >= 0.85 * H or b[2] <= 0.12 * W or b[0] >= 0.88 * W
                if not margin:
                    continue
                t = p["ftexts"][i]
                if t and PAGE_NO_RE.match(t.strip()) and (b[3] <= 0.12 * H or b[1] >= 0.88 * H):
                    marks.add(i)
                    continue
                nt = _norm(t)
                if not nt:
                    continue
                cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
                for q in nb:
                    hit = False
                    for j, c in enumerate(q["fboxes"]):
                        qx, qy = (c[0] + c[2]) / 2 / q["W"], (c[1] + c[3]) / 2 / q["H"]
                        if abs(qx - cx) > 0.05 or abs(qy - cy) > 0.03:
                            continue
                        if difflib.SequenceMatcher(None, nt, _norm(q["ftexts"][j])).ratio() >= 0.8:
                            hit = True
                            break
                    if hit:
                        marks.add(i)
                        break
            res[p["id"]] = marks
    return res


# ---------------------------------------------------------------- driver
def order(boxes, labels, texts, W, H, rects=(), addons=frozenset(), deco=frozenset()):
    n = len(boxes)
    work = [i for i in range(n) if i not in deco] if "deco" in addons else list(range(n))
    tmap = tiny_map(boxes, labels, texts, work, W, H) if "tiny" in addons else {}
    work = [i for i in work if i not in tmap]

    def solve(idx, rs):
        units = []
        rest = list(idx)
        if "box" in addons and rs:
            for r, mem in sections(rs, boxes, labels, rest):
                inner = solve(mem, [q for q in rs if area(q) < area(r) and frac_in(q, r) > 0.95])
                units.append((ubox([boxes[i] for i in mem] + [r]), inner))
                ms = set(mem)
                rest = [i for i in rest if i not in ms]
        if "fig" in addons:
            for gr in fig_groups(boxes, labels, texts, rest, W, H):
                o = order_fig(gr, boxes, W, H)
                units.append((ubox([boxes[i] for i in o]), o))
                ms = set(o)
                rest = [i for i in rest if i not in ms]
        for i in rest:
            units.append((boxes[i], [i]))
        if not units:
            return []
        o = xycutpp([u[0] for u in units], W, H)
        return [i for k in o for i in units[k][1]]

    seq = solve(work, list(rects))
    if tmap:
        out = []
        for i in seq:
            before = sorted([t for t, (h, b) in tmap.items() if h == i and b], key=lambda t: (boxes[t][1], boxes[t][0]))
            after = sorted([t for t, (h, b) in tmap.items() if h == i and not b], key=lambda t: (boxes[t][1], boxes[t][0]))
            out += before + [i] + after
        seq = out
    if "num" in addons:
        seq = numbering(seq, boxes, texts)
    if "deco" in addons and deco:
        top = sorted([i for i in deco if (boxes[i][1] + boxes[i][3]) / 2 < H / 2], key=lambda i: (boxes[i][1], boxes[i][0]))
        bot = sorted([i for i in deco if i not in top], key=lambda i: (boxes[i][1], boxes[i][0]))
        seq = top + seq + bot
    assert sorted(seq) == list(range(n)), (len(seq), n)
    return seq


# ---------------------------------------------------------------- rects from the page image
def detect_rects(path, W, H):
    import cv2
    import numpy as np
    img = cv2.imread(path)
    if img is None:
        return []
    ih, iw = img.shape[:2]
    sx, sy = iw / W, ih / H
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = []
    bw = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, iw // 10), 1)))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, ih // 25))))
    grid = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.12 * iw or h < 0.04 * ih or (w > 0.97 * iw and h > 0.97 * ih):
            continue
        top = hl[max(0, y - 3):y + 4, x:x + w].max(axis=0).mean() / 255 if w else 0
        bot = hl[max(0, y + h - 4):y + h + 3, x:x + w].max(axis=0).mean() / 255 if w else 0
        lef = vl[y:y + h, max(0, x - 3):x + 4].max(axis=1).mean() / 255 if h else 0
        rig = vl[y:y + h, max(0, x + w - 4):x + w + 3].max(axis=1).mean() / 255 if h else 0
        if min(top, bot, lef, rig) >= 0.7:
            rects.append([x, y, x + w, y + h])
    med = cv2.medianBlur(img, 15)
    small = cv2.resize(med, (max(1, iw // 4), max(1, ih // 4)), interpolation=cv2.INTER_AREA)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(small.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = (diff > 18).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    sa = small.shape[0] * small.shape[1]
    for k in range(1, nlab):
        x, y, w, h, a = st[k]
        if a >= 0.02 * sa and a >= 0.7 * w * h and not (w > 0.97 * small.shape[1] and h > 0.97 * small.shape[0]):
            rects.append([x * 4, y * 4, (x + w) * 4, (y + h) * 4])
    out = []
    for r in rects:
        r = [r[0] / sx, r[1] / sy, r[2] / sx, r[3] / sy]
        if not any(inter(r, q) > 0.8 * max(area(r), area(q)) for q in out):
            out.append(r)
    return out
