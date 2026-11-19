import os
import statistics
import sys
from scipy.stats import kendalltau

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")


def _groups(boxes, idx, axis, tol):
    a0, a1 = (0, 2) if axis == "x" else (1, 3)
    iv = sorted(((boxes[i][a0] + tol, boxes[i][a1] - tol, i) for i in idx), key=lambda t: t[0])
    groups, cur, end = [], [], None
    for s, e, i in iv:
        if e < s:
            m = (s + e) / 2
            s = e = m
        if cur and s > end:
            groups.append(cur)
            cur, end = [], None
        cur.append(i)
        end = e if end is None else max(end, e)
    if cur:
        groups.append(cur)
    return groups


def _fallback(boxes, idx):
    return sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0]))


def xycut(boxes, W, H, first="y"):
    tx, ty = 0.002 * W, 0.002 * H
    tol = {"x": tx, "y": ty}
    other = {"x": "y", "y": "x"}

    def rec(idx, axis, depth=0):
        if len(idx) <= 1 or depth > 200:
            return list(idx)
        g = _groups(boxes, idx, axis, tol[axis])
        if len(g) > 1:
            out = []
            for gg in g:
                out += rec(gg, other[axis], depth + 1)
            return out
        g = _groups(boxes, idx, other[axis], tol[other[axis]])
        if len(g) > 1:
            out = []
            for gg in g:
                out += rec(gg, axis, depth + 1)
            return out
        return _fallback(boxes, idx)

    return rec(list(range(len(boxes))), first)


def _xov(a, b, t=0.0):
    return min(a[2], b[2]) - max(a[0], b[0]) > t


def _yov(a, b, t=0.0):
    return min(a[3], b[3]) - max(a[1], b[1]) > t


def _viou(a, b):
    i = min(a[3], b[3]) - max(a[1], b[1])
    if i <= 0:
        return 0.0
    u = max(a[3], b[3]) - min(a[1], b[1])
    return i / u if u > 0 else 0.0


def xycutpp(boxes, W, H, row_rule=False, stack_rule=True, beta=1.3):
    tx, ty = 0.002 * W, 0.002 * H

    def rowlike(cols):
        if len(cols) < 2:
            return False
        if any(len(c) < 2 for c in cols):
            return False
        allb = [i for c in cols for i in c]
        colof = {i: k for k, c in enumerate(cols) for i in c}
        hit = 0
        for i in allb:
            if any(colof[j] != colof[i] and _viou(boxes[i], boxes[j]) > 0.7 for j in allb):
                hit += 1
        return hit >= 0.6 * len(allb)

    def rec(idx, depth=0):
        if len(idx) <= 1 or depth > 200:
            return list(idx)
        cols = _groups(boxes, idx, "x", tx)
        stacked = False
        if len(cols) > 1 and stack_rule:
            rows0 = _groups(boxes, idx, "y", ty)
            if len(rows0) > 1:
                colof = {i: k for k, c in enumerate(cols) for i in c}
                stacked = all(len({colof[i] for i in r}) == 1 for r in rows0)
        if len(cols) > 1 and not stacked and not (row_rule and rowlike(cols)):
            out = []
            for c in cols:
                out += rec(c, depth + 1)
            return out
        widths = [boxes[i][2] - boxes[i][0] for i in idx]
        med = statistics.median(widths)
        cand = [i for i in idx if not any(j != i and _yov(boxes[i], boxes[j], ty) and not _xov(boxes[i], boxes[j], tx) for j in idx)]
        cset = set(cand)
        rest = [i for i in idx if i not in cset]
        rg = _groups(boxes, rest, "x", tx) if rest else []
        if len(rg) >= 2:
            seps = [c for c in cand if sum(1 for g in rg if any(_xov(boxes[c], boxes[j], tx) for j in g)) >= 2]
        else:
            seps = cand
        if seps:
            sset = set(seps)
            rest = [i for i in idx if i not in sset]
            ss = sorted(seps, key=lambda i: boxes[i][1])
            bands = [[] for _ in range(len(ss) + 1)]
            for i in rest:
                cy = (boxes[i][1] + boxes[i][3]) / 2
                k = sum(1 for s in ss if (boxes[s][1] + boxes[s][3]) / 2 < cy)
                bands[k].append(i)
            if len(ss) + sum(1 for b in bands if b) > 1:
                out = []
                for k, s in enumerate(ss):
                    out += rec(bands[k], depth + 1)
                    out.append(s)
                out += rec(bands[-1], depth + 1)
                return out
        rows = _groups(boxes, idx, "y", ty)
        if len(rows) > 1:
            out = []
            for r in rows:
                out += rec(r, depth + 1)
            return out
        if len(cols) > 1:
            out = []
            for c in cols:
                out += rec(c, depth + 1)
            return out
        return _fallback(boxes, idx)

    return rec(list(range(len(boxes))))


_DL = None


def docling_rb(boxes, labels, W, H):
    global _DL
    if _DL is None:
        sys.path.insert(0, VENDOR)
        import docling_ro
        from docling_core.types.doc.base import CoordOrigin, Size
        from docling_core.types.doc.labels import DocItemLabel
        _DL = (docling_ro, CoordOrigin, Size, DocItemLabel)
    m, CoordOrigin, Size, DocItemLabel = _DL
    order0 = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    cid = {i: k for k, i in enumerate(order0)}
    els = []
    for i, b in enumerate(boxes):
        lab = labels[i] if labels else "text"
        try:
            dl = DocItemLabel(lab)
        except ValueError:
            dl = DocItemLabel.TEXT
        els.append(m.PageElement(cid=cid[i], text=str(i), page_no=1, page_size=Size(width=W, height=H), label=dl,
                                 l=b[0], t=b[1], r=b[2], b=b[3], coord_origin=CoordOrigin.TOPLEFT))
    ro = m.ReadingOrderPredictor()
    res = ro.predict_reading_order(page_elements=els)
    return [int(e.text) for e in res]


def lev(a, b):
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def score(pred, gt):
    pos = {e: k for k, e in enumerate(pred)}
    gr = list(range(len(gt)))
    pr = [pos[e] for e in gt]
    if len(gt) >= 2:
        t = kendalltau(gr, pr).statistic
    else:
        t = 1.0
    ed = lev(pred, gt) / max(1, len(gt))
    return dict(tau=float(t), ed=float(ed), exact=int(list(pred) == list(gt)), n=len(gt))


def xycutpp_lab(boxes, labels, W, H, **kw):
    n = len(boxes)
    graphic = {"picture", "table", "chart"}
    attach = {"caption", "footnote"}
    owner = {}
    for c in range(n):
        if labels[c] not in attach:
            continue
        cb = boxes[c]; best, bf = None, None
        for f in range(n):
            if labels[f] not in graphic:
                continue
            fb = boxes[f]
            xo = min(cb[2], fb[2]) - max(cb[0], fb[0]); yo = min(cb[3], fb[3]) - max(cb[1], fb[1])
            vg = max(fb[1] - cb[3], cb[1] - fb[3]); hg = max(fb[0] - cb[2], cb[0] - fb[2])
            d = None
            if xo > 0.5 * (cb[2] - cb[0]) and vg < 0.03 * H:
                d = max(0, vg)
            elif yo > 0.5 * (cb[3] - cb[1]) and hg < 0.03 * W:
                d = max(0, hg)
            if d is not None and (best is None or d < best):
                best, bf = d, f
        if bf is not None:
            owner[c] = bf
    groups = {}
    for i in range(n):
        groups.setdefault(owner.get(i, i), []).append(i)
    keys = list(groups)
    ub = []
    for k in keys:
        g = groups[k]
        ub.append([min(boxes[i][0] for i in g), min(boxes[i][1] for i in g), max(boxes[i][2] for i in g), max(boxes[i][3] for i in g)])
    o = xycutpp(ub, W, H, **kw)
    out = []
    for u in o:
        out += sorted(groups[keys[u]], key=lambda i: (boxes[i][1], boxes[i][0]))
    return out
