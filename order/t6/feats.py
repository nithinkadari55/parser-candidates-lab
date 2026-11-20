"""Test 6: pairwise reading-order re-ranker (kraken ROMLP idea, our own code), XY-Cut++ with an
uncertainty tree, and a port of PdfPig's UnsupervisedReadingOrderDetector.

kraken (Apache-2.0) reference: kraken/lib/ro/layers.py (ROMLP: Linear(F, 2F) -> ReLU -> Linear(2F, 1)),
kraken/lib/dataset/ro.py (per element: class one-hot, centre, box min, box max; pair = concat(i, j),
target 1 if i precedes j), kraken/lib/segmentation.py::_greedy_order_decoder.
PdfPig (Apache-2.0) reference: ReadingOrderDetector/UnsupervisedReadingOrderDetector.cs and
IntervalRelationHelper.cs (ColumnWise rule, T = 5 points, no rendering order: image pages have none).
"""
import math, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import ro  # noqa: E402

CLASSES = ["text", "section_header", "picture", "table", "caption", "footnote", "formula", "list_item", "code",
           "page_header", "page_footer", "formula_number", "other"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
NUM_RE = re.compile(r"^\s*(?:q(?:uestion)?\s*)?\(?(\d{1,3})\s*[\.\):]\s*\S", re.I)
N_EXTRA_E = 7
N_EXTRA_P = 7


def canon(boxes):
    """Canonical input order (top, then left) so no method can see the ground-truth order."""
    return sorted(range(len(boxes)), key=lambda i: (round(boxes[i][1], 1), round(boxes[i][0], 1), i))


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def frac_in(b, r):
    a = area(b)
    if a <= 0:
        return 0.0
    i = max(0.0, min(b[2], r[2]) - max(b[0], r[0])) * max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
    return i / a


def col_index(boxes, W):
    xs = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    col = {}
    k = 0
    last = None
    for i in xs:
        if last is not None and boxes[i][0] - last > 0.05 * W:
            k += 1
        col[i] = k
        last = boxes[i][0]
    return [col[i] / max(1, k) for i in range(len(boxes))], [col[i] for i in range(len(boxes))]


def element_features(boxes, labels, texts, W, H, rects=(), xy_rank=None, extras=True):
    n = len(boxes)
    C = len(CLASSES)
    E = np.zeros((n, C + 6 + (N_EXTRA_E if extras else 0)), np.float32)
    for i, b in enumerate(boxes):
        E[i, CIDX.get(labels[i], CIDX["other"])] = 1
        E[i, C:C + 6] = [(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, b[0] / W, b[1] / H, b[2] / W, b[3] / H]
    info = dict(num=[None] * n, rect=[-1] * n)
    if not extras:
        return E, info
    colf, _ = col_index(boxes, W)
    est = []
    for i, b in enumerate(boxes):
        t = (texts[i] if texts else "") or ""
        m = NUM_RE.match(t)
        info["num"][i] = int(m.group(1)) if m else None
        chars = len(t.strip())
        est.append(math.sqrt(area(b) / chars) / H if chars >= 8 and labels[i] not in ("picture", "table") else None)
        for k, r in enumerate(rects):
            if frac_in(b, r) >= 0.8:
                info["rect"][i] = k
                break
    known = sorted(e for e in est if e)
    med = known[len(known) // 2] if known else None
    for i in range(n):
        o = C + 6
        E[i, o] = (xy_rank[i] / max(1, n - 1)) if xy_rank is not None else 0
        E[i, o + 1] = colf[i]
        E[i, o + 2] = 1.0 if info["num"][i] is not None else 0.0
        E[i, o + 3] = min(1.0, (info["num"][i] or 0) / 100.0)
        E[i, o + 4] = min(4.0, est[i] / med) / 4.0 if (est[i] and med) else 0.0
        E[i, o + 5] = 1.0 if est[i] else 0.0
        E[i, o + 6] = 1.0 if info["rect"][i] >= 0 else 0.0
    return E, info


def pair_features(E, boxes, W, H, info, xy_rank=None, extras=True, pairs=None):
    n = len(boxes)
    if pairs is None:
        ii, jj = np.where(~np.eye(n, dtype=bool))
    else:
        ii, jj = pairs
    X = [E[ii], E[jj]]
    if extras:
        b = np.asarray(boxes, np.float32)
        cx = (b[:, 0] + b[:, 2]) / 2 / W
        cy = (b[:, 1] + b[:, 3]) / 2 / H
        num = np.array([np.nan if v is None else v for v in info["num"]], np.float32)
        rect = np.asarray(info["rect"])
        both = ~np.isnan(num[ii]) & ~np.isnan(num[jj])
        ns = np.where(both, np.sign(np.nan_to_num(num[ii]) - np.nan_to_num(num[jj])), 0)
        same = ((rect[ii] >= 0) & (rect[ii] == rect[jj])).astype(np.float32)
        r = np.asarray(xy_rank if xy_rank is not None else range(n), np.float32)
        xyb = np.where(r[ii] < r[jj], 1.0, -1.0)
        xo = np.maximum(0, np.minimum(b[ii, 2], b[jj, 2]) - np.maximum(b[ii, 0], b[jj, 0])) / np.maximum(1, np.minimum(b[ii, 2] - b[ii, 0], b[jj, 2] - b[jj, 0]))
        yo = np.maximum(0, np.minimum(b[ii, 3], b[jj, 3]) - np.maximum(b[ii, 1], b[jj, 1])) / np.maximum(1, np.minimum(b[ii, 3] - b[ii, 1], b[jj, 3] - b[jj, 1]))
        X.append(np.stack([ns, same, xyb, cx[jj] - cx[ii], cy[jj] - cy[ii], xo, yo], 1).astype(np.float32))
    return np.concatenate(X, 1), ii, jj


class MLP:
    """kraken ROMLP shape: Linear(F, 2F) -> ReLU -> Linear(2F, 1). Weights from train6.py (npz)."""

    def __init__(self, path):
        z = np.load(path)
        self.w1, self.b1, self.w2, self.b2 = z["w1"], z["b1"], z["w2"], z["b2"]
        self.mu, self.sd = z["mu"], z["sd"]
        self.extras = bool(z["extras"])
        self.params = int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    def prob(self, X):
        X = (X - self.mu) / self.sd
        h = np.maximum(0, X @ self.w1.T + self.b1)
        z = h @ self.w2.T + self.b2
        return 1 / (1 + np.exp(-z[:, 0]))

    def matrix(self, boxes, labels, texts, W, H, rects, xy_rank):
        n = len(boxes)
        E, info = element_features(boxes, labels, texts, W, H, rects, xy_rank, self.extras)
        X, ii, jj = pair_features(E, boxes, W, H, info, xy_rank, self.extras)
        P = np.full((n, n), 0.5, np.float32)
        if len(ii):
            P[ii, jj] = self.prob(X)
        return P


def greedy_decode(P):
    """kraken _greedy_order_decoder, numpy port."""
    n = P.shape[0]
    if n <= 1:
        return list(range(n))
    eps = np.finfo(np.float32).eps
    A = P.astype(np.float64) + eps
    A = (A + (1 - A).T) / 2
    np.fill_diagonal(A, eps)
    lP = np.log(A)
    np.fill_diagonal(lP, 0)
    path = []
    used = np.zeros(n, bool)
    for _ in range(n):
        s = lP.sum(axis=1)
        s[used] = -np.inf
        idx = int(np.argmax(s))
        path.append(idx)
        used[idx] = True
        lP[idx, :] = lP[:, idx]
        lP[:, idx] = 0
    return path


# --------------------------------------------------------------------------------------------
def xycutpp_tree(boxes, W, H, rerank=None, stack_rule=True):
    """ro.xycutpp (row_rule off) re-written to mark uncertain decisions. With rerank=None the output is
    identical to ro.xycutpp. rerank(children) -> reordered children, applied only at uncertain nodes:
      - vertical cut whose columns are very unequal (narrow side < 0.6 x wide side: sidebar-like) or
        grid-like (most blocks have a same-row partner in another column: numbered grids);
      - horizontal row cut inside a region that has two or more x-groups (columns joined by a block);
      - the no-cut fallback (overlapping boxes)."""
    tx, ty = 0.002 * W, 0.002 * H
    _groups, _xov, _yov, _viou = ro._groups, ro._xov, ro._yov, ro._viou
    stats = {"uncertain_nodes": 0, "nodes": 0}

    def rowlike(cols):
        if len(cols) < 2 or any(len(c) < 2 for c in cols):
            return False
        allb = [i for c in cols for i in c]
        colof = {i: k for k, c in enumerate(cols) for i in c}
        hit = sum(1 for i in allb if any(colof[j] != colof[i] and _viou(boxes[i], boxes[j]) > 0.7 for j in allb))
        return hit >= 0.6 * len(allb)

    def width(c):
        return max(boxes[i][2] for i in c) - min(boxes[i][0] for i in c)

    def combine(children, unc):
        stats["nodes"] += 1
        if unc:
            stats["uncertain_nodes"] += 1
            if rerank is not None and len(children) > 1:
                children = rerank(children)
        return [i for c in children for i in c]

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
        if len(cols) > 1 and not stacked:
            ws = [width(c) for c in cols]
            unc = min(ws) < 0.6 * max(ws) or rowlike(cols)
            return combine([rec(c, depth + 1) for c in cols], unc)
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
                ch = []
                for k, s in enumerate(ss):
                    ch.append(rec(bands[k], depth + 1))
                    ch.append([s])
                ch.append(rec(bands[-1], depth + 1))
                return combine([c for c in ch if c], False)
        rows = _groups(boxes, idx, "y", ty)
        if len(rows) > 1:
            return combine([rec(r, depth + 1) for r in rows], len(cols) > 1)
        if len(cols) > 1:
            return combine([rec(c, depth + 1) for c in cols], True)
        return combine([[i] for i in ro._fallback(boxes, idx)], True)

    return rec(list(range(len(boxes)))), stats


def child_reranker(P):
    """Order child segments by the mean pairwise probability (greedy, kraken decoder on segments)."""
    def rr(children):
        k = len(children)
        Q = np.full((k, k), 0.5)
        for a in range(k):
            for b in range(k):
                if a != b:
                    Q[a, b] = float(P[np.ix_(children[a], children[b])].mean())
        o = greedy_decode(Q)
        return [children[i] for i in o]
    return rr


# --------------------------------------------------------------------------------------------
# PdfPig UnsupervisedReadingOrderDetector (ColumnWise, no rendering order). Boxes are converted to
# PDF-style y-up coordinates (top > bottom) so the interval relations port verbatim.
UNK, PRE, MEE, OVE, STA, DUR, FIN, PREI, MEEI, OVEI, STAI, DURI, FINI, EQU = range(14)


def _rel_x(a, b, T):
    al, ar, bl, br = a[0], a[2], b[0], b[2]
    if bl - T <= al <= bl + T and br - T <= ar <= br + T:
        return EQU
    if bl - T <= ar <= bl + T:
        return MEE
    if al - T <= br <= al + T:
        return MEEI
    if bl - T <= al <= bl + T and ar < br - T:
        return STA
    if al - T <= bl <= al + T and br < ar - T:
        return STAI
    if al > bl + T and br - T <= ar <= br + T:
        return FIN
    if bl > al + T and ar - T <= br <= ar + T:
        return FINI
    if al > bl + T and ar < br - T:
        return DUR
    if bl > al + T and br < ar - T:
        return DURI
    if al < bl - T and bl + T < ar < br - T:
        return OVE
    if bl < al - T and al + T < br < ar - T:
        return OVEI
    if ar < bl - T:
        return PRE
    if br < al - T:
        return PREI
    return UNK


def _rel_y(a, b, T):
    at, ab, bt, bb = a[1], a[3], b[1], b[3]  # y-up: top > bottom
    if bt - T <= at <= bt + T and bb - T <= ab <= bb + T:
        return EQU
    if at - T <= bb <= at + T:
        return MEEI
    if bt - T <= ab <= bt + T:
        return MEE
    if bt - T <= at <= bt + T and ab < bb - T:
        return STAI
    if at - T <= bt <= at + T and bb < ab - T:
        return STA
    if at > bt + T and bb - T <= ab <= bb + T:
        return FINI
    if bt > at + T and ab - T <= bb <= ab + T:
        return FIN
    if at > bt + T and ab < bb - T:
        return DURI
    if bt > at + T and bb < ab - T:
        return DUR
    if at < bt - T and (bb + T < at and ab < bb - T):
        return OVEI
    if bt < at - T and (ab + T < bt and bb < ab - T):
        return OVE
    if ab < bt - T:
        return PREI
    if bb < at - T:
        return PRE
    return UNK


_COLWISE_X = {PRE, MEE, OVE, STA, FINI, EQU, DUR, DURI, FIN, STAI, OVEI}


def _before_vertical(a, b, T):
    x = _rel_x(a, b, T)
    y = _rel_y(a, b, T)
    return (x in (PRE, MEE) or (x == OVE and y in (PRE, MEE, OVE)) or (y in (PRE, MEE, OVE) and x in _COLWISE_X))


def pdfpig_order(boxes, W, H, T=None):
    if T is None:
        T = 5.0 * W / 612.0
    n = len(boxes)
    up = [(b[0], H - b[1], b[2], H - b[3]) for b in boxes]
    graph = {i: [j for j in range(n) if j != i and _before_vertical(up[i], up[j], T)] for i in range(n)}
    out = []
    while graph:
        m = max(len(v) for v in graph.values())
        cur = next(k for k, v in graph.items() if len(v) == m)
        del graph[cur]
        for v in graph.values():
            if cur in v:
                v.remove(cur)
        out.append(cur)
    return out
