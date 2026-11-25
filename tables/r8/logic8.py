"""Round 8 arm 3: deterministic repairs of TATR's grid from the words (no model). Each rule decides on the grid it is
given, then applies its edits from the bottom/right up, so indices stay valid. Cells: dicts with r0, r1, c0, c1, bbox
(crop pixels), header. Words: dicts with text, bbox. Rules and their settings (tuned on the dev split only):

  E   empty rows / columns: drop grid rows (columns) that hold no word, optionally only thin ones (< 1 line high,
      < 1.5 line widths wide)                                           rows, cols in {off, thin, all}
  CM  column merge: adjacent columns that many words straddle (>= 20% of the word's width in each)
                                                                        share in {.15, .3, .5}
  CS  column split: a vertical gap of no ink >= g line heights inside a column with words on both sides of it in
      >= f of the column's non-empty rows                                g in {1, 2}, f in {.5, .8}
  RS  row split: a row where >= m cells each hold the same number (>= 2) of text lines at the same heights
      (optionally one of them all numbers)                               m in {2, 3}, numeric in {yes, no}
  J   wrapped-row join: row i continues row i-1 when its filled columns are a subset of row i-1's (no number-only
      cell), or row i-1 holds only the first filled column and row i adds the values; gated on the vertical gap
      <= G line heights and <= R times the table's typical gap between data rows, optionally on a continuation cue
      (row i starts in lower case, or row i-1 ends in a hyphen, comma or joining word); header rows optional
                                                                        G, R, cue, header
"""
import re
import numpy as np

NUM = re.compile(r"^[\s\d.,:;%()$€£¥+\-−–—/*xX]+$")
JOIN_WORDS = {"and", "or", "of", "the", "to", "in", "for", "with", "on", "by", "a", "an", "&", "from", "at", "as", "per", "vs", "vs."}


def _arr(xs, key="bbox"):
    return np.array([x[key] for x in xs], dtype=float).reshape(-1, 4)


def assign(cells, words):
    """index of the cell holding each word (largest share of the word's area), or -1"""
    if not cells or not words: return np.full(len(words), -1)
    C, W = _arr(cells), _arr(words)
    ix = np.clip(np.minimum(W[:, None, 2], C[None, :, 2]) - np.maximum(W[:, None, 0], C[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(W[:, None, 3], C[None, :, 3]) - np.maximum(W[:, None, 1], C[None, :, 1]), 0, None)
    a = np.clip((W[:, 2] - W[:, 0]) * (W[:, 3] - W[:, 1]), 1e-6, None)
    sh = ix * iy / a[:, None]
    best = sh.argmax(1)
    return np.where(sh[np.arange(len(words)), best] > 0, best, -1)


def line_height(words):
    hs = [w["bbox"][3] - w["bbox"][1] for w in words if w["bbox"][3] > w["bbox"][1]]
    return float(np.median(hs)) if hs else 10.0


def dims(cells):
    return (max(c["r1"] for c in cells), max(c["c1"] for c in cells)) if cells else (0, 0)


def _union(a, b): return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def resolve(cells):
    """merge cells whose grid rectangles overlap (after a row/column merge) into their union"""
    cells = [dict(c) for c in cells]
    changed = True
    while changed:
        changed = False
        occ = {}
        for k, c in enumerate(cells):
            for r in range(c["r0"], c["r1"]):
                for q in range(c["c0"], c["c1"]):
                    if (r, q) in occ:
                        j = occ[(r, q)]; a, b = cells[j], c
                        m = dict(r0=min(a["r0"], b["r0"]), r1=max(a["r1"], b["r1"]), c0=min(a["c0"], b["c0"]), c1=max(a["c1"], b["c1"]),
                                 bbox=_union(a["bbox"], b["bbox"]), header=bool(a.get("header") or b.get("header")))
                        cells = [x for i, x in enumerate(cells) if i not in (j, k)] + [m]
                        changed = True; break
                    occ[(r, q)] = k
                if changed: break
            if changed: break
    return cells


def merge_index(cells, i, axis):
    """grid lines i-1 and i (rows if axis == 'r', columns if 'c') become one"""
    a0, a1 = ("r0", "r1") if axis == "r" else ("c0", "c1")
    f = lambda v: v if v < i else v - 1
    out = []
    for c in cells:
        n = dict(c); n[a0] = f(c[a0]); n[a1] = f(c[a1] - 1) + 1; out.append(n)
    return resolve(out)


def drop_index(cells, i, axis):
    a0, a1 = ("r0", "r1") if axis == "r" else ("c0", "c1")
    out = []
    for c in cells:
        if c[a0] == i and c[a1] == i + 1: continue
        n = dict(c)
        if n[a0] > i: n[a0] -= 1
        if n[a1] > i: n[a1] -= 1
        out.append(n)
    return out


def split_index(cells, i, cuts, axis):
    """grid line i becomes len(cuts) + 1 lines, cut at the given pixel positions (y for rows, x for columns)"""
    a0, a1 = ("r0", "r1") if axis == "r" else ("c0", "c1")
    lo, hi = (1, 3) if axis == "r" else (0, 2)
    k = len(cuts); out = []
    for c in cells:
        if c[a0] == i and c[a1] == i + 1:
            edges = [c["bbox"][lo]] + list(cuts) + [c["bbox"][hi]]
            for j in range(k + 1):
                n = dict(c); n[a0] = i + j; n[a1] = i + j + 1
                b = list(c["bbox"]); b[lo] = edges[j]; b[hi] = max(edges[j], edges[j + 1]); n["bbox"] = b
                out.append(n)
        else:
            n = dict(c)
            if n[a0] > i: n[a0] += k
            if n[a1] > i: n[a1] += k
            out.append(n)
    return out


# ---------- per-row / per-column content ----------

class View:
    def __init__(self, cells, words):
        self.cells, self.words = cells, words
        self.R, self.C = dims(cells)
        self.mh = line_height(words)
        self.own = assign(cells, words)
        self.single_r = {}  # row -> {c0: [word idx]} for one-row cells
        self.single_c = {}
        for wi, k in enumerate(self.own):
            if k < 0: continue
            c = cells[k]
            if c["r1"] - c["r0"] == 1: self.single_r.setdefault(c["r0"], {}).setdefault(c["c0"], []).append(wi)
            if c["c1"] - c["c0"] == 1: self.single_c.setdefault(c["c0"], {}).setdefault(c["r0"], []).append(wi)

    def filled(self, r): return set(self.single_r.get(r, {}))

    def ys(self, r):
        ws = [self.words[w]["bbox"] for lst in self.single_r.get(r, {}).values() for w in lst]
        return (min(b[1] for b in ws), max(b[3] for b in ws)) if ws else None

    def text(self, r, c0):
        ws = sorted((self.words[w] for w in self.single_r.get(r, {}).get(c0, [])), key=lambda w: (round(w["bbox"][1] / max(1.0, 0.5 * self.mh)), w["bbox"][0]))
        return " ".join(w["text"] for w in ws).strip()

    def header_row(self, r):
        return any(c.get("header") for c in self.cells if c["r0"] <= r < c["r1"])

    def any_word_in_row(self, r):
        band = [c["bbox"] for c in self.cells if c["r0"] == r and c["r1"] == r + 1]
        if r in self.single_r: return True
        if not band: return False
        y0, y1 = min(b[1] for b in band), max(b[3] for b in band)
        return any(y0 <= (w["bbox"][1] + w["bbox"][3]) / 2 <= y1 for w in self.words)

    def any_word_in_col(self, q):
        band = [c["bbox"] for c in self.cells if c["c0"] == q and c["c1"] == q + 1]
        if q in self.single_c: return True
        if not band: return False
        x0, x1 = min(b[0] for b in band), max(b[2] for b in band)
        return any(x0 <= (w["bbox"][0] + w["bbox"][2]) / 2 <= x1 for w in self.words)

    def band(self, i, axis):
        if axis == "r":
            b = [c["bbox"] for c in self.cells if c["r0"] == i and c["r1"] == i + 1]
            return (float(np.median([x[1] for x in b])), float(np.median([x[3] for x in b]))) if b else None
        b = [c["bbox"] for c in self.cells if c["c0"] == i and c["c1"] == i + 1]
        return (float(np.median([x[0] for x in b])), float(np.median([x[2] for x in b]))) if b else None


# ---------- rules ----------

def rule_empty(cells, words, rows="off", cols="off"):
    if not cells or (rows == "off" and cols == "off"): return cells
    v = View(cells, words)
    if rows != "off" and v.R > 2:
        drop = []
        for r in range(v.R):
            if v.any_word_in_row(r): continue
            b = v.band(r, "r")
            if rows == "thin" and (b is None or b[1] - b[0] >= v.mh): continue
            drop.append(r)
        if len(drop) < v.R - 1:
            for r in sorted(drop, reverse=True): cells = drop_index(cells, r, "r")
    if cols != "off":
        v = View(cells, words)
        if v.C > 2:
            drop = []
            for q in range(v.C):
                if v.any_word_in_col(q): continue
                b = v.band(q, "c")
                if cols == "thin" and (b is None or b[1] - b[0] >= 1.5 * v.mh): continue
                drop.append(q)
            if len(drop) < v.C - 1:
                for q in sorted(drop, reverse=True): cells = drop_index(cells, q, "c")
    return cells


def rule_colmerge(cells, words, share=None):
    if not cells or share is None: return cells
    v = View(cells, words)
    bands = [v.band(q, "c") for q in range(v.C)]
    merges = []
    for q in range(1, v.C):
        a, b = bands[q - 1], bands[q]
        if a is None or b is None: continue
        n_in = straddle = 0
        for w in words:
            x0, x1 = w["bbox"][0], w["bbox"][2]; wd = max(1e-6, x1 - x0)
            oa = max(0.0, min(x1, a[1]) - max(x0, a[0])) / wd; ob = max(0.0, min(x1, b[1]) - max(x0, b[0])) / wd
            if oa > 0 or ob > 0: n_in += 1
            if oa >= 0.2 and ob >= 0.2: straddle += 1
        if straddle >= 2 and straddle >= share * max(1, n_in): merges.append(q)
    for q in sorted(merges, reverse=True): cells = merge_index(cells, q, "c")
    return cells


def rule_colsplit(cells, words, gap=None, frac=0.8):
    if not cells or gap is None: return cells
    v = View(cells, words)
    plan = []
    for q in range(v.C):
        rows = v.single_c.get(q, {})
        b = v.band(q, "c")
        if b is None or len(rows) < 2: continue
        x0, x1 = int(np.floor(b[0])), int(np.ceil(b[1]))
        if x1 - x0 < 4: continue
        cov = np.zeros(x1 - x0 + 1, bool)
        for lst in rows.values():
            for wi in lst:
                wb = words[wi]["bbox"]; cov[max(0, int(wb[0]) - x0):max(0, int(np.ceil(wb[2])) - x0) + 1] = True
        idx = np.where(cov)[0]
        if len(idx) < 2: continue
        best = None; run = None
        for p in range(idx[0], idx[-1] + 1):
            if not cov[p]:
                run = (run[0], p) if run else (p, p)
            elif run:
                if best is None or run[1] - run[0] > best[1] - best[0]: best = run
                run = None
        if best is None or best[1] - best[0] + 1 < gap * v.mh: continue
        cut = x0 + (best[0] + best[1]) / 2.0
        both = sum(1 for lst in rows.values() if any(words[w]["bbox"][2] <= cut for w in lst) and any(words[w]["bbox"][0] >= cut for w in lst))
        if both >= 2 and both >= frac * len(rows): plan.append((q, cut))
    for q, cut in sorted(plan, reverse=True): cells = split_index(cells, q, [cut], "c")
    return cells


def _lines(ws, mh):
    ys = sorted((w["bbox"][1] + w["bbox"][3]) / 2 for w in ws)
    out = []
    for y in ys:
        if out and y - out[-1][-1] <= 0.5 * mh: out[-1].append(y)
        else: out.append([y])
    return [float(np.mean(l)) for l in out]


def rule_rowsplit(cells, words, m=None, numeric=True):
    if not cells or m is None: return cells
    v = View(cells, words)
    plan = []
    for r in range(v.R):
        per = v.single_r.get(r, {})
        groups = {}
        for c0, lst in per.items():
            ls = _lines([words[w] for w in lst], v.mh)
            if len(ls) >= 2: groups.setdefault(len(ls), []).append((c0, ls, lst))
        for k, g in sorted(groups.items(), key=lambda x: -len(x[1])):
            ref = g[0][1]
            al = [x for x in g if all(abs(a - b) <= 0.4 * v.mh for a, b in zip(x[1], ref))]
            if len(al) < m: continue
            if numeric and not any(all(NUM.match(words[w]["text"]) for w in x[2]) for x in al): continue
            cuts = [(ref[j] + ref[j + 1]) / 2 for j in range(k - 1)]
            plan.append((r, cuts)); break
    for r, cuts in sorted(plan, reverse=True): cells = split_index(cells, r, cuts, "r")
    return cells


def rule_join(cells, words, G=None, R=float("inf"), cue=True, header=False):
    if not cells or G is None: return cells
    v = View(cells, words)
    first = min((c for r in range(v.R) for c in v.filled(r)), default=0)
    gaps = []
    for r in range(v.R - 1):
        a, b = v.ys(r), v.ys(r + 1)
        if a and b and len(v.filled(r)) >= 2 and len(v.filled(r + 1)) >= 2: gaps.append(b[0] - a[1])
    typical = float(np.median(gaps)) if len(gaps) >= 2 else None
    joins = []
    for i in range(1, v.R):
        fa, fb = v.filled(i - 1), v.filled(i)
        if not fa or not fb: continue
        if not header and (v.header_row(i) or v.header_row(i - 1)): continue
        if header and v.header_row(i) != v.header_row(i - 1): continue
        p1 = fb <= fa and not any(NUM.match(v.text(i, c)) for c in fb)
        p2 = fa == {first} and first in fb and len(fb) >= 2
        if not (p1 or p2): continue
        a, b = v.ys(i - 1), v.ys(i)
        if not a or not b: continue
        g = b[0] - a[1]
        if g > G * v.mh: continue
        if typical is not None and np.isfinite(R) and g > R * max(typical, 1e-6): continue
        if cue:
            c0 = min(fb); t = v.text(i, c0); above = v.text(i - 1, c0) if c0 in fa else ""
            low = bool(t) and (t[0].islower() or t[0] in "([")
            tail = above.split()[-1].lower() if above.split() else ""
            inc = bool(above) and (above.endswith(("-", ",", "/", "(")) or tail in JOIN_WORDS)
            if not (low or inc): continue
        joins.append(i)
    for i in sorted(joins, reverse=True): cells = merge_index(cells, i, "r")
    return cells


ORDER = ["E", "CM", "CS", "RS", "J"]
SPACE = {
    "E": [dict(rows=r, cols=c) for r in ("off", "thin", "all") for c in ("off", "thin", "all") if (r, c) != ("off", "off")],
    "CM": [dict(share=s) for s in (0.15, 0.3, 0.5)],
    "CS": [dict(gap=g, frac=f) for g in (1.0, 2.0) for f in (0.5, 0.8)],
    "RS": [dict(m=m, numeric=nu) for m in (2, 3) for nu in (True, False)],
    "J": [dict(G=G, R=R, cue=cue, header=False) for G in (0.35, 0.6, 1.0) for R in (0.6, 0.85, float("inf")) for cue in (True, False)],
}
FUNCS = {"E": rule_empty, "CM": rule_colmerge, "CS": rule_colsplit, "RS": rule_rowsplit, "J": rule_join}


def apply(cells, words, config):
    """config: {rule: settings} applied in ORDER"""
    cells = [dict(r0=c["r0"], r1=c["r1"], c0=c["c0"], c1=c["c1"], bbox=list(c["bbox"]), header=bool(c.get("header"))) for c in cells]
    for k in ORDER:
        if k in config and config[k] is not None:
            cells = FUNCS[k](cells, words, **config[k])
    return cells


def signature(cells):
    return tuple(sorted((c["r0"], c["r1"], c["c0"], c["c1"]) for c in cells))
