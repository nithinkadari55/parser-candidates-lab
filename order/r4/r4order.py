"""Round 4 reading-order add-ons on top of our XY-Cut++ (order/scripts/ro.py), for newspapers, magazines, sidebars
and exercise grids. Geometry still proposes the orders; the new cues only choose or constrain:

cont   text continuity: several geometric candidate orders (XY-Cut++ variants, plain XY-cuts, column-major and
       local successor repairs) are scored by transitions between consecutive text blocks: unfinished sentence,
       line-end hyphen, lowercase start, lexical continuity (rare shared terms, TextTiling-style); a candidate
       replaces XY-Cut++ only if it beats it by a margin chosen on the dev split.
side   sidebar masks: tinted or ruled boxes found in the page image (rects from order/scripts/addons.py
       detect_rects) that enclose >= 2 blocks become one unit ordered inside; policies inline (unit placed by
       XY-Cut++), after (after the main-flow blocks of the column it sits in), end (after the page's main flow).
grid   exercise numbers: numbered items (1., 2), (3), Q4, Exercise 5) must read in increasing order; gaps allowed.
jump   "continued on/from page X" lines are kept with the article block they end or start.
"""
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import addons  # noqa: E402
import ro  # noqa: E402

TEXTISH = {"text", "list_item"}
STOP = set("""the and for that with this from have were which their there been will would could should about into
than then them they these those such other also more most some very what when where while your only over under
after before between because being does done each just like made make many much must same than upon whom whose
https http www""".split())
TERM = set(".!?:;\"'”’)]")
NUM_RE = re.compile(r"^\s*(?:(?:question|exercise|problem|q|ex)\.?\s*)?\(?(\d{1,3})\s*[\.\):]?\s+\S", re.I)
JUMP_RE = re.compile(r"\b(continued|cont'?d\.?|see|turn to)\s+(on|from|to)?\s*(page|p\.)\s*[a-z]?\d+", re.I)


# ---------------------------------------------------------------- text continuity
def words(t):
    return re.findall(r"[A-Za-z][A-Za-z'\-]+", t or "")


class Cont:
    def __init__(self, texts):
        df = Counter()
        for t in texts:
            df.update({w.lower() for w in words(t) if len(w) >= 4 and w.lower() not in STOP})
        n = max(1, len(texts))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def lex(self, a, b):
        ta = {w.lower() for w in words(a)[-25:] if w.lower() in self.idf}
        tb = {w.lower() for w in words(b)[:25] if w.lower() in self.idf}
        sh = ta & tb
        return min(1.0, sum(self.idf[w] for w in sh) / 6.0) if sh else 0.0

    def trans(self, a, la, b, lb):
        """score of reading block b right after block a"""
        ta, tb = (a or "").rstrip(), (b or "").lstrip()
        if not ta or not tb:
            return 0.0
        end, start = ta[-1], next((c for c in tb if c.isalnum()), "")
        unfinished = end.isalnum() or end in ",-­"
        s = 0.0
        if lb == "section_header":
            return -0.8 if unfinished and la in TEXTISH else 0.0
        if la not in TEXTISH or lb not in TEXTISH:
            return 0.0
        if end in "-­" and start.islower():
            s += 2.0
        elif unfinished and start.islower():
            s += 1.5
        elif unfinished and start.isupper():
            s -= 0.4
        elif end in TERM and start.islower():
            s -= 1.0
        elif end in TERM and start.isupper():
            s += 0.2
        return s + 0.6 * self.lex(ta, tb)

    def score(self, order, labels, texts):
        return sum(self.trans(texts[a], labels[a], texts[b], labels[b]) for a, b in zip(order, order[1:]))


# ---------------------------------------------------------------- candidate orders
def column_major(boxes, idx, W):
    if not idx:
        return []
    cols = ro._groups(boxes, idx, "x", 0.002 * W)
    out = []
    for c in cols:
        out += sorted(c, key=lambda i: (boxes[i][1], boxes[i][0]))
    return out


def repairs(order, labels, texts, C, k=3):
    """move a strong continuation block (and the run after it) right behind an unfinished block"""
    pos = {u: i for i, u in enumerate(order)}
    out = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        cur = C.trans(texts[a], labels[a], texts[b], labels[b])
        if cur > 0:
            continue
        best, bj = 1.4, None
        for c in order[i + 2:]:
            v = C.trans(texts[a], labels[a], texts[c], labels[c])
            if v > best:
                best, bj = v, c
        if bj is None:
            continue
        j = pos[bj]
        e = j + 1
        while e < len(order) and C.trans(texts[order[e - 1]], labels[order[e - 1]], texts[order[e]], labels[order[e]]) > 0:
            e += 1
        run = order[j:e]
        rest = order[:i + 1] + run + [u for u in order[i + 1:] if u not in set(run)]
        out.append((best - cur, rest))
    out.sort(key=lambda t: -t[0])
    return [o for _, o in out[:k]]


def candidates(boxes, idx, W, H):
    sub = [boxes[i] for i in idx]

    def back(o):
        return [idx[k] for k in o]
    cs = [back(ro.xycutpp(sub, W, H)), back(ro.xycutpp(sub, W, H, stack_rule=False)),
          back(ro.xycutpp(sub, W, H, row_rule=True)), back(ro.xycut(sub, W, H, "y")), back(ro.xycut(sub, W, H, "x")),
          column_major(boxes, list(idx), W)]
    return cs


# ---------------------------------------------------------------- sidebars
def sidebars(rects, boxes, labels, idx):
    return addons.sections(rects, boxes, labels, idx) if rects else []


def place_after(main, units, boxes):
    """insert each sidebar unit after the last main block that overlaps it horizontally and starts above its bottom"""
    seq = list(main)
    for ub, mem in sorted(units, key=lambda u: u[0][1]):
        anchor = None
        for k, i in enumerate(seq):
            b = boxes[i]
            if min(b[2], ub[2]) - max(b[0], ub[0]) > 0 and b[1] < ub[3]:
                anchor = k
        if anchor is None:
            seq = seq + mem
        else:
            seq = seq[:anchor + 1] + mem + seq[anchor + 1:]
    return seq


# ---------------------------------------------------------------- grid numbering
def numbering(seq, boxes, texts):
    num = {}
    for i in seq:
        m = NUM_RE.match(texts[i] or "")
        if m:
            num[i] = int(m.group(1))
    if len(num) < 3:
        return seq, False
    vals = [num[i] for i in seq if i in num]
    if len(set(vals)) != len(vals) or vals == sorted(vals):
        return seq, False
    sv = sorted(vals)
    if max(b - a for a, b in zip(sv, sv[1:])) > 2:
        return seq, False
    heads = [k for k, i in enumerate(seq) if i in num]
    groups = []
    for a, k in enumerate(heads):
        end = heads[a + 1] if a + 1 < len(heads) else len(seq)
        h = seq[k]
        g = [h]
        for m in range(k + 1, end):
            j = seq[m]
            if boxes[j][1] >= boxes[h][1] - 2 and ro._xov(boxes[j], boxes[h]):
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
    return out, True


# ---------------------------------------------------------------- jump lines
def jumps(seq, boxes, texts):
    hits = [i for i in seq if JUMP_RE.search(texts[i] or "") and len((texts[i] or "").split()) <= 14]
    if not hits:
        return seq, 0
    out = [i for i in seq if i not in set(hits)]
    for j in hits:
        b = boxes[j]
        frm = re.search(r"from", texts[j] or "", re.I) is not None
        cands = [i for i in out if ro._xov(boxes[i], b)]
        if frm:
            below = [i for i in cands if boxes[i][1] >= b[3] - 2]
            tgt = min(below, key=lambda i: boxes[i][1]) if below else None
            if tgt is None:
                out.append(j)
            else:
                k = out.index(tgt)
                out = out[:k] + [j] + out[k:]
        else:
            above = [i for i in cands if boxes[i][3] <= b[1] + 2]
            tgt = max(above, key=lambda i: boxes[i][3]) if above else None
            if tgt is None:
                out.append(j)
            else:
                k = out.index(tgt)
                out = out[:k + 1] + [j] + out[k + 1:]
    return out, len(hits)


# ---------------------------------------------------------------- one configuration -> candidate orders
CONFIGS = ["base", "side_inline", "side_after", "side_end", "grid", "jump", "all_inline", "all_after", "all_end"]


def run_config(cfg, boxes, labels, texts, W, H, rects, keep):
    """-> list of candidate orders over `keep` (candidate 0 = no continuity choice), trigger counts"""
    trig = {"sidebars": 0, "grid": 0, "jumps": 0}
    side = cfg.split("_")[1] if "_" in cfg else None
    idx = list(keep)
    units = []
    if side:
        for r, mem in sidebars(rects, boxes, labels, idx):
            inner = [mem[k] for k in ro.xycutpp([boxes[i] for i in mem], W, H)]
            units.append((addons.ubox([boxes[i] for i in mem] + [r]), inner))
        trig["sidebars"] = len(units)
        used = {i for _, m in units for i in m}
        idx = [i for i in idx if i not in used]
    ext = list(boxes)
    if side == "inline" and units:
        unit_ids = []
        for ub, mem in units:
            ext.append(ub)
            unit_ids.append(len(ext) - 1)
        work = idx + unit_ids
        umap = {u: units[k][1] for k, u in enumerate(unit_ids)}
    else:
        work, umap = idx, {}
    base_cands = candidates(ext, work, W, H)
    cands = []
    for o in base_cands:
        o = [x for u in o for x in umap.get(u, [u])]
        if side in ("after", "end") and units:
            o = place_after(o, units, boxes) if side == "after" else o + [i for _, m in sorted(units, key=lambda u: u[0][1]) for i in m]
        if cfg in ("grid",) or cfg.startswith("all"):
            o, g = numbering(o, boxes, texts)
            trig["grid"] = max(trig["grid"], int(g))
        if cfg in ("jump",) or cfg.startswith("all"):
            o, nj = jumps(o, boxes, texts)
            trig["jumps"] = max(trig["jumps"], nj)
        cands.append(o)
    return cands, trig


def with_repairs(cands, labels, texts, C):
    return cands + repairs(cands[0], labels, texts, C)
