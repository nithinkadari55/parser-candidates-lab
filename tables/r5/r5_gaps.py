"""Round 5c: split-row features for the selector (TATR's output is not changed).

TATR sometimes gives each line of a tall row its own grid row under a spanning label (elsevier-00 p7). Real sub-rows
under a spanning label look the same in the grid, so the signal is geometric: wrapped lines sit about one line
height apart, real rows further. Per model, on the cells with a positive area (as feats() filters them):
  join_n        rows i that continue the row above under a spanning label: some cell covers rows i-1 and i, and
                every cell starting at row i is one row high under a cell of the same columns that ends at row i
  join_share    join_n / (rows - 1)
  join_gap_line median word gap of those row pairs / median word height      (-1 without candidates)
  join_gap_row  the same median gap / median gap between all adjacent rows   (-1 without candidates or row gaps)
  row_gap_line  median gap between adjacent rows / median word height        (-1 without row gaps)
The gap between rows r and r+1 is the top of row r+1's words minus the bottom of row r's words, counting the words
whose best cell (largest overlap share, ties to the later cell, as in feats) is one row high in that row.
Selector row: a_<name>, b_<name>, d_<name> = b - a for each name."""
import numpy as np

NAMES = ["join_n", "join_share", "join_gap_line", "join_gap_row", "row_gap_line"]


def _area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b): return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def candidates(cells):
    if not cells: return []
    rows = max(c["r1"] for c in cells)
    out = []
    for i in range(1, rows):
        starting = [c for c in cells if c["r0"] == i]
        if not starting: continue
        if not any(c["r0"] < i < c["r1"] for c in cells): continue
        ok = True
        for c in starting:
            if c["r1"] != i + 1:
                ok = False; break
            above = [p for p in cells if p["r1"] == i and p["c0"] == c["c0"] and p["c1"] == c["c1"]]
            if len(above) != 1:
                ok = False; break
        if ok: out.append(i)
    return out


def gap_feats(cells, words):
    cells = [c for c in cells if c.get("bbox") and _area(c["bbox"]) > 0]
    f = {k: (0.0 if k in ("join_n", "join_share") else -1.0) for k in NAMES}
    if not cells: return f
    rows = max(c["r1"] for c in cells)
    hs = [w["bbox"][3] - w["bbox"][1] for w in words if _area(w["bbox"]) > 0]
    mh = float(np.median(hs)) if hs else 1.0
    top, bottom = {}, {}
    for w in words:
        wb = w["bbox"]; wa = _area(wb)
        if wa <= 0: continue
        ov = sorted(((_inter(wb, c["bbox"]) / wa, k) for k, c in enumerate(cells)), reverse=True)
        if ov[0][0] <= 0: continue
        c = cells[ov[0][1]]
        if c["r1"] - c["r0"] != 1: continue
        r = c["r0"]
        top[r] = min(top.get(r, wb[1]), wb[1]); bottom[r] = max(bottom.get(r, wb[3]), wb[3])
    def gap(r):
        return top[r + 1] - bottom[r] if r in bottom and r + 1 in top else None
    row_gaps = [g for g in (gap(r) for r in range(rows - 1)) if g is not None]
    cand = candidates(cells)
    cand_gaps = [g for g in (gap(i - 1) for i in cand) if g is not None]
    f["join_n"] = float(len(cand))
    f["join_share"] = len(cand) / max(1, rows - 1)
    med_row = float(np.median(row_gaps)) if row_gaps else None
    if med_row is not None: f["row_gap_line"] = med_row / mh
    if cand_gaps:
        med_c = float(np.median(cand_gaps))
        f["join_gap_line"] = med_c / mh
        if med_row is not None and med_row > 0: f["join_gap_row"] = med_c / med_row
    return f


def add_to_row(x, ca, cb, words):
    ga, gb = gap_feats(ca, words), gap_feats(cb, words)
    for k in NAMES:
        x["a_" + k] = ga[k]; x["b_" + k] = gb[k]; x["d_" + k] = gb[k] - ga[k]
    return x
