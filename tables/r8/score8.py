"""Round 8 scoring: score.py's score_one (TEDS, TEDS-Struct, GriTS-Top/Con, exact rows and columns) with the run-time
words as the cell text. Answer keys over 1,200 grid cells are left unscored for every arm (as rounds 3-5); a prediction
over 1,200 cells on a smaller answer key is still scored (up to 5,000 cells), so a runaway grid is not dropped."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import score as SC
from prep_sets import parse_html, place

KEEP = ("teds", "teds_s", "grits_top", "grits_con", "rows", "cols", "gt_rows", "gt_cols", "rc_exact", "header_ok", "n_cells", "fail", "score_err", "grits_err")
_size = {}


def gt_cells(gt):
    k = gt["id"]
    if k not in _size:
        rows, _ = parse_html(gt["html"]); _, R, C = place(rows); _size[k] = R * C
    return _size[k]


def score_cells(gt, words, cells, structure_only=False):
    g = dict(gt); g["words"] = [dict(text=w["text"], bbox=w["bbox"], gi=None) for w in words]
    big = gt_cells(gt) > 1200
    SC.TEDS_MAX = SC.GRITS_MAX = 1200 if big else 5000
    try:
        if structure_only:
            r = structure(g, cells)
        else:
            r = SC.score_one((g, dict(id=gt["id"], cells=cells)))
    except Exception as e:
        r = dict(id=gt["id"], score_err=repr(e)[:200], teds_s=None, grits_top=None, rc_exact=False)
    out = {k: r.get(k) for k in KEEP}
    if not big and out.get("teds_s") is None and out.get("grits_top") is None and not out.get("score_err"):
        out["teds_s"] = out["grits_top"] = 0.0; out["runaway"] = True
    return out


def structure(g, cells):
    """TEDS-Struct, GriTS-Top only when TEDS is capped, and exact shape (the tuning loop's cheap score)"""
    from metric import TEDS
    import distance as _d
    from rapidfuzz.distance import Levenshtein as _L
    _d.levenshtein = _L.distance
    rows, thead = parse_html(g["html"]); pos, GR, GC = place(rows)
    if not cells:
        z = None if GR * GC > SC.TEDS_MAX else 0.0
        return dict(teds_s=z, grits_top=z, rows=0, cols=0, gt_rows=GR, gt_cols=GC, rc_exact=False)
    R = max(c["r1"] for c in cells); C = max(c["c1"] for c in cells)
    res = dict(rows=R, cols=C, gt_rows=GR, gt_cols=GC, rc_exact=(R == GR and C == GC), teds_s=None, grits_top=None)
    ph = SC.pred_html(R, C, cells, {}); gh = SC.gt_html(g)
    if GR * GC <= SC.TEDS_MAX and R * C <= SC.TEDS_MAX:
        res["teds_s"] = TEDS(structure_only=True, ignore_nodes=["thead", "tbody"]).evaluate(ph, gh)
    elif GR * GC <= SC.GRITS_MAX and R * C <= SC.GRITS_MAX:
        import grits
        gg = grits.grits_from_html(SC.gt_html(g, True)[12:-14], SC.pred_html(R, C, cells, {}, True)[12:-14])
        res["grits_top"] = float(gg["grits_top"])
    return res


def value(s):
    """the per-table score of rounds 3-5: TEDS-Struct, GriTS-Top where TEDS is capped"""
    if not s: return None
    return s["teds_s"] if s.get("teds_s") is not None else s.get("grits_top")
