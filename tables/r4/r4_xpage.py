"""Gap 6: cross-page / cross-column table continuation and over-merged region splitting on PubTables-v2 Full Documents (test).

Fragments are the ground-truth table parts (oracle boxes); every decision uses only the page words inside and
around them (the dataset's PDF word boxes), never the logical-table labels. Rules:
  textract   - faithful port of aws-samples/amazon-textract-response-parser t_tables.ExecuteTableValidations
               (Apache-2.0): last table of page p vs first table of page p+1, no lines between outside the header/footer
               band, same column count or identical header row, width and left within (100 - accuracy)%.
  stitcher   - port of PebbleRoad/table-stitcher merger._classify_sequential_pair (MIT) with MultiPageConfig defaults.
  r4         - ours: Stitcher-style evidence plus column-boundary match, column-to-column continuation on one page,
               'continued' captions and caption-number guards.
usage: r4_xpage.py"""
import os, sys, re, json, glob, collections, subprocess
from r4_common import *
import numpy as np

REPO = "kensho/PubTables-v2"
LIMIT = int(os.environ.get("TB_LIMIT") or 0)
CAP = re.compile(r"^\s*(table|tab\.)\s*([A-Z]?\d+)", re.I)
CONT = re.compile(r"contin|cont'd|contd|\(cont", re.I)
DATA = [re.compile(p, re.I) for p in [r"^[-+]?\d+$", r"^[-+]?[\d,]*\.\d+$", r"^[\d,]+$", r"^[\d,.]+\s*%$", r"^\d+\.?\d*\s*\([\d,\s.\-–]+\)", r"^[<>≤≥]?\s*0?\.\d+", r"^\d{1,2}/\d{1,2}", r"^[\d.]+\s*±"]]


def fetch():
    from huggingface_hub import hf_hub_download
    d = os.path.join(RAW, "pt2fd"); os.makedirs(d, exist_ok=True)
    for part in ("tables", "words", "xml-annotations", "cross-page-table-pairs", "single-page-table-continuations"):
        p = hf_hub_download(REPO, f"PubTables-v2_Full-Documents_test_{part}.tar.gz", repo_type="dataset", local_dir=d)
        subprocess.run(["tar", "-xzf", p, "-C", d], check=True); os.remove(p)
    return os.path.join(d, "Full Documents", "test")


def lines_of(words):
    ws = sorted(words, key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2, w["bbox"][0]))
    out = []
    for w in ws:
        cy = (w["bbox"][1] + w["bbox"][3]) / 2; h = max(1.0, w["bbox"][3] - w["bbox"][1])
        for ln in out[-3:]:
            if abs(ln["cy"] - cy) <= 0.5 * min(ln["h"], h):
                ln["w"].append(w); ln["cy"] = (ln["cy"] * (len(ln["w"]) - 1) + cy) / len(ln["w"]); ln["h"] = max(ln["h"], h); break
        else:
            out.append(dict(cy=cy, h=h, w=[w]))
    for ln in out:
        ln["w"].sort(key=lambda w: w["bbox"][0])
        ln["x0"] = min(w["bbox"][0] for w in ln["w"]); ln["x1"] = max(w["bbox"][2] for w in ln["w"])
        ln["y0"] = min(w["bbox"][1] for w in ln["w"]); ln["y1"] = max(w["bbox"][3] for w in ln["w"])
        ln["text"] = " ".join(w["text"] for w in ln["w"])
        thr = max(6.0, 0.8 * ln["h"]); ph = []
        for w in ln["w"]:
            if ph and w["bbox"][0] - ph[-1][1] <= thr: ph[-1][1] = max(ph[-1][1], w["bbox"][2]); ph[-1][2] += " " + w["text"]
            else: ph.append([w["bbox"][0], w["bbox"][2], w["text"]])
        ln["ph"] = ph
    out.sort(key=lambda l: l["y0"])
    return out


def is_data(t): return any(p.match(t.strip()) for p in DATA)


def col_bounds(lines, x0, x1):
    """whitespace column separators (normalised to the box width); one spanning line may cross a gap when >= 4 lines"""
    if not lines or x1 <= x0: return []
    n = int(x1 - x0) + 2; occ = np.zeros(n, np.int32)
    for ln in lines:
        for a, b, _ in ln["ph"]: occ[max(0, int(a - x0)):max(0, int(b - x0)) + 1] += 1
    minw = max(4.0, 0.6 * float(np.median([l["h"] for l in lines])))
    for allowed in (0, 1):
        if allowed and len(lines) < 4: break
        seps, start = [], None
        for i, v in enumerate(occ):
            if v <= allowed and start is None: start = i
            elif v > allowed and start is not None:
                if start > 0 and i - start >= minw: seps.append((start + i) / 2.0 / (x1 - x0))
                start = None
        if seps: return seps
    return []


def bound_match(a, b, tol=0.04):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return (sum(1 for x in a if any(abs(x - y) <= tol for y in b)) + sum(1 for y in b if any(abs(x - y) <= tol for x in a))) / (len(a) + len(b))


class Frag:
    def __init__(self, doc, page, bbox, table, part, W, H, words):
        self.doc, self.page, self.bbox, self.table, self.part, self.W, self.H = doc, page, bbox, table, part, W, H
        x0, y0, x1, y1 = bbox
        ws = [w for w in words if x0 - 2 <= (w["bbox"][0] + w["bbox"][2]) / 2 <= x1 + 2 and y0 - 2 <= (w["bbox"][1] + w["bbox"][3]) / 2 <= y1 + 2]
        self.lines = lines_of(ws)
        self.rows = len(self.lines)
        self.bounds = col_bounds(self.lines, x0, x1)
        self.width = len(self.bounds) + 1
        first = self.lines[0] if self.lines else None
        self.header_text = first["text"] if first else ""
        self.header_tokens = tokens(self.header_text)
        self.first_row_tokens = tokens(self.lines[1]["text"]) if self.rows > 1 else set()
        cells = [p[2] for p in first["ph"]] if first else []
        self.headerless = bool(cells) and sum(1 for c in cells if is_data(c)) >= 0.5 * len(cells)
        self.header_orphan = (not self.headerless) and self.rows <= 1
        self.vtop, self.vbot = y0 / H, y1 / H
        self.col = 1 if x0 > 0.45 * W else 0
        self.cap = None


def caption_above(frag, page_lines):
    above = [l for l in page_lines if l["y1"] <= frag.bbox[1] + 2 and min(l["x1"], frag.bbox[2]) - max(l["x0"], frag.bbox[0]) > 0]
    for l in sorted(above, key=lambda l: -l["y1"])[:3]:
        m = CAP.match(l["text"])
        if m: return m.group(2).upper(), bool(CONT.search(l["text"])), l
    return None


def between(A, B, plines):
    """substantive lines between A and B in reading order, outside header/footer bands (1/11 page), column-aware.
    Returns (lines, continued_caption_seen)."""
    band_a, band_b = A.H / 11, B.H / 11
    la = [l for l in plines[A.page] if l["y0"] >= A.bbox[3] - 1 and l["y1"] <= A.H - band_a and min(l["x1"], A.bbox[2]) - max(l["x0"], A.bbox[0]) > 0]
    lb = [l for l in plines[B.page] if l["y1"] <= B.bbox[1] + 1 and l["y0"] >= band_b and min(l["x1"], B.bbox[2]) - max(l["x0"], B.bbox[0]) > 0]
    if A.page == B.page:
        lb = [l for l in lb if l["y0"] >= 0]
    cont = any(CAP.match(l["text"]) and CONT.search(l["text"]) for l in lb) or any(CONT.search(l["text"]) and len(l["text"]) < 40 for l in la + lb)
    sub = [l for l in la + lb if not (CONT.search(l["text"]) and (CAP.match(l["text"]) or len(l["text"]) < 40))]
    return sub, cont


def textract_links(frags_by_page, plines, accuracy):
    """returns set of (idA, idB) for cross-page links, faithful to ExecuteTableValidations (HeaderFooterType.NORMAL)"""
    out = set(); pages = sorted(frags_by_page)
    for p in pages:
        if p + 1 not in frags_by_page: continue
        A = sorted(frags_by_page[p], key=lambda f: (f.col, f.bbox[1]))[-1]; B = sorted(frags_by_page[p + 1], key=lambda f: (f.col, f.bbox[1]))[0]
        hf = A.H / 11
        if any(A.H - hf > l["y1"] > A.bbox[3] for l in plines[p]): continue
        hf = B.H / 11
        if any(hf < l["y0"] < B.bbox[1] for l in plines[p + 1]): continue
        if not (A.width == B.width or A.header_text.strip() == B.header_text.strip()): continue
        pd_ = lambda a, b: abs(100 * (a - b) / ((a + b) / 2)) if (a + b) else 0.0
        wa, wb = (A.bbox[2] - A.bbox[0]) / A.W, (B.bbox[2] - B.bbox[0]) / B.W; la, lb = A.bbox[0] / A.W, B.bbox[0] / B.W
        if pd_(wa, wb) < 100 - accuracy and pd_(la, lb) < 100 - accuracy: out.add((id(A), id(B)))
    return out


def stitcher(A, B, plines):
    """Table Stitcher _classify_sequential_pair with MultiPageConfig defaults"""
    gap = B.page - A.page
    if gap < 1 or gap > 1: return False, "page_gap_out_of_range"
    sub, _ = between(A, B, plines)
    sub = [l for l in sub if not CAP.match(l["text"])]
    if sub: return False, "content_between_tables"
    if B.headerless and B.width == 1 and A.width > 1: return True, "spillover"
    if B.header_orphan: return False, "right_header_orphan"
    wd = abs(A.width - B.width)
    if wd > 4: return False, "width_difference_too_large"
    layout = A.vbot >= 0.60 and B.vtop <= 0.40
    if A.header_orphan and B.headerless: return True, "header_orphan_to_headerless"
    if B.headerless:
        if A.width == B.width and (not A.headerless or layout): return True, "headerless_width_match"
        if wd <= 2 and layout: return True, "headerless_width_drift_layout"
        if jaccard(A.first_row_tokens, B.first_row_tokens) >= 0.3: return True, "row_similarity"
        return False, "headerless_no_signal"
    hs = jaccard(A.header_tokens, B.header_tokens)
    if hs >= 0.6:
        if A.header_tokens and B.header_tokens and (A.header_tokens - B.header_tokens) and (B.header_tokens - A.header_tokens) and not layout: return False, "strict_disjoint"
        return True, "header_similarity_strict"
    if hs >= 0.3 and layout: return True, "header_similarity_loose_layout"
    return False, "header_similarity_too_low"


def r4rule(A, B, plines):
    same_page = A.page == B.page
    if same_page:
        if not (A.col == 0 and B.col == 1 and A.vbot >= 0.6 and B.vtop <= 0.4): return False, "same_page_not_column_flow"
    elif B.page - A.page != 1:
        return False, "page_gap"
    elif not (A.vbot >= 0.55 and B.vtop <= 0.45):
        return False, "not_bottom_to_top"
    sub, cont = between(A, B, plines)
    capB = caption_above(B, plines[B.page]); capA = A.cap
    if capB and not capB[1] and (capA is None or capB[0] != capA): return False, "new_caption"
    if capB and capA and capB[0] != capA: return False, "caption_number_differs"
    if sub and not cont: return False, "content_between"
    bm = bound_match(A.bounds, B.bounds)
    geo = abs(A.width - B.width) <= 1 and (bm >= 0.5 or (abs((A.bbox[2] - A.bbox[0]) / A.W - (B.bbox[2] - B.bbox[0]) / B.W) <= 0.05 and (same_page or abs(A.bbox[0] / A.W - B.bbox[0] / B.W) <= 0.05)))
    if not geo and not cont: return False, "column_geometry"
    hs = jaccard(A.header_tokens, B.header_tokens)
    if cont: return True, "continued_marker"
    if hs >= 0.6:
        if (A.header_tokens - B.header_tokens) and (B.header_tokens - A.header_tokens) and hs < 0.9: return False, "disjoint_headers"
        return True, "repeated_header"
    if B.headerless: return True, "headerless_geometry"
    return False, "no_header_evidence"


def find(par, x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x


def split_region(lines):
    """cut positions (y) inside a region: caption line, or a large gap + new header + column geometry change / repeated header"""
    if len(lines) < 4: return []
    gaps = [lines[i + 1]["y0"] - lines[i]["y1"] for i in range(len(lines) - 1)]
    mg = float(np.median(gaps)); mh = float(np.median([l["h"] for l in lines]))
    cuts = []
    x0 = min(l["x0"] for l in lines); x1 = max(l["x1"] for l in lines)
    for i in range(1, len(lines) - 1):
        nxt = lines[i + 1]
        if CAP.match(nxt["text"]) and not CONT.search(nxt["text"]):
            cuts.append((lines[i]["y1"] + nxt["y0"]) / 2); continue
        if gaps[i] < max(1.8 * mg, 0.6 * mh) or i + 1 > len(lines) - 2: continue
        hdr = [p[2] for p in nxt["ph"]]
        new_header = len(hdr) >= 2 and sum(1 for c in hdr if is_data(c)) < 0.5 * len(hdr)
        above, below = lines[:i + 1], lines[i + 1:]
        geo = bound_match(col_bounds(above, x0, x1), col_bounds(below, x0, x1)) < 0.5
        rep = jaccard(tokens(nxt["text"]), tokens(lines[0]["text"])) >= 0.8
        if new_header and (geo or rep): cuts.append((lines[i]["y1"] + nxt["y0"]) / 2)
    return cuts


def main():
    base = fetch()
    tabs = sorted(glob.glob(os.path.join(base, "tables", "*_tables.json")))
    if LIMIT: tabs = tabs[: max(5, LIMIT)]
    rules = ["textract (accuracy 98, default)", "textract (accuracy 90)", "stitcher", "r4"]
    stats = {r: collections.Counter() for r in rules}
    assembly = {r: collections.Counter() for r in rules}
    reasons = {r: collections.Counter() for r in rules}
    pagepair = {r: collections.Counter() for r in rules}
    split = collections.Counter()
    hard = collections.Counter()
    ndocs = 0
    for tf in tabs:
        doc = os.path.basename(tf)[: -len("_tables.json")]
        T = json.load(open(tf, encoding="utf-8"))
        sizes, words, plines = {}, {}, {}
        for x in glob.glob(os.path.join(base, "xml_annotations", f"{doc}_page_*.xml")):
            pg = int(re.search(r"_page_(\d+)\.xml$", x).group(1))
            import xml.etree.ElementTree as ET
            s = ET.parse(x).getroot().find("size"); sizes[pg] = (float(s.find("width").text), float(s.find("height").text))
        for x in glob.glob(os.path.join(base, "words", f"{doc}_page_*_words.json")):
            pg = int(re.search(r"_page_(\d+)_words\.json$", x).group(1))
            words[pg] = json.load(open(x, encoding="utf-8")); plines[pg] = lines_of(words[pg])
        frags = []
        for ti, t in enumerate(T):
            for j, p in enumerate(t.get("parts") or []):
                pg = p["page_num"]
                if pg not in sizes or pg not in words: continue
                W, H = sizes[pg]
                frags.append(Frag(doc, pg, p["bbox"], ti, j, W, H, words[pg]))
        if not frags: continue
        ndocs += 1
        for f in frags:
            c = caption_above(f, plines[f.page]); f.cap = c[0] if c else None
        frags.sort(key=lambda f: (f.page, f.col, f.bbox[1]))
        idx = {id(f): k for k, f in enumerate(frags)}
        gt_links = set()
        for ti in set(f.table for f in frags):
            ps = sorted([f for f in frags if f.table == ti], key=lambda f: f.part)
            for a, b in zip(ps, ps[1:]): gt_links.add((id(a), id(b)))
        bypage = collections.defaultdict(list)
        for f in frags: bypage[f.page].append(f)
        pairs = list(zip(frags, frags[1:]))
        for rname in rules:
            if rname.startswith("textract"):
                pred = textract_links(bypage, plines, 98 if "98" in rname else 90)
            else:
                pred = set()
                for A, B in pairs:
                    ok, why = (stitcher if rname == "stitcher" else r4rule)(A, B, plines)
                    reasons[rname][why] += 1
                    if ok: pred.add((id(A), id(B)))
            st = stats[rname]
            for l in gt_links:
                A_, B_ = frags[idx[l[0]]], frags[idx[l[1]]]; kind = "same-page" if A_.page == B_.page else "cross-page"
                st["gt_" + kind] += 1
                if l in pred: st["tp_" + kind] += 1
            for l in pred:
                A_, B_ = frags[idx[l[0]]], frags[idx[l[1]]]; kind = "same-page" if A_.page == B_.page else "cross-page"
                st["pred_" + kind] += 1
                if l not in gt_links: st["fp_" + kind] += 1
            for A, B in pairs:
                if A.table != B.table:
                    st["neg"] += 1
                    hardneg = abs(A.width - B.width) <= 1 and (jaccard(A.header_tokens, B.header_tokens) >= 0.6 or (A.vbot >= 0.6 and B.vtop <= 0.4))
                    if hardneg: st["hardneg"] += 1
                    if (id(A), id(B)) in pred:
                        st["neg_merged"] += 1
                        if hardneg: st["hardneg_merged"] += 1
            par = list(range(len(frags)))
            for a, b in pred: par[find(par, idx[a])] = find(par, idx[b])
            clusters = collections.defaultdict(set)
            for k in range(len(frags)): clusters[find(par, k)].add(k)
            gtc = collections.defaultdict(set)
            for k, f in enumerate(frags): gtc[f.table].add(k)
            cl_sets = [frozenset(v) for v in clusters.values()]
            for ti, s in gtc.items():
                multi = len(s) > 1
                assembly[rname]["tables"] += 1; assembly[rname]["multi"] += multi
                if frozenset(s) in cl_sets:
                    assembly[rname]["exact"] += 1; assembly[rname]["exact_multi"] += multi
            for c in cl_sets:
                if len(set(frags[k].table for k in c)) > 1: assembly[rname]["false_merged_clusters"] += 1
            assembly[rname]["clusters"] += len(cl_sets)
            pp = [x for x in jl_pairs(base, doc)]
            for pa, pb, lab in pp:
                hit = any(frags[idx[a]].page == pa and frags[idx[b]].page == pb for a, b in pred)
                pagepair[rname]["tp" if (hit and lab) else "fp" if hit else "fn" if lab else "tn"] += 1
        # split test (same for every rule)
        for pg, fs in bypage.items():
            W, H = sizes[pg]
            for f in fs:
                if f.rows >= 4:
                    split["neg"] += 1
                    if split_region(f.lines): split["neg_split"] += 1
            fs2 = sorted(fs, key=lambda f: f.bbox[1])
            for A, B in zip(fs2, fs2[1:]):
                if A.table == B.table: continue
                ov = min(A.bbox[2], B.bbox[2]) - max(A.bbox[0], B.bbox[0])
                if ov < 0.5 * min(A.bbox[2] - A.bbox[0], B.bbox[2] - B.bbox[0]) or B.bbox[1] - A.bbox[3] > 0.12 * H or B.bbox[1] < A.bbox[3] - 2: continue
                U = [min(A.bbox[0], B.bbox[0]), A.bbox[1], max(A.bbox[2], B.bbox[2]), B.bbox[3]]
                ws = [w for w in words[pg] if U[0] - 2 <= (w["bbox"][0] + w["bbox"][2]) / 2 <= U[2] + 2 and U[1] - 2 <= (w["bbox"][1] + w["bbox"][3]) / 2 <= U[3] + 2]
                cuts = split_region(lines_of(ws))
                mh = float(np.median([l["h"] for l in A.lines + B.lines])) if (A.lines + B.lines) else 5.0
                good = [c for c in cuts if A.bbox[3] - 0.5 * mh <= c <= B.bbox[1] + 0.5 * mh]
                split["pos"] += 1; split["pos_split"] += bool(good); split["pos_exact"] += bool(good) and len(cuts) == 1
                split["pos_wrongcut"] += bool(cuts) and not good
    out = []; P = out.append
    P(f"## Gap 6: cross-page tables (PubTables-v2 Full Documents test, CDLA-Permissive-2.0): {ndocs} documents")
    P("Fragments = ground-truth table parts; decisions use only page words around the fragments. Link = consecutive parts of one logical table. "
      "False-merge rate = linked adjacent pairs that belong to different tables / all adjacent different-table pairs. Hard negatives = adjacent different-table pairs "
      "with column counts within 1 and (header similarity >= 0.6 or bottom-of-page -> top-of-page position).")
    P("\n| rule | link precision | link recall | F1 | cross-page P / R | same-page (column) P / R | false-merge rate | hard-negative false merges | exact logical tables (all) | exact (multi-part) | clusters mixing tables |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    summ = {}
    for r in rules:
        st = stats[r]; a = assembly[r]
        tp = st["tp_cross-page"] + st["tp_same-page"]; pr = st["pred_cross-page"] + st["pred_same-page"]; gt = st["gt_cross-page"] + st["gt_same-page"]
        p_, r_ = tp / max(1, pr), tp / max(1, gt); f1 = 2 * p_ * r_ / max(1e-9, p_ + r_)
        cp = lambda k: f"{st['tp_' + k] / max(1, st['pred_' + k]):.3f} / {st['tp_' + k] / max(1, st['gt_' + k]):.3f}"
        P(f"| {r} | {p_:.3f} | {r_:.3f} | {f1:.3f} | {cp('cross-page')} | {cp('same-page')} | {st['neg_merged'] / max(1, st['neg']):.4f} ({st['neg_merged']}/{st['neg']}) | "
          f"{st['hardneg_merged']}/{st['hardneg']} | {a['exact'] / max(1, a['tables']):.3f} | {a['exact_multi'] / max(1, a['multi']):.3f} ({a['exact_multi']}/{a['multi']}) | {a['false_merged_clusters']} |")
        summ[r] = dict(P=p_, R=r_, F1=f1, false_merge=st['neg_merged'] / max(1, st['neg']), exact=a['exact'] / max(1, a['tables']), exact_multi=a['exact_multi'] / max(1, a['multi']),
                       gt_links=gt, gt_cross=st["gt_cross-page"], gt_same=st["gt_same-page"], neg=st["neg"], hardneg=st["hardneg"], hardneg_merged=st["hardneg_merged"])
    P("\n**Official page-pair labels** (cross_page_table_pairs: does a table on page A continue on page B):")
    P("| rule | precision | recall | F1 | pairs |\n|---|---|---|---|---|")
    for r in rules:
        c = pagepair[r]; p_ = c["tp"] / max(1, c["tp"] + c["fp"]); r_ = c["tp"] / max(1, c["tp"] + c["fn"])
        P(f"| {r} | {p_:.3f} | {r_:.3f} | {2 * p_ * r_ / max(1e-9, p_ + r_):.3f} | {sum(c.values())} |")
        summ[r]["pagepair_F1"] = 2 * p_ * r_ / max(1e-9, p_ + r_)
    P("\nDecision reasons (adjacent pairs): " + "; ".join(f"{r}: " + ", ".join(f"{k} {v}" for k, v in reasons[r].most_common(8)) for r in ("stitcher", "r4")))
    P(f"\n**Splitting one region that holds two logical tables** (synthetic over-merge: union box of two vertically adjacent different tables in one column; "
      f"negatives = every single table part with >= 4 text lines)")
    P("| positives | split at the right place | exactly one correct cut | wrong cut only | negatives | false split rate |\n|---|---|---|---|---|---|")
    P(f"| {split['pos']} | {split['pos_split'] / max(1, split['pos']):.3f} | {split['pos_exact'] / max(1, split['pos']):.3f} | {split['pos_wrongcut']} | {split['neg']} | {split['neg_split'] / max(1, split['neg']):.4f} |")
    summ["split"] = dict(split)
    txt = "\n".join(out); print(txt)
    os.makedirs(R4OUT, exist_ok=True)
    open(os.path.join(R4OUT, "xpage.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(summ, open(os.path.join(R4OUT, "xpage_summary.json"), "w"))


_PAIRS = {}


def jl_pairs(base, doc):
    p = os.path.join(base, "cross_page_table_pairs", f"{doc}.json")
    if not os.path.exists(p): return []
    out = []
    for x in json.load(open(p)):
        pa = int(x["page_A"].rsplit("_", 1)[1]); pb = int(x["page_B"].rsplit("_", 1)[1]); out.append((pa, pb, int(x["label"])))
    return out


if __name__ == "__main__":
    main()
