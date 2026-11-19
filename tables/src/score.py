import sys, os, json, re, time, html as H, argparse
GRITS_MAX = 600
from multiprocessing import Pool
import paths
from paths import SETS
from prep_sets import parse_html, place
TAGRE = re.compile(r"<[^>]+>")
TAGS = ["ruled", "borderless", "multirow_header", "nested_header", "hier_first_col", "wide", "long", "many_spans", "multiline", "empty_cells", "footnote_marks", "tiny", "split_page", "rotated"]


def gt_html(gt, plain=False):
    toks = gt["html"]; out = []; ci = 0
    for t in toks:
        if t in ("<thead>", "</thead>", "<tbody>", "</tbody>"): continue
        if t == "</td>":
            txt = gt["cells"][ci]["text"] if ci < len(gt["cells"]) else ""
            ci += 1
            out.append(H.escape(TAGRE.sub("", txt), quote=False) if plain else txt)
        out.append(t)
    return "<html><body><table>" + "".join(out) + "</table></body></html>"


def pred_html(R, C, cells, texts, plain=False):
    occ = {}; anchor = {}
    for k, c in enumerate(cells):
        r0, r1, c0, c1 = c["r0"], c["r1"], c["c0"], c["c1"]
        if r1 <= r0 or c1 <= c0: continue
        if any((r, cc) in occ for r in range(r0, r1) for cc in range(c0, c1)):
            free = [(r, cc) for r in range(r0, r1) for cc in range(c0, c1) if (r, cc) not in occ]
            if not free: continue
            r0, c0 = free[0]; r1, c1 = r0 + 1, c0 + 1
        for r in range(r0, r1):
            for cc in range(c0, c1): occ[(r, cc)] = k
        anchor[(r0, c0)] = (k, r1 - r0, c1 - c0)
    out = []
    for r in range(R):
        out.append("<tr>")
        for cc in range(C):
            if (r, cc) in anchor:
                k, rs, cs = anchor[(r, cc)]
                a = ""
                if cs > 1: a += f' colspan="{cs}"'
                if rs > 1: a += f' rowspan="{rs}"'
                t = texts.get(k, "")
                if plain: t = H.escape(TAGRE.sub("", t), quote=False)
                out.append(f"<td{a}>{t}</td>")
            elif (r, cc) not in occ:
                out.append("<td></td>")
        out.append("</tr>")
    return "<html><body><table>" + "".join(out) + "</table></body></html>"


def gt_chunks(gt):
    if "words" in gt: return [dict(text=w["text"], bbox=w["bbox"], gi=w.get("gi")) for w in gt["words"]]
    return [dict(text=c["text"], bbox=c["bbox"][:4], gi=i) for i, c in enumerate(gt["cells"]) if c["text"].strip()]


def fill(cells, ch):
    texts = {}; members = {}
    for t in ch:
        tb = t["bbox"]; ta = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1])); best = None; bv = 0
        for k, c in enumerate(cells):
            b = c["bbox"]
            ix = max(0, min(b[2], tb[2]) - max(b[0], tb[0])); iy = max(0, min(b[3], tb[3]) - max(b[1], tb[1]))
            v = ix * iy / ta
            if v > bv: bv = v; best = k
        if best is not None: members.setdefault(best, []).append(t)
    for k, lst in members.items():
        lst.sort(key=lambda t: (round(t["bbox"][1] / 4), t["bbox"][0]))
        texts[k] = " ".join(t["text"] for t in lst)
    return texts, members


def score_one(args):
    gt, rec = args
    from metric import TEDS
    import distance as _d
    from rapidfuzz.distance import Levenshtein as _L
    _d.levenshtein = _L.distance
    import grits
    rows, thead = parse_html(gt["html"]); pos, GR, GC = place(rows)
    cells = rec.get("cells") or []
    res = dict(id=gt["id"], bucket=gt["tags"]["bucket"], tags={k: bool(gt["tags"].get(k)) for k in TAGS})
    res["ms_model"] = rec.get("ms_model"); res["ms_total"] = rec.get("ms_total"); res["err"] = rec.get("err")
    res["gt_rows"], res["gt_cols"] = GR, GC
    if not cells:
        res.update(teds=0.0, teds_s=0.0, grits_top=0.0, grits_con=0.0, rows=0, cols=0, rc_exact=False, merged_row_err=0, header_ok=(None if not thead else False), fail=True, n_cells=0)
        return res
    R = max(c["r1"] for c in cells); C = max(c["c1"] for c in cells)
    ch = gt_chunks(gt)
    texts, members = fill(cells, ch)
    ph = pred_html(R, C, cells, texts); gh = gt_html(gt)
    tf = TEDS(structure_only=False, ignore_nodes=["thead", "tbody"]); ts = TEDS(structure_only=True, ignore_nodes=["thead", "tbody"])
    t1 = time.time(); res["teds"] = tf.evaluate(ph, gh); res["teds_s"] = ts.evaluate(ph, gh); res["teds_sec"] = round(time.time() - t1, 2)
    try:
        if GR * GC > GRITS_MAX or R * C > GRITS_MAX: raise ValueError("grid too large for GriTS cap")
        g = grits.grits_from_html(gt_html(gt, True)[12:-14], pred_html(R, C, cells, texts, True)[12:-14])
        res["grits_top"] = float(g["grits_top"]); res["grits_con"] = float(g["grits_con"])
    except Exception as e:
        res["grits_top"] = res["grits_con"] = None; res["grits_err"] = repr(e)[:200]
    res["rows"], res["cols"] = R, C
    res["rc_exact"] = (R == GR and C == GC)
    merged = 0
    for k, lst in members.items():
        rr = set()
        for t in lst:
            gi = t.get("gi")
            if gi is None or gi >= len(pos): continue
            r0, r1, c0, c1 = pos[gi]
            if r1 - r0 == 1: rr.add(r0)
        if len(rr) >= 2:
            merged += 1

    res["merged_row_err"] = merged
    if thead:
        prow = set()
        for c in cells:
            if c.get("header"): prow.update(range(c["r0"], c["r1"]))
        res["header_ok"] = len(prow) == thead and prow == set(range(thead))
        res["pred_header_rows"] = len(prow); res["gt_header_rows"] = thead
    else:
        res["header_ok"] = None
    res["n_cells"] = len(cells)
    res["fail"] = res["teds_s"] < 0.5
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    for f in a.files:
        base = os.path.basename(f)[:-6]
        setname = base.split("_")[1]
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        d = os.path.join(SETS, setname)
        jobs = [(json.load(open(os.path.join(d, r["id"] + ".json"), encoding="utf-8")), r) for r in recs]
        of = f[:-6] + ".scored.jsonl"; tmp = of + ".part"
        done = {}
        if os.path.exists(tmp):
            for l in open(tmp, encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(l); done[r["id"]] = r
                except Exception:
                    pass
        todo = [j for j in jobs if j[1]["id"] not in done]
        todo.sort(key=lambda j: len(j[0]["cells"]))
        if todo:
            with Pool(a.jobs) as p, open(tmp, "a", encoding="utf-8") as g:
                for r in p.imap_unordered(score_one, todo, chunksize=1):
                    g.write(json.dumps(r) + "\n"); g.flush(); done[r["id"]] = r
        out = [done[r["id"]] for r in recs]
        with open(of, "w", encoding="utf-8") as g:
            for r in out: g.write(json.dumps(r) + "\n")
        if os.path.exists(tmp): os.remove(tmp)
        n = len(out)
        if not n: continue
        print(base, n, "TEDS %.3f TEDS-S %.3f rc_exact %.1f%% fail %d" % (sum(r["teds"] for r in out) / n, sum(r["teds_s"] for r in out) / n, 100 * sum(r["rc_exact"] for r in out) / n, sum(r["fail"] for r in out)))


if __name__ == "__main__":
    main()
