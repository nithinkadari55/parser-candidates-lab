import sys, os, io, json, glob, re, random
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from paths import RAW, SETS as SETDIR
SETS = {
    "ptn": (glob.glob(os.path.join(RAW, "ptn", "data", "val-*.parquet")), 600),
    "ftn": (glob.glob(os.path.join(RAW, "ftn", "data", "test-*.parquet")), 600),
    "pt1m": (glob.glob(os.path.join(RAW, "pt1m", "data", "test-*.parquet")), 400),
}
FOOT = re.compile(r"(<sup>|[\*\u2020\u2021\u00a7\u00b6]|\([a-z]\)$|\(\d\)$|\[\d+\]$|[a-z\d][\u00b9\u00b2\u00b3\u2070-\u2079])")


def parse_html(tokens):
    rows = []; thead_rows = 0; inhead = False; i = 0; cur = None
    while i < len(tokens):
        t = tokens[i]
        if t == "<thead>": inhead = True
        elif t == "</thead>": inhead = False
        elif t == "<tr>": cur = []; rows.append(cur); thead_rows += inhead
        elif t in ("<td>", "<td"):
            rs = cs = 1
            if t == "<td":
                i += 1
                while tokens[i] != ">":
                    m = re.search(r'(rowspan|colspan)="(\d+)"', tokens[i])
                    if m:
                        if m.group(1) == "rowspan": rs = int(m.group(2))
                        else: cs = int(m.group(2))
                    i += 1
            cur.append((rs, cs))
        i += 1
    return rows, thead_rows


def place(rows):
    occ = {}; cells = []
    for r, row in enumerate(rows):
        c = 0
        for rs, cs in row:
            while (r, c) in occ: c += 1
            for dr in range(rs):
                for dc in range(cs): occ[(r + dr, c + dc)] = len(cells)
            cells.append((r, r + rs, c, c + cs))
            c += cs
    R = max(k[0] for k in occ) + 1 if occ else 0
    C = max(k[1] for k in occ) + 1 if occ else 0
    return cells, R, C


def rules(im):
    g = np.asarray(im.convert("L")) < 160
    H, W = g.shape

    def longest(a):
        best = 0
        for line in a:
            d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
            s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
            if len(s): best = max(best, int((e - s).max()))
        return best

    def count(mask_len, L, frac):
        hits = [i for i, v in enumerate(mask_len) if v >= frac * L]
        n = 0; last = -10
        for h in hits:
            if h - last > 2: n += 1
            last = h
        return n
    hl = [longest(g[y:y + 1]) for y in range(H)]
    vl = [longest(g[:, x:x + 1].T) for x in range(W)]
    return count(hl, W, 0.4), count(vl, H, 0.4)


def tag(rec, im):
    rows, thead = parse_html(rec["html"])
    pos, R, C = place(rows)
    cells = rec["cells"][0]
    texts = ["".join(c["tokens"]).strip() for c in cells]
    nonempty = [i for i, t in enumerate(texts) if t]
    nspan = sum(1 for (r0, r1, c0, c1) in pos if r1 - r0 > 1 or c1 - c0 > 1)
    nh, nv = rules(im)
    T = {}
    T["rows"], T["cols"], T["thead_rows"], T["n_span"], T["h_rules"], T["v_rules"] = R, C, thead, nspan, nh, nv
    T["ruled"] = nv >= max(1, 0.6 * (C - 1)) and nh >= 0.6 * (R - 1)
    T["borderless"] = nv == 0 and nh <= 4
    T["multirow_header"] = thead >= 2
    T["nested_header"] = thead >= 2 and any(c1 - c0 > 1 for (r0, r1, c0, c1) in pos if r0 < thead)
    body_first_only = 0; fullrow = 0; indents = []
    byrow = {}
    for i, (r0, r1, c0, c1) in enumerate(pos):
        byrow.setdefault(r0, []).append((c0, c1, i))
    for r, lst in byrow.items():
        if r < thead: continue
        ne = [(c0, c1) for c0, c1, i in lst if texts[i]]
        if len(ne) == 1 and ne[0][0] == 0 and C > 2:
            body_first_only += 1
            if ne[0][1] - ne[0][0] >= C - 1: fullrow += 1
        for c0, c1, i in lst:
            if c0 == 0 and texts[i]: indents.append(cells[i]["bbox"][0])
    lvl = 0
    if len(indents) >= 3:
        m = min(indents); lvl = sum(1 for x in indents if x - m >= 6)
    T["hier_first_col"] = body_first_only >= 2 or (lvl >= 2 and body_first_only >= 1) or lvl >= 3
    T["wide"] = C >= 10
    T["long"] = R >= 30
    T["many_spans"] = nspan >= 3
    hs = [cells[i]["bbox"][3] - cells[i]["bbox"][1] for i in nonempty if len(cells[i]["bbox"]) >= 4]
    med = float(np.median(hs)) if hs else 0
    T["multiline"] = med > 0 and any(h >= 1.7 * med for h in hs)
    T["empty_cells"] = len(cells) > 0 and (len(cells) - len(nonempty)) / len(cells) >= 0.15
    T["footnote_marks"] = any("<sup>" in "".join(cells[i]["tokens"]) or FOOT.search(texts[i]) for i in nonempty)
    T["tiny"] = R <= 3
    complex_ = T["many_spans"] or T["nested_header"] or T["hier_first_col"] or T["wide"] or T["long"]
    edge = T["multiline"] or T["empty_cells"] or T["footnote_marks"] or T["tiny"]
    if complex_: b = "complex"
    elif edge: b = "edge"
    elif T["ruled"] and not T["multirow_header"]: b = "simple"
    else: b = "medium"
    T["bucket"] = b
    return T


def main(name):
    files, target = SETS[name]
    out = os.path.join(SETDIR, name); os.makedirs(out, exist_ok=True)
    allrecs = []
    for f in sorted(files):
        t = pq.read_table(f)
        for r in t.to_pylist():
            allrecs.append(r)
    print(name, "total", len(allrecs), flush=True)
    rng = random.Random(42)
    order = list(range(len(allrecs))); rng.shuffle(order)
    pool = {"simple": [], "medium": [], "edge": [], "complex": []}
    per = target // 4
    scanned = 0
    for idx in order:
        if all(len(v) >= per * 3 for v in pool.values()) or scanned >= 100000: break
        r = allrecs[idx]
        try:
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            T = tag(r, im)
        except Exception as e:
            print("skip", idx, e); continue
        scanned += 1
        pool[T["bucket"]].append((idx, T))
    print(name, "scanned", scanned, {k: len(v) for k, v in pool.items()})
    chosen = {k: v[:per] for k, v in pool.items()}
    short = target - sum(len(v) for v in chosen.values())
    while short > 0:
        added = False
        for k in ["medium", "edge", "complex", "simple"]:
            if short > 0 and len(chosen[k]) < len(pool[k]):
                chosen[k].append(pool[k][len(chosen[k])]); short -= 1; added = True
        if not added: break
    index = []
    for k, lst in chosen.items():
        for idx, T in lst:
            r = allrecs[idx]
            tid = f"{name}_{idx:05d}"
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            cells = [dict(text="".join(c["tokens"]), bbox=c["bbox"][:4]) for c in r["cells"][0]]
            if r.get("table_bbox"):
                W, Hh = im.size; x0, y0, x1, y1 = r["table_bbox"]
                x0, y0, x1, y1 = max(0, x0 - 4), max(0, y0 - 4), min(W, x1 + 4), min(Hh, y1 + 4)
                im = im.crop((x0, y0, x1, y1))
                for c in cells:
                    if c["text"].strip(): c["bbox"] = [c["bbox"][0] - x0, c["bbox"][1] - y0, c["bbox"][2] - x0, c["bbox"][3] - y0]
            im.save(os.path.join(out, tid + ".png"))
            gt = dict(id=tid, filename=r["filename"], html=r["html"], cells=cells, tags=T, table_bbox=r.get("table_bbox"))
            json.dump(gt, open(os.path.join(out, tid + ".json"), "w", encoding="utf-8"))
            index.append(tid)
    json.dump(sorted(index), open(os.path.join(out, "index.json"), "w"))
    print(name, "chosen", {k: len(v) for k, v in chosen.items()})


if __name__ == "__main__":
    main(sys.argv[1])
