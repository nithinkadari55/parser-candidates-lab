import sys, os, io, json, glob, re, random
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from paths import RAW, SETS as SETDIR
SETS = {
    "ptn": (glob.glob(os.path.join(RAW, "ptn", "data", "val-*.parquet")), 2000),
    "ftn": (glob.glob(os.path.join(RAW, "ftn", "data", "test-*.parquet")), 2000),
    "pt1m": (glob.glob(os.path.join(RAW, "pt1m", "data", "test-*.parquet")), 2000),
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


def tag(rec, im, extra=None):
    rows, thead = parse_html(rec["html"])
    pos, R, C = place(rows)
    cells = rec["cells"][0]
    texts = ["".join(c["tokens"]).strip() for c in cells]
    nonempty = [i for i, t in enumerate(texts) if t]
    nspan = sum(1 for (r0, r1, c0, c1) in pos if r1 - r0 > 1 or c1 - c0 > 1)
    nh, nv = rules(im) if im is not None else (0, 0)
    T = {}
    T["rows"], T["cols"], T["thead_rows"], T["n_span"], T["h_rules"], T["v_rules"] = R, C, thead, nspan, nh, nv
    T["ruled"] = nv >= max(1, 0.6 * (C - 1)) and nh >= 0.6 * (R - 1)
    T["borderless"] = nv == 0 and nh <= 4
    T["multirow_header"] = thead >= 2
    T["nested_header"] = thead >= 2 and any(c1 - c0 > 1 for (r0, r1, c0, c1) in pos if r0 < thead)
    body_first_only = 0; indents = []
    byrow = {}
    for i, (r0, r1, c0, c1) in enumerate(pos):
        byrow.setdefault(r0, []).append((c0, c1, i))
    for r, lst in byrow.items():
        if r < thead: continue
        ne = [(c0, c1) for c0, c1, i in lst if i < len(texts) and texts[i]]
        if len(ne) == 1 and ne[0][0] == 0 and C > 2: body_first_only += 1
        for c0, c1, i in lst:
            if c0 == 0 and i < len(texts) and texts[i] and len(cells[i].get("bbox") or []) >= 4: indents.append(cells[i]["bbox"][0])
    lvl = 0
    if len(indents) >= 3:
        m = min(indents); lvl = sum(1 for x in indents if x - m >= 6)
    T["hier_first_col"] = body_first_only >= 2 or (lvl >= 2 and body_first_only >= 1) or lvl >= 3
    T["wide"] = C >= 10
    T["long"] = R >= 30
    T["very_wide"] = C >= 15
    T["very_long"] = R >= 50
    T["many_spans"] = nspan >= 3
    hs = [cells[i]["bbox"][3] - cells[i]["bbox"][1] for i in nonempty if len(cells[i].get("bbox") or []) >= 4 and cells[i]["bbox"][3] > cells[i]["bbox"][1]]
    med = float(np.median(hs)) if hs else 0
    T["multiline"] = med > 0 and any(h >= 1.7 * med for h in hs)
    T["empty_cells"] = len(cells) > 0 and (len(cells) - len(nonempty)) / len(cells) >= 0.15
    T["footnote_marks"] = any("<sup>" in "".join(cells[i]["tokens"]) or FOOT.search(texts[i]) for i in nonempty)
    T["tiny"] = R <= 3
    T["rotated"] = False
    T["split_page"] = False
    if extra: T.update(extra)
    T["bucket"] = bucket(T)
    return T


def bucket(T):
    if T["very_long"] or T["very_wide"] or T["tiny"] or T["rotated"] or T["split_page"]: return "edge"
    if T["many_spans"] or T["nested_header"] or T["hier_first_col"] or T["wide"] or T["long"]: return "complex"
    if T["multiline"] or T["empty_cells"] or T["footnote_marks"]: return "edge"
    if T["ruled"] and not T["multirow_header"]: return "simple"
    return "medium"


SHARE = {"simple": 0.15, "medium": 0.20, "complex": 0.35, "edge": 0.30}
RARE = ["very_long", "very_wide", "tiny", "rotated", "split_page"]


def stratify(pool, target, rng):
    by = {b: [x for x in pool if x[1]["bucket"] == b] for b in SHARE}
    for v in by.values(): rng.shuffle(v)
    quota = {b: int(round(target * f)) for b, f in SHARE.items()}
    for b in ("simple", "medium"):
        short = max(0, quota[b] - len(by[b]))
        quota[b] -= short
        quota["complex"] += short // 2; quota["edge"] += short - short // 2
    chosen = {b: [] for b in SHARE}
    rare = [x for x in by["edge"] if any(x[1].get(t) for t in RARE)]
    common = [x for x in by["edge"] if not any(x[1].get(t) for t in RARE)]
    per_tag = max(1, quota["edge"] // 4)
    taken = set()
    for t in RARE:
        lst = [x for x in rare if x[1].get(t) and x[0] not in taken][:per_tag]
        for x in lst:
            if len(chosen["edge"]) < quota["edge"]: chosen["edge"].append(x); taken.add(x[0])
    for x in common + rare:
        if len(chosen["edge"]) >= quota["edge"]: break
        if x[0] not in taken: chosen["edge"].append(x); taken.add(x[0])
    for b in ("simple", "medium", "complex"):
        chosen[b] = by[b][:quota[b]]
    short = target - sum(len(v) for v in chosen.values())
    for b in ("complex", "edge", "medium", "simple"):
        rest = [x for x in by[b] if x[0] not in {y[0] for y in chosen[b]}]
        while short > 0 and rest:
            chosen[b].append(rest.pop()); short -= 1
    return [x for b in SHARE for x in chosen[b]]


def crop_rec(r):
    im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
    cells = [dict(text="".join(c["tokens"]), bbox=list(c["bbox"][:4])) for c in r["cells"][0]]
    if r.get("table_bbox"):
        W, Hh = im.size; x0, y0, x1, y1 = r["table_bbox"]
        x0, y0, x1, y1 = max(0, x0 - 4), max(0, y0 - 4), min(W, x1 + 4), min(Hh, y1 + 4)
        im = im.crop((x0, y0, x1, y1))
        for c in cells: c["bbox"] = [c["bbox"][0] - x0, c["bbox"][1] - y0, c["bbox"][2] - x0, c["bbox"][3] - y0]
    return im, cells


def write_set(name, chosen, get, jpeg=False):
    out = os.path.join(SETDIR, name); os.makedirs(out, exist_ok=True)
    index = []
    for key, T in chosen:
        tid, im, gt = get(key)
        gt["tags"] = T; gt["id"] = tid
        if jpeg: im.save(os.path.join(out, tid + ".png"), format="JPEG", quality=92)
        else: im.save(os.path.join(out, tid + ".png"))
        json.dump(gt, open(os.path.join(out, tid + ".json"), "w", encoding="utf-8"))
        index.append(tid)
    json.dump(sorted(index), open(os.path.join(out, "index.json"), "w"))
    import collections
    print(name, "chosen", len(index), dict(collections.Counter(T["bucket"] for _, T in chosen)), flush=True)


def main(name):
    files, target = SETS[name]
    allrecs = []
    for f in sorted(files):
        allrecs += pq.read_table(f).to_pylist()
    print(name, "total", len(allrecs), flush=True)
    pool = []
    for idx, r in enumerate(allrecs):
        try:
            im, cells = crop_rec(r)
            T = tag(dict(html=r["html"], cells=[[dict(tokens=list(c["text"]), bbox=c["bbox"]) for c in cells]]), im)
        except Exception as e:
            print("skip", idx, e); continue
        pool.append((idx, T))
    import collections
    print(name, "pool", dict(collections.Counter(T["bucket"] for _, T in pool)), flush=True)
    chosen = stratify(pool, target, random.Random(42))

    def get(idx):
        r = allrecs[idx]; im, cells = crop_rec(r)
        return f"{name}_{idx:05d}", im, dict(filename=r["filename"], html=r["html"], cells=cells, table_bbox=r.get("table_bbox"))
    write_set(name, chosen, get)


if __name__ == "__main__":
    main(sys.argv[1])
