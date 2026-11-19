import os, io, json, re, random, zipfile
from PIL import Image, ImageOps
from paths import RAW, SETS as SETDIR
from prep_sets import tag, stratify, write_set

Z = os.path.join(RAW, "tabrec", "tabrecset.zip")
ROOT = "TabRecSet (CurveTabSet)/"
LABEL = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)-(.*)$", re.S)
PER_TYPE = 200


def parse(ann):
    shapes = [s for s in ann.get("shapes", []) if s.get("group_id", 0) in (0, None)]
    cells = []
    for s in shapes:
        m = LABEL.match(s.get("label", ""))
        if not m: continue
        r, c, rs, cs = (int(m.group(i)) for i in range(1, 5))
        xs = [p[0] for p in s["points"]]; ys = [p[1] for p in s["points"]]
        cells.append(dict(r=r, c=c, rs=rs, cs=cs, text=m.group(5).replace("\n", " ").strip(), box=[min(xs), min(ys), max(xs), max(ys)]))
    if not cells: return None
    r0 = min(x["r"] for x in cells); c0 = min(x["c"] for x in cells)
    for x in cells: x["r"] -= r0; x["c"] -= c0
    return cells


def build(cells, ox, oy):
    R = max(x["r"] + x["rs"] for x in cells); C = max(x["c"] + x["cs"] for x in cells)
    occ = {}; anchors = {}
    for k, x in enumerate(cells):
        if any((a, b) in occ for a in range(x["r"], x["r"] + x["rs"]) for b in range(x["c"], x["c"] + x["cs"])): continue
        for a in range(x["r"], x["r"] + x["rs"]):
            for b in range(x["c"], x["c"] + x["cs"]): occ[(a, b)] = k
        anchors[(x["r"], x["c"])] = k
    toks = []; out = []
    for a in range(R):
        toks.append("<tr>")
        for b in range(C):
            if (a, b) in anchors:
                x = cells[anchors[(a, b)]]
                if x["rs"] > 1 or x["cs"] > 1:
                    toks.append("<td")
                    if x["cs"] > 1: toks.append(f' colspan="{x["cs"]}"')
                    if x["rs"] > 1: toks.append(f' rowspan="{x["rs"]}"')
                    toks.append(">")
                else:
                    toks.append("<td>")
                toks.append("</td>")
                bx = x["box"]; out.append(dict(text=x["text"], bbox=[bx[0] - ox, bx[1] - oy, bx[2] - ox, bx[3] - oy]))
            elif (a, b) not in occ:
                toks += ["<td>", "</td>"]; out.append(dict(text="", bbox=[0, 0, 0, 0]))
        toks.append("</tr>")
    return toks, out


def main():
    rng = random.Random(42)
    with zipfile.ZipFile(Z) as z:
        names = z.namelist()
        anns = {os.path.basename(n)[:-5]: n for n in names if n.startswith(ROOT + "TSR_TCR_annotation/english/") and n.endswith(".json")}
        chosen = []; recs = {}; used = set()
        for kind in ("english_all-line", "english_no-line", "english_three-line"):
            ims = sorted(n for n in names if n.startswith(ROOT + f"image/{kind}/") and not n.endswith("/"))
            rng.shuffle(ims)
            pool = []
            for n in ims:
                base = os.path.splitext(os.path.basename(n))[0]
                if base not in anns or base in used: continue
                try:
                    cells = parse(json.loads(z.read(anns[base])))
                    if not cells: continue
                    im = ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(n)))).convert("RGB")
                    x0 = max(0, min(c["box"][0] for c in cells) - 10); y0 = max(0, min(c["box"][1] for c in cells) - 10)
                    x1 = min(im.size[0], max(c["box"][2] for c in cells) + 10); y1 = min(im.size[1], max(c["box"][3] for c in cells) + 10)
                    im = im.crop((int(x0), int(y0), int(x1), int(y1)))
                    toks, out = build(cells, int(x0), int(y0))
                    T = tag(dict(html=toks, cells=[[dict(tokens=list(c["text"]), bbox=c["bbox"]) for c in out]]), im, dict(line_type=kind.split("_")[1]))
                except Exception as e:
                    print("skip", n, repr(e)[:160]); continue
                tid = "tabrec_" + re.sub(r"[^A-Za-z0-9]", "", base) + kind.split("_")[1].replace("-", "")
                recs[tid] = (n, (int(x0), int(y0), int(x1), int(y1)), dict(filename=n, html=toks, cells=out), base)
                pool.append((tid, T))
                if len(pool) >= PER_TYPE * 4: break
            pick = stratify(pool, PER_TYPE, rng)
            for tid, _ in pick: used.add(recs[tid][3])
            chosen += pick
            print(kind, "pool", len(pool), "picked", len(pick), flush=True)
        def get(k):
            n, box, gt, _ = recs[k]
            return k, ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(n)))).convert("RGB").crop(box), gt
        write_set("tabrec", chosen, get, jpeg=True)


if __name__ == "__main__":
    main()
