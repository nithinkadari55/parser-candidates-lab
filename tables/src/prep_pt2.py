import os, sys, io, json, re, tarfile, random
import xml.etree.ElementTree as ET
from PIL import Image
from paths import RAW, SETS as SETDIR
from prep_sets import tag, stratify, write_set

D = os.path.join(RAW, "pt2")
X = os.path.join(D, "x")
TARGET = 1000


def extract():
    if os.path.isdir(os.path.join(X, "done")): return
    for p in ("tables", "words", "xml-annotations", "images"):
        with tarfile.open(os.path.join(D, f"PubTables-v2_Cropped-Tables_test_{p}.tar.gz")) as t:
            t.extractall(X)
    os.makedirs(os.path.join(X, "done"))


def find(sub):
    for root, dirs, files in os.walk(X):
        if os.path.basename(root) == sub: return root
    raise FileNotFoundError(sub)


def structure(tb):
    gt = tb["grid_top"]; gc = tb["grid_con"]; html = tb.get("html", "")
    head = html.split("</thead>")[0].count("<tr") if "</thead>" in html else 0
    toks = []; cells = []; anchor = {}
    R = len(gt)
    for r in range(R):
        if r == 0 and head: toks.append("<thead>")
        if r == head and head: toks.append("<tbody>")
        toks.append("<tr>")
        for c in range(len(gt[r])):
            dx, dy, w, h = gt[r][c]
            if dx == 0 and dy == 0:
                if w > 1 or h > 1:
                    toks.append("<td")
                    if w > 1: toks.append(f' colspan="{w}"')
                    if h > 1: toks.append(f' rowspan="{h}"')
                    toks.append(">")
                else:
                    toks.append("<td>")
                toks.append("</td>")
                anchor[(r, c)] = len(cells)
                cells.append(dict(text=gc[r][c] or "", bbox=None))
        toks.append("</tr>")
        if r == head - 1 and head: toks.append("</thead>")
    if head: toks.append("</tbody>")
    owner = {}
    for r in range(R):
        for c in range(len(gt[r])):
            dx, dy, w, h = gt[r][c]
            owner[(r, c)] = anchor.get((r + dy, c + dx))
    return toks, cells, owner


def boxes(xmlp, label):
    out = []
    for o in ET.parse(xmlp).getroot().iter("object"):
        if o.find("name").text == label:
            b = o.find("bndbox"); out.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def assign(words, rows, cols, owner):
    rows = sorted(rows, key=lambda b: b[1]); cols = sorted(cols, key=lambda b: b[0])
    out = []
    for w in words:
        cx, cy = (w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2
        ri = next((i for i, b in enumerate(rows) if b[1] <= cy <= b[3]), None)
        ci = next((i for i, b in enumerate(cols) if b[0] <= cx <= b[2]), None)
        gi = owner.get((ri, ci)) if ri is not None and ci is not None else None
        out.append(dict(text=w["text"], bbox=list(w["bbox"]), gi=gi))
    return out


def load(name, dirs):
    tb = json.load(open(os.path.join(dirs["tables"], name + ".json"), encoding="utf-8"))[0]
    toks, cells, owner = structure(tb)
    words = json.load(open(os.path.join(dirs["words"], name + "_words.json"), encoding="utf-8"))
    xmlp = os.path.join(dirs["xml_annotations"], name + ".xml")
    rows, cols = boxes(xmlp, "table row"), boxes(xmlp, "table column")
    ok = len(rows) == len(tb["grid_top"]) and len(cols) == max(len(r) for r in tb["grid_top"])
    ws = assign(words, rows, cols, owner) if ok else [dict(text=w["text"], bbox=list(w["bbox"]), gi=None) for w in words]
    for w in ws:
        if w["gi"] is None: continue
        c = cells[w["gi"]]; b = w["bbox"]
        c["bbox"] = b if c["bbox"] is None else [min(c["bbox"][0], b[0]), min(c["bbox"][1], b[1]), max(c["bbox"][2], b[2]), max(c["bbox"][3], b[3])]
    for c in cells:
        if c["bbox"] is None: c["bbox"] = [0, 0, 0, 0]
    return toks, cells, ws, ok


def main():
    extract()
    dirs = {k: find(k) for k in ("tables", "words", "xml_annotations", "images")}
    names = sorted(f[:-5] for f in os.listdir(dirs["tables"]) if f.endswith(".json"))
    print("pt2 tables", len(names), flush=True)
    pool = []; cache = {}
    for n in names:
        try:
            toks, cells, ws, ok = load(n, dirs)
            im = Image.open(os.path.join(dirs["images"], n + ".jpg")).convert("RGB")
            T = tag(dict(html=toks, cells=[[dict(tokens=list(c["text"]), bbox=c["bbox"]) for c in cells]]), im, dict(words_aligned=ok))
        except Exception as e:
            print("skip", n, repr(e)[:200]); continue
        pool.append((n, T))
    chosen = stratify(pool, TARGET, random.Random(42))

    def get(n):
        toks, cells, ws, ok = load(n, dirs)
        im = Image.open(os.path.join(dirs["images"], n + ".jpg")).convert("RGB")
        return "pt2_" + re.sub(r"[^A-Za-z0-9]", "", n), im, dict(filename=n, html=toks, cells=cells, words=ws)
    write_set("pt2", chosen, get, jpeg=True)


if __name__ == "__main__":
    main()
