import os, io, json, re, random
import pyarrow.parquet as pq
import pypdfium2 as pdfium
from PIL import Image
from paths import RAW, SETS as SETDIR
from prep_sets import tag, write_set

FILES = [os.path.join(RAW, "scitsr", "data", "test-00000-of-00001.parquet"), os.path.join(RAW, "scitsr_pd", "data", "test-00000-of-00001.parquet")]


def build(cells_in):
    cells_in = [c for c in cells_in if c["start_row"] >= 0 and c["start_col"] >= 0]
    R = max(c["end_row"] for c in cells_in) + 1; C = max(c["end_col"] for c in cells_in) + 1
    occ = {}; anchors = {}
    for k, c in enumerate(cells_in):
        r0, r1, c0, c1 = c["start_row"], c["end_row"] + 1, c["start_col"], c["end_col"] + 1
        if any((a, b) in occ for a in range(r0, r1) for b in range(c0, c1)): continue
        for a in range(r0, r1):
            for b in range(c0, c1): occ[(a, b)] = k
        anchors[(r0, c0)] = (k, r1 - r0, c1 - c0)
    toks = []; cells = []
    for a in range(R):
        toks.append("<tr>")
        for b in range(C):
            if (a, b) in anchors:
                k, rs, cs = anchors[(a, b)]
                if rs > 1 or cs > 1:
                    toks.append("<td")
                    if cs > 1: toks.append(f' colspan="{cs}"')
                    if rs > 1: toks.append(f' rowspan="{rs}"')
                    toks.append(">")
                else:
                    toks.append("<td>")
                toks.append("</td>")
                cells.append(dict(text=" ".join(cells_in[k]["content"]), bbox=[0, 0, 0, 0]))
            elif (a, b) not in occ:
                toks += ["<td>", "</td>"]; cells.append(dict(text="", bbox=[0, 0, 0, 0]))
        toks.append("</tr>")
    return toks, cells


def main():
    out = os.path.join(SETDIR, "scitsr"); os.makedirs(out, exist_ok=True)
    rows = []
    for f in FILES:
        if os.path.exists(f): rows += pq.read_table(f).to_pylist()
    print("scitsr rows", len(rows), flush=True)
    chosen = []; recs = {}
    for r in rows:
        try:
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            pdf = pdfium.PdfDocument(r["pdf"]); l, b, rr, t = pdf[0].get_cropbox()
            W, H = rr - l, t - b
            sx, sy = im.size[0] / W, im.size[1] / H
            words = [dict(text=c["text"], bbox=[(c["x1"] - l) * sx, (t - c["y2"]) * sy, (c["x2"] - l) * sx, (t - c["y1"]) * sy], gi=None) for c in r["chunks"]]
            inside = sum(1 for w_ in words if 0 <= w_["bbox"][0] <= im.size[0] and 0 <= w_["bbox"][1] <= im.size[1])
            if not inside: print("words outside image", r["table_id"], r["chunks"][:1], (l, b, rr, t), im.size)
            toks, cells = build(r["cells"])
            T = tag(dict(html=toks, cells=[[dict(tokens=list(c["text"]), bbox=c["bbox"]) for c in cells]]), im)
        except Exception as e:
            print("skip", r.get("table_id"), repr(e)[:200]); continue
        tid = "scitsr_" + re.sub(r"[^A-Za-z0-9]", "", r["table_id"])
        open(os.path.join(out, tid + ".pdf"), "wb").write(r["pdf"])
        recs[tid] = (im, dict(filename=r["table_id"], html=toks, cells=cells, words=words, pdf_scale=[sx, sy], licence=r.get("paper_license")))
        chosen.append((tid, T))
    write_set("scitsr", chosen, lambda k: (k, recs[k][0], recs[k][1]))


if __name__ == "__main__":
    main()
