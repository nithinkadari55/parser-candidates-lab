import sys, os, json, glob, io
import xml.etree.ElementTree as ET
import pypdfium2 as pdfium, pdfplumber
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RAW, SETS as SETDIR
from prep_sets import tag
SRC = os.path.join(RAW, "icdar2013", "x")
OUT = os.path.join(SETDIR, "icdar"); os.makedirs(OUT, exist_ok=True)
S = 2.0; M = 4.0
import re
def num(s): return float(re.match(r'-?[0-9.]+', s.strip()).group(0))


def main():
    index = []; skipped = []
    for strf in sorted(glob.glob(os.path.join(SRC, "*", "*-str.xml"))):
        base = strf[:-8]; pdf = base + ".pdf"; regf = base + "-reg.xml"
        if not os.path.exists(pdf) or not os.path.exists(regf): skipped.append((strf, "missing pdf/reg")); continue
        st = ET.parse(strf).getroot(); rg = ET.parse(regf).getroot()
        regbox = {}
        for t in rg.iter("table"):
            for r in t.iter("region"):
                b = r.find("bounding-box")
                regbox[(t.get("id"), r.get("id"))] = (int(r.get("page")), [num(b.get(k)) for k in ("x1", "y1", "x2", "y2")])
        doc = pdfium.PdfDocument(pdf); plb = pdfplumber.open(pdf)
        for t in st.iter("table"):
            regions = list(t.iter("region"))
            for r in regions:
                key = (t.get("id"), r.get("id"))
                tid = f"icdar_{os.path.basename(base)}_t{key[0]}r{key[1]}"
                if key not in regbox: skipped.append((tid, "no region box")); continue
                page, (x1, y1, x2, y2) = regbox[key]
                pp = plb.pages[page - 1]
                if pp.rotation: skipped.append((tid, "rotated page")); continue
                W, Hh = float(pp.width), float(pp.height)
                cells = []
                for c in r.iter("cell"):
                    sr = int(c.get("start-row")); sc = int(c.get("start-col"))
                    er = int(c.get("end-row", sr)); ec = int(c.get("end-col", sc))
                    b = c.find("bounding-box"); cont = c.find("content")
                    bb = [num(b.get(k)) for k in ("x1", "y1", "x2", "y2")] if b is not None else None
                    cells.append(dict(sr=sr, er=er, sc=sc, ec=ec, bb=bb, text=(cont.text or "").replace("\n", " ").strip() if cont is not None else ""))
                if not cells: skipped.append((tid, "no cells")); continue
                r0 = min(c["sr"] for c in cells); c0 = min(c["sc"] for c in cells)
                R = max(c["er"] for c in cells) - r0 + 1; C = max(c["ec"] for c in cells) - c0 + 1
                L, T = x1 - M, Hh - y2 - M; Rr, B = x2 + M, Hh - y1 + M
                img = doc[page - 1].render(scale=S).to_pil().convert("RGB")
                im = img.crop((int(L * S), int(T * S), int(Rr * S), int(B * S)))

                def tobox(bb):
                    return [(bb[0] - L) * S, (Hh - bb[3] - T) * S, (bb[2] - L) * S, (Hh - bb[1] - T) * S]
                occ = {}; anchors = {}
                for k, c in enumerate(cells):
                    rr0, cc0 = c["sr"] - r0, c["sc"] - c0; rr1, cc1 = c["er"] - r0 + 1, c["ec"] - c0 + 1
                    if any((a, b_) in occ for a in range(rr0, rr1) for b_ in range(cc0, cc1)): continue
                    for a in range(rr0, rr1):
                        for b_ in range(cc0, cc1): occ[(a, b_)] = k
                    anchors[(rr0, cc0)] = (k, rr1 - rr0, cc1 - cc0)
                html = []; gcells = []
                for a in range(R):
                    html.append("<tr>")
                    for b_ in range(C):
                        if (a, b_) in anchors:
                            k, rs, cs = anchors[(a, b_)]
                            if rs > 1 or cs > 1:
                                html.append("<td")
                                if cs > 1: html.append(f' colspan="{cs}"')
                                if rs > 1: html.append(f' rowspan="{rs}"')
                                html.append(">")
                            else: html.append("<td>")
                            html.append("</td>")
                            c = cells[k]
                            gcells.append(dict(text=c["text"], bbox=tobox(c["bb"]) if c["bb"] else [0, 0, 0, 0]))
                        elif (a, b_) not in occ:
                            html += ["<td>", "</td>"]; gcells.append(dict(text="", bbox=[0, 0, 0, 0]))
                    html.append("</tr>")
                words = []
                for w in pp.extract_words(keep_blank_chars=False, use_text_flow=False):
                    cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
                    if not (L <= cx <= Rr and T <= cy <= B): continue
                    wb = [(w["x0"] - L) * S, (w["top"] - T) * S, (w["x1"] - L) * S, (w["bottom"] - T) * S]
                    wcx, wcy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
                    gi = None
                    for i, gc in enumerate(gcells):
                        b = gc["bbox"]
                        if gc["text"] and b[0] - 1 <= wcx <= b[2] + 1 and b[1] - 1 <= wcy <= b[3] + 1: gi = i; break
                    words.append(dict(text=w["text"], bbox=wb, gi=gi))
                rec = dict(html=html, cells=[[dict(tokens=list(g["text"]), bbox=[int(v) for v in g["bbox"]] + [2 if g["text"] else 1]) for g in gcells]])
                T_ = tag(rec, im)
                T_["split_page"] = len(regions) > 1
                T_["rotated"] = False
                gt = dict(id=tid, filename=os.path.basename(pdf), page=page, region_pt=[x1, y1, x2, y2], crop_pt=[L, T, Rr, B], scale=S,
                          html=html, cells=gcells, words=words, tags=T_)
                im.save(os.path.join(OUT, tid + ".png"))
                json.dump(gt, open(os.path.join(OUT, tid + ".json"), "w", encoding="utf-8"))
                index.append(tid)
        plb.close()
    json.dump(sorted(index), open(os.path.join(OUT, "index.json"), "w"))
    print("regions", len(index), "pdfs", len(set(i.split("_t")[0] for i in index)), "skipped", skipped)


if __name__ == "__main__":
    main()
