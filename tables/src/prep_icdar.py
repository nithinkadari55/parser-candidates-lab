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


def mappers(Wu, Hu):
    return {
        0: lambda x, y: (x, Hu - y),
        90: lambda x, y: (y, x),
        180: lambda x, y: (Wu - x, y),
        270: lambda x, y: (Hu - y, Wu - x),
    }


def map_box(f, bb):
    pts = [f(bb[0], bb[1]), f(bb[2], bb[3]), f(bb[0], bb[3]), f(bb[2], bb[1])]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def main():
    import shutil
    pdfdir = os.path.join(SETDIR, "icdar_pdf"); os.makedirs(pdfdir, exist_ok=True)
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
        shutil.copy(pdf, os.path.join(pdfdir, os.path.basename(pdf)))
        for t in st.iter("table"):
            regions = list(t.iter("region"))
            for r in regions:
                key = (t.get("id"), r.get("id"))
                tid = f"icdar_{os.path.basename(base)}_t{key[0]}r{key[1]}"
                cells = []
                for c in r.iter("cell"):
                    sr = int(c.get("start-row")); sc = int(c.get("start-col"))
                    er = int(c.get("end-row", sr)); ec = int(c.get("end-col", sc))
                    b = c.find("bounding-box"); cont = c.find("content")
                    bb = [num(b.get(k)) for k in ("x1", "y1", "x2", "y2")] if b is not None else None
                    cells.append(dict(sr=sr, er=er, sc=sc, ec=ec, bb=bb, text=(cont.text or "").replace("\n", " ").strip() if cont is not None else ""))
                if not cells: skipped.append((tid, "no cells")); continue
                if key in regbox:
                    page, (x1, y1, x2, y2) = regbox[key]
                else:
                    bbs = [c["bb"] for c in cells if c["bb"]]
                    if not bbs or not r.get("page"): skipped.append((tid, "no region box")); continue
                    page = int(r.get("page")); x1, y1 = min(b[0] for b in bbs), min(b[1] for b in bbs); x2, y2 = max(b[2] for b in bbs), max(b[3] for b in bbs)
                pp = plb.pages[page - 1]
                ml, mb, mr, mt = doc[page - 1].get_mediabox()
                Wu, Hu = mr - ml, mt - mb
                pw = [((w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2) for w in pp.extract_words(keep_blank_chars=False, use_text_flow=False)]
                best = None
                for rot, f in mappers(Wu, Hu).items():
                    gb = [map_box(f, c["bb"]) for c in cells if c["bb"]]
                    hit = sum(1 for (px, py) in pw if any(b[0] - 1 <= px <= b[2] + 1 and b[1] - 1 <= py <= b[3] + 1 for b in gb))
                    if best is None or hit > best[0]: best = (hit, rot, f)
                rot, f = best[1], best[2]
                if pp.rotation == 0: f = mappers(Wu, Hu)[0]; rot = 0
                rx0, ry0, rx1, ry1 = map_box(f, [x1, y1, x2, y2])
                L, T, Rr, B = rx0 - M, ry0 - M, rx1 + M, ry1 + M
                r0 = min(c["sr"] for c in cells); c0 = min(c["sc"] for c in cells)
                R = max(c["er"] for c in cells) - r0 + 1; C = max(c["ec"] for c in cells) - c0 + 1
                img = doc[page - 1].render(scale=S).to_pil().convert("RGB")
                im = img.crop((int(L * S), int(T * S), int(Rr * S), int(B * S)))

                def tobox(bb):
                    b = map_box(f, bb)
                    return [(b[0] - L) * S, (b[1] - T) * S, (b[2] - L) * S, (b[3] - T) * S]
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
                T_ = tag(rec, im, dict(split_page=len(regions) > 1, rotated=False, rotated_page=bool(pp.rotation)))
                if pp.rotation: print("rotated page", tid, "rotation", pp.rotation, "gt mapping", rot, "word hits", best[0], "of", len(pw), flush=True)
                gt = dict(id=tid, filename=os.path.basename(pdf), page=page, region_pt=[x1, y1, x2, y2], crop_pt=[L, T, Rr, B], scale=S, page_rotation=pp.rotation, gt_rot=rot,
                          html=html, cells=gcells, words=words, tags=T_)
                im.save(os.path.join(OUT, tid + ".png"))
                json.dump(gt, open(os.path.join(OUT, tid + ".json"), "w", encoding="utf-8"))
                index.append(tid)
        plb.close()
    json.dump(sorted(index), open(os.path.join(OUT, "index.json"), "w"))
    print("regions", len(index), "pdfs", len(set(i.split("_t")[0] for i in index)), "skipped", skipped)


if __name__ == "__main__":
    main()
