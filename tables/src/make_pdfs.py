import os, json, re
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from paths import SETS

PAD = 20
TAG = re.compile(r"<[^>]+>")


def clean(s):
    s = TAG.sub("", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def make_multi(setname, ids, out, page="tight"):
    d = os.path.join(SETS, setname)
    c = canvas.Canvas(out)
    offsets = []
    for tid in ids:
        gt = json.load(open(os.path.join(d, tid + ".json"), encoding="utf-8"))
        im = Image.open(os.path.join(d, tid + ".png")); W, H = im.size
        if page == "letter":
            PW, PH = max(W + 2 * PAD, 612), max(H + 2 * 40, 792); ox, oy = (PW - W) / 2.0, 40.0
        else:
            PW, PH = W + 2 * PAD, H + 2 * PAD; ox, oy = float(PAD), float(PAD)
        offsets.append((ox, oy))
        c.setPageSize((PW, PH))
        rot = gt.get("rot") == 90
        src = gt.get("words") or [dict(text=cl["text"], bbox=cl["bbox"]) for cl in gt["cells"]]
        src = [x for x in src if clean(x["text"]).strip() and x.get("bbox") and x["bbox"][2] > x["bbox"][0] and x["bbox"][3] > x["bbox"][1]]
        hs = sorted(((x["bbox"][2] - x["bbox"][0]) if rot else (x["bbox"][3] - x["bbox"][1])) for x in src)
        lh = max(4.0, hs[len(hs) // 2] if hs else 10.0)
        for cl in src:
            t = clean(cl["text"]).strip()
            x0, y0, x1, y1 = cl["bbox"]
            x0 = max(x0, 0); w = max(1.0, x1 - x0); h = max(1.0, y1 - y0)
            if rot: w, h = h, w
            n = max(1, int(round(h / (lh * 1.15))))
            words = t.split()
            if n == 1 or len(words) < 2: lines = [t]
            else:
                per = max(1, -(-len(words) // n)); lines = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
            lhh = h / len(lines); fs = max(2.0, min(lhh * 0.85, lh * 0.85))
            for k, ln in enumerate(lines):
                sw = stringWidth(ln, "Helvetica", fs) or 1.0
                if rot:
                    c.saveState(); c.translate(ox + x0 + (k + 1) * lhh - 0.2 * lhh, PH - (oy + y1)); c.rotate(90)
                    to = c.beginText(); to.setFont("Helvetica", fs); to.setHorizScale(100.0 * w / sw); to.setTextOrigin(0, 0); to.textLine(ln); c.drawText(to)
                    c.restoreState()
                else:
                    to = c.beginText(); to.setFont("Helvetica", fs); to.setHorizScale(100.0 * w / sw)
                    to.setTextOrigin(ox + x0, PH - (oy + y0 + (k + 1) * lhh) + 0.2 * lhh); to.textLine(ln); c.drawText(to)
        c.drawImage(os.path.join(d, tid + ".png"), ox, PH - oy - H, width=W, height=H)
        c.showPage()
    c.save()
    return offsets
