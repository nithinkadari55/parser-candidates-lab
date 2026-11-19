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


def make_multi(setname, ids, out):
    d = os.path.join(SETS, setname)
    c = canvas.Canvas(out)
    for tid in ids:
        gt = json.load(open(os.path.join(d, tid + ".json"), encoding="utf-8"))
        im = Image.open(os.path.join(d, tid + ".png")); W, H = im.size
        PW, PH = W + 2 * PAD, H + 2 * PAD
        c.setPageSize((PW, PH))
        hs = sorted(cl["bbox"][3] - cl["bbox"][1] for cl in gt["cells"] if cl["text"].strip())
        lh = max(4.0, hs[len(hs) // 2] if hs else 10.0)
        for cl in gt["cells"]:
            t = clean(cl["text"]).strip()
            if not t: continue
            x0, y0, x1, y1 = cl["bbox"]
            x0 = max(x0, 0); w = max(1.0, x1 - x0); h = max(1.0, y1 - y0)
            n = max(1, int(round(h / (lh * 1.15))))
            words = t.split()
            if n == 1 or len(words) < 2: lines = [t]
            else:
                per = max(1, -(-len(words) // n)); lines = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
            lhh = h / len(lines); fs = max(2.0, min(lhh * 0.85, lh * 0.85))
            for k, ln in enumerate(lines):
                sw = stringWidth(ln, "Helvetica", fs) or 1.0
                to = c.beginText(); to.setFont("Helvetica", fs); to.setHorizScale(100.0 * w / sw)
                to.setTextOrigin(PAD + x0, PH - (PAD + y0 + (k + 1) * lhh) + 0.2 * lhh); to.textLine(ln); c.drawText(to)
        c.drawImage(os.path.join(d, tid + ".png"), PAD, PAD, width=W, height=H)
        c.showPage()
    c.save()
