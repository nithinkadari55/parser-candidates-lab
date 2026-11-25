"""Milestone 14 fixtures for the harness's scanned-table path, from synthetic pages only (no dataset content):

  scan_table.pdf          a letter page that is one image (a ruled 7 x 4 table drawn with DejaVu Sans, light
                          noise and blur, JPEG): no text layer
  scan_table_rot.pdf      the same page with the table turned 90 degrees counter-clockwise
  scan_fixtures.json      per page: the table region (points), the crop docling.rs renders (r5words: the harness's
                          own render and crop rule), PP-LCNet's label and confidence on the crop and, after the
                          gated turn, the round-8 scan arm on the upright crop: PP-OCRv6 small words (ppocr.py,
                          docling.rs's detector on the crop + recognizer) and TATR's cells from them
  lcnet_fixtures.json     PP-LCNet probabilities on the crop turned 0/90/180/270 degrees (and the png files)

usage: gen14.py OUT_DIR R5WORDS_BIN"""
import os, sys, io, json, glob, random, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tatr8 as T8
import bench_models as BM

GATE = 0.8
HEADER = ["Region", "Quarter", "Revenue", "Margin"]
BODY = [["North", "Q1", "1200", "12%"], ["South", "Q1", "980", "9%"], ["East", "Q2", "1430", "15%"],
        ["West", "Q2", "1105", "17%"], ["Central", "Q3", "760", "8%"], ["Total", "", "5475", "12%"]]


def font(size):
    for p in glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True) + glob.glob("/usr/share/fonts/**/*.ttf", recursive=True):
        return ImageFont.truetype(p, size)
    raise RuntimeError("no TrueType font on this runner")


def table_image():
    """the table at 2 px/pt (the render scale docling.rs uses): 4 columns of 110 pt, 7 rows of 22 pt, 10 pt text"""
    cw, rh, s = 110, 22, 2
    W, H = 4 * cw * s + 2, 7 * rh * s + 2
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); f = font(10 * s)
    for r in range(8): d.line([(0, r * rh * s), (W - 1, r * rh * s)], fill="black", width=2)
    for c in range(5): d.line([(c * cw * s, 0), (c * cw * s, H - 1)], fill="black", width=2)
    for r, row in enumerate([HEADER] + BODY):
        for c, t in enumerate(row):
            d.text((c * cw * s + 8 * s, r * rh * s + 5 * s), t, fill="black", font=f)
    return im


def scan(page, seed):
    rng = np.random.default_rng(seed)
    a = np.asarray(page.filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32)
    a = np.clip(a * 0.93 + 8 + rng.normal(0, 6, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(a).save(buf, format="JPEG", quality=85); buf.seek(0)
    return Image.open(buf).convert("RGB")


def page_pdf(table, turn, out_pdf, seed):
    """letter page at 2 px/pt with the table at (100 pt, 150 pt), turned `turn` degrees counter-clockwise"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    t = table.rotate(turn, expand=True) if turn else table
    page = Image.new("RGB", (1224, 1584), "white"); x, y = 200, 300
    page.paste(t, (x, y)); page = scan(page, seed)
    c = canvas.Canvas(out_pdf, pagesize=(612, 792))
    buf = io.BytesIO(); page.save(buf, format="JPEG", quality=90); buf.seek(0)
    c.drawImage(ImageReader(buf), 0, 0, width=612, height=792); c.showPage(); c.save()
    return [x / 2, y / 2, (x + t.size[0]) / 2, (y + t.size[1]) / 2]


def main():
    out, words_bin = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    table = table_image()
    M = BM.TATR16(0); M.keep_text = True
    lc = T8.Lcnet(); ocr = T8.Ocr()
    cases = []; jobs = []
    for name, turn, seed in (("scan_table", 0, 1), ("scan_table_rot", 90, 2)):
        region = page_pdf(table, turn, os.path.join(out, name + ".pdf"), seed)
        crop = [region[0] - 4, region[1] - 4, region[2] + 4, region[3] + 4]
        jobs.append(dict(set="fx", id=name, pdf=os.path.join(out, name + ".pdf"), page=0, crop_pt=crop))
        cases.append(dict(name=name, pdf=name + ".pdf", region=region, crop_pt=crop, drawn_turn=turn))
    jl_path = os.path.join(out, "jobs.jsonl")
    with open(jl_path, "w") as f:
        for j in jobs: f.write(json.dumps(j) + "\n")
    subprocess.run([words_bin, jl_path, os.path.join(out, "w")], check=True)
    for c in cases:
        wj = json.load(open(os.path.join(out, "w", "fx", c["name"] + ".json")))
        assert not wj["words"], "the scanned page must have no text layer"
        im = Image.open(os.path.join(out, "w", "fx", c["name"] + ".png")).convert("RGB")
        png = c["name"] + "_crop.png"; im.save(os.path.join(out, png))
        lab, conf = lc(im)
        turn = lab if (lab and conf >= GATE) else 0
        up = T8.turn_image(im, turn)
        words = ocr(up)
        cells, _, _ = T8.tatr(M, up, words, 0)
        R = max(x["r1"] for x in cells) if cells else 0; C = max(x["c1"] for x in cells) if cells else 0
        c.update(crop_png=png, crop_size=list(im.size), scale=wj["scale"], lcnet=dict(label=lab, confidence=conf), turn=turn,
                 upright_words=words, cells=[dict(r0=x["r0"], r1=x["r1"], c0=x["c0"], c1=x["c1"], bbox=x["bbox"], text=x.get("text", "")) for x in cells],
                 shape=[R, C])
        print(c["name"], "lcnet", lab, round(conf, 4), "turn", turn, "words", len(words), "shape", (R, C), [w["text"] for w in words][:12], flush=True)
    json.dump(dict(source="parser-candidates-lab tables/r8/gen14.py (tatr8.py's Ocr, Lcnet and tatr: the round-8 scan arm)", gate=GATE,
                   truth=dict(rows=7, cols=4, texts=HEADER + [t for row in BODY for t in row if t]), cases=cases),
              open(os.path.join(out, "scan_fixtures.json"), "w"), indent=1)
    crop = Image.open(os.path.join(out, cases[0]["crop_png"])).convert("RGB")
    probs = []
    for turn in (0, 90, 180, 270):
        im = crop.rotate(turn, expand=True); png = f"lcnet_{turn}.png"; im.save(os.path.join(out, png))
        w, h = im.size; s = 256 / min(w, h)
        r = im.resize((max(224, int(round(w * s))), max(224, int(round(h * s)))), Image.BILINEAR)
        W, H = r.size; l, t = (W - 224) // 2, (H - 224) // 2
        a = np.asarray(r.crop((l, t, l + 224, t + 224))).astype(np.float32)[:, :, ::-1] / 255.0
        a = ((a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)[None].astype(np.float32)
        p = lc.s.run(None, {lc.i: np.ascontiguousarray(a)})[0][0]
        probs.append(dict(png=png, turned=turn, resized=[W, H], probs=[float(v) for v in p], label=[0, 90, 180, 270][int(np.argmax(p))],
                          input_mean=[float(a[0, k].mean()) for k in range(3)]))
        print("lcnet", turn, [round(float(v), 4) for v in p], flush=True)
    json.dump(dict(model="PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx inference.onnx @7330ab7", cases=probs), open(os.path.join(out, "lcnet_fixtures.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
