"""Picture crops for gap 10 (chart, diagram, photo, screenshot, logo, signature, stamp, table_image, other).
Sources are downloaded inside the job, cropped to JPEG under ~/pics (actions/cache), never committed:
  DocFigure (CVIT, non-commercial research terms; 28 classes mapped below), ChartX PNGs (InternScience/ChartX),
  AI2D diagrams (lmms-lab/ai2d), Imagenette-160 photos, WebSight v0.1 screenshots (HuggingFaceM4/WebSight,
  CC-BY-4.0), LogoDet-3K logo boxes (axonstan/LogoDet-3K), SignverOD signature boxes (ondrs/signverod,
  Apache-2.0 card; its redaction boxes, category 3, also go to "other"), document stamps (mapo80/stamps, CC-BY-4.0), DocLayNet test Table boxes (CDLA-Permissive-1.0)
  and DocLayNet Picture boxes (unlabelled, only for a domain check).
Splits are by group (paper / chart / company / document / page) so no group is in both train and test.
usage: pics_data.py OUTDIR"""
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "layout"))

CLASSES = ["chart", "diagram", "photo", "screenshot", "logo", "signature", "stamp", "table_image", "other"]
DOCFIG = {"line graph": "chart", "graph plots": "chart", "bar plot": "chart", "scatter plot": "chart",
          "histogram": "chart", "box plot": "chart", "pie chart": "chart", "area chart": "chart",
          "bubble chart": "chart", "polar plot": "chart", "radar chart": "chart", "pareto chart": "chart",
          "heat map": "chart", "contour plot": "chart", "surface plot": "chart", "vector plot": "chart",
          "confusion matrix": "chart", "flow chart": "diagram", "block diagram": "diagram",
          "venn diagram": "diagram", "tree diagram": "diagram", "sketch": "diagram", "natural image": "photo",
          "medical image": "photo", "table": "table_image", "algorithm": "other", "mask": "other",
          "3d object": "other", "geographic map": "other"}
CAP = {"train": 1200, "test": 500}
OUT = None
MAN = []
TMP = os.path.join(os.path.expanduser("~"), "pics-tmp")


def split_of(group):
    h = int(hashlib.md5(("pics:" + group).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < 0.3 else "train"


def save(pil, label, src, group, split=None):
    from PIL import Image
    pil = pil.convert("RGB")
    w, h = pil.size
    if w < 12 or h < 12:
        return
    s = 512 / max(w, h)
    if s < 1:
        pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    i = len(MAN)
    p = os.path.join(OUT, "img", f"{i:06d}.jpg")
    pil.save(p, quality=92)
    MAN.append({"id": i, "path": p, "label": label, "src": src, "group": f"{src}:{group}",
                "split": split or split_of(f"{src}:{group}")})


def get(url, dst, tries=8):
    """download with retries and HTTP Range resume (the CVIT server drops long transfers)"""
    import time
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return dst
    part = dst + ".part"
    for k in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ci-eval", **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=300) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                total = r.headers.get("Content-Length")
                with open(part, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
            if total is None or os.path.getsize(part) >= int(total) + (have if mode == "ab" else 0):
                os.replace(part, dst)
                return dst
        except Exception as e:  # noqa: BLE001
            print("download retry", k, type(e).__name__, str(e)[:120], "bytes", have, flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed after {tries} tries: {url}")


def docfigure():
    from PIL import Image
    z = zipfile.ZipFile(get("https://cvit.iiit.ac.in/usodi/img/projects/classification/Docfig/DocFigure_dataset.zip",
                            os.path.join(TMP, "docfig.zip")))
    names = z.namelist()
    print("docfigure entries", len(names), names[:12])
    imgs = {os.path.basename(n): n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))}
    labels = {}
    split_hint = {}
    for n in names:
        if not n.lower().endswith((".txt", ".csv")):
            continue
        txt = z.read(n).decode("utf-8", "replace")
        sp = "test" if "test" in n.lower() else "train" if "train" in n.lower() else None
        for line in txt.splitlines():
            parts = [x.strip() for x in re.split(r"[,\t]", line) if x.strip()]
            fn = next((x for x in parts if os.path.basename(x) in imgs), None)
            lab = next((DOCFIG[x.lower()] for x in parts if x.lower() in DOCFIG), None)
            if fn and lab:
                labels[os.path.basename(fn)] = lab
                if sp:
                    split_hint[os.path.basename(fn)] = sp
    if not labels:
        for b, n in imgs.items():
            parent = os.path.basename(os.path.dirname(n)).lower().replace("_", " ")
            if parent in DOCFIG:
                labels[b] = DOCFIG[parent]
    print("docfigure labelled", len(labels), Counter(labels.values()))
    cnt = Counter()
    for b, lab in sorted(labels.items()):
        sp = split_hint.get(b) or split_of("docfig:" + b)
        if cnt[(lab, sp)] >= CAP[sp]:
            continue
        try:
            save(Image.open(io.BytesIO(z.read(imgs[b]))), lab, "docfig", b, sp)
            cnt[(lab, sp)] += 1
        except Exception:  # noqa: BLE001
            pass


def hf(repo, pattern=None, first=True):
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if pattern is None or re.search(pattern, f))
    print(repo, len(files), files[:6])
    files = files[:1] if first else files
    return [hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(TMP, repo.replace("/", "__"))) for f in files]


def parquet_rows(path, cols=None, limit=None):
    import pyarrow.parquet as pq
    n = 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=16, columns=cols):
        for r in b.to_pylist():
            yield r
            n += 1
            if limit and n >= limit:
                return


def img_of(v):
    from PIL import Image
    if isinstance(v, dict):
        v = v.get("bytes")
    return Image.open(io.BytesIO(v))


def chartx():
    from PIL import Image
    p = hf("InternScience/ChartX", r"ChartX_png\.zip$")[0]
    z = zipfile.ZipFile(p)
    pngs = [n for n in z.namelist() if n.lower().endswith(".png")]
    random.Random(1).shuffle(pngs)
    for n in pngs[:1700]:
        save(Image.open(io.BytesIO(z.read(n))), "chart", "chartx", os.path.basename(n))


def ai2d():
    for r in parquet_rows(hf("lmms-lab/ai2d", r"\.parquet$")[0], limit=1700):
        im = r.get("image")
        if im:
            save(img_of(im), "diagram", "ai2d", hashlib.md5(str(r.get("question", "")).encode()).hexdigest()[:8]
                 + str(len(MAN)))


def imagenette():
    from PIL import Image
    t = tarfile.open(get("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", os.path.join(TMP, "inette.tgz")))
    ms = [m for m in t.getmembers() if m.name.endswith(".JPEG")]
    random.Random(2).shuffle(ms)
    for m in ms[:1700]:
        save(Image.open(t.extractfile(m)), "photo", "imagenette", m.name)


def websight():
    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files("HuggingFaceM4/WebSight", repo_type="dataset") if f.endswith(".parquet"))
    v01 = [f for f in files if "v0.1" in f] or files
    p = hf("HuggingFaceM4/WebSight", re.escape(v01[0]) + "$")[0]
    for k, r in enumerate(parquet_rows(p, ["image"], limit=1700)):
        im = img_of(r["image"])
        w, h = im.size
        top = random.Random(k).randint(0, max(0, h - int(0.7 * w)))
        save(im.crop((0, top, w, min(h, top + int(0.7 * w)))), "screenshot", "websight", str(k))


def logos():
    p = hf("axonstan/LogoDet-3K", r"test.*\.parquet$")[0]
    per = Counter()
    for r in parquet_rows(p, limit=6000):
        comp = str(r.get("company_name"))
        if per[comp] >= 3:
            continue
        b = r.get("bbox")
        im = img_of(r.get("image_path") or r.get("image"))
        if b and len(b) == 4:
            x0, y0, x1, y1 = b
            if x1 > x0 and y1 > y0:
                im = im.crop((x0, y0, x1, y1))
        save(im, "logo", "logodet", comp)
        per[comp] += 1
        if sum(per.values()) >= 1700:
            break


def signatures():
    p = hf("ondrs/signverod", r"\.parquet$")[0]
    cats = Counter()
    n = red = 0
    for k, r in enumerate(parquet_rows(p, limit=3000)):
        o = r["objects"]
        im = None
        for b, c in zip(o["bbox"], o["category"]):
            cats[c] += 1
            if c == 3 and red < 700:
                im = im or img_of(r["image"])
                x, y, w, h = b
                save(im.crop((x, y, x + w, y + h)), "other", "signverod_redaction", str(r.get("image_id", k)))
                red += 1
            if c != 1 or n >= 1700:
                continue
            im = im or img_of(r["image"])
            x, y, w, h = b
            save(im.crop((x, y, x + w, y + h)), "signature", "signverod", str(r.get("image_id", k)))
            n += 1
    print("signverod categories", dict(cats))


def stamps():
    from PIL import Image
    from huggingface_hub import hf_hub_download
    n = 0
    for sp in ("test", "val"):
        z = zipfile.ZipFile(hf_hub_download("mapo80/stamps", f"{sp}.zip", repo_type="dataset",
                                            local_dir=os.path.join(TMP, "stamps")))
        names = set(z.namelist())
        for lab in sorted(x for x in names if x.endswith(".txt") and "/labels/" in "/" + x):
            stem = os.path.splitext(os.path.basename(lab))[0]
            img = next((x for x in names if "/images/" in "/" + x and os.path.splitext(os.path.basename(x))[0] == stem), None)
            rows = [l.split() for l in z.read(lab).decode().splitlines() if l.strip()]
            if not img or not rows:
                continue
            im = Image.open(io.BytesIO(z.read(img))).convert("RGB")
            W, H = im.size
            for c, cx, cy, w, h in [r[:5] for r in rows if len(r) >= 5]:
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                save(im.crop((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)), "stamp", "stamps", stem)
                n += 1
            if n >= 1700:
                return


def doclaynet():
    import dln
    tables = pics = 0
    for ri, pid, img, gt, cat, w, h in dln.rows(0, lambda i: i % 2 == 0):
        for c, x0, y0, x1, y1 in gt:
            if c == "Table" and tables < 1700:
                save(img.crop((x0, y0, x1, y1)), "table_image", "doclaynet", pid)
                tables += 1
            elif c == "Picture" and pics < 1500 and (x1 - x0) * (y1 - y0) > 2500:
                save(img.crop((x0, y0, x1, y1)), "unlabeled", "doclaynet_pic", pid, "domain")
                pics += 1
        if tables >= 1700 and pics >= 1500:
            break


def main():
    global OUT
    OUT = sys.argv[1]
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    status = {}
    for fn in (docfigure, chartx, ai2d, imagenette, websight, logos, signatures, stamps, doclaynet):
        n0 = len(MAN)
        try:
            fn()
            status[fn.__name__] = len(MAN) - n0
        except Exception as e:  # noqa: BLE001
            status[fn.__name__] = f"failed: {type(e).__name__}: {str(e)[:200]}"
        print(fn.__name__, status[fn.__name__], flush=True)
    rng = random.Random(3)
    keep = []
    for sp in ("train", "test"):
        for c in CLASSES:
            xs = [m for m in MAN if m["split"] == sp and m["label"] == c]
            rng.shuffle(xs)
            keep += xs[:CAP[sp]]
    keep += [m for m in MAN if m["split"] == "domain"]
    json.dump({"items": keep, "status": status}, open(os.path.join(OUT, "manifest.json"), "w"))
    print(status)
    print(Counter((m["split"], m["label"]) for m in keep))


if __name__ == "__main__":
    main()
