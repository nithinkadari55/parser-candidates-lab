"""Round 8, test A: picture crops from many sources, each class from several sources.
Crops are written to OUT/<source>/img/*.jpg with OUT/<source>/items.jsonl and OUT/<source>/status.json.
Everything is downloaded inside the job (actions/cache, never committed or uploaded).
usage: p8_data.py GROUP OUTDIR [--probe]   (GROUP as in p8_common.SOURCES; --probe = a few items per source)"""
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p8_common import CAP, CLASSES, SOURCES, fingerprint  # noqa: E402

TMP = os.path.join(os.environ.get("RUNNER_TEMP", os.path.expanduser("~")), "p8-tmp")
UA = {"User-Agent": "layout-benchmark/1.0 (research test)"}
PROBE = False


class Sink:
    def __init__(self, out, src):
        self.src = src
        self.dir = os.path.join(out, src)
        os.makedirs(os.path.join(self.dir, "img"), exist_ok=True)
        self.items = []
        self.count = Counter()
        self.cap = 6 if PROBE else CAP

    def full(self, label=None):
        if label is None:
            return False
        return self.count[label] >= self.cap

    def save(self, pil, label, group, sub=None, extra=None):
        from PIL import Image
        if label not in CLASSES or self.full(label):
            return False
        pil = pil.convert("RGB")
        w, h = pil.size
        if w < 16 or h < 16:
            return False
        s = 512 / max(w, h)
        if s < 1:
            pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        i = len(self.items)
        name = f"{self.src}_{i:05d}.jpg"
        pil.save(os.path.join(self.dir, "img", name), quality=92)
        m = {"id": f"{self.src}:{i:05d}", "path": "img/" + name, "label": label, "src": self.src,
             "group": f"{self.src}:{group}", "sub": sub, "w": w, "h": h}
        m.update(fingerprint(pil))
        if extra:
            m.update(extra)
        self.items.append(m)
        self.count[label] += 1
        return True

    def close(self, status):
        with open(os.path.join(self.dir, "items.jsonl"), "w", encoding="utf-8") as f:
            for m in self.items:
                f.write(json.dumps(m) + "\n")
        json.dump({"src": self.src, "status": status, "counts": dict(self.count)},
                  open(os.path.join(self.dir, "status.json"), "w"), indent=1)


# ---------------------------------------------------------------- download helpers

def get(url, dst, tries=8, budget_s=None, headers=None):
    """download with retries and HTTP Range resume"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return dst
    part = dst + ".part"
    t0 = time.time()
    for k in range(tries):
        if budget_s and time.time() - t0 > budget_s:
            break
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            h = {**UA, **(headers or {}), **({"Range": f"bytes={have}-"} if have else {})}
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                total = r.headers.get("Content-Length")
                with open(part, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        if budget_s and time.time() - t0 > budget_s:
                            raise TimeoutError(f"time budget {budget_s} s used up")
            if total is None or os.path.getsize(part) >= int(total) + (have if mode == "ab" else 0):
                os.replace(part, dst)
                return dst
        except Exception as e:  # noqa: BLE001
            print("download retry", k, type(e).__name__, str(e)[:160], "bytes", have, flush=True)
            if isinstance(e, TimeoutError):
                break
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed: {url}")


def fetch_bytes(url, tries=5, headers=None, timeout=60):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code in (403, 404, 410):
                raise
            time.sleep(3 * (k + 1) * (4 if code == 429 else 1))
    raise RuntimeError(f"fetch failed: {url}")


def hf(repo, path):
    from huggingface_hub import hf_hub_download
    for k in range(5):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", local_dir=os.path.join(TMP, repo.replace("/", "__")))
        except Exception as e:  # noqa: BLE001
            if k == 4:
                raise
            print("hf retry", repo, path, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(10 * (k + 1))


def hf_files(repo, pattern=None):
    from huggingface_hub import HfApi
    fs = sorted(HfApi().list_repo_files(repo, repo_type="dataset"))
    return [f for f in fs if pattern is None or re.search(pattern, f)]


def parquet_rows(path, cols=None, limit=None, batch=16):
    import pyarrow.parquet as pq
    n = 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=cols):
        for r in b.to_pylist():
            yield r
            n += 1
            if limit and n >= limit:
                return


def img_of(v):
    from PIL import Image
    if isinstance(v, dict):
        v = v.get("bytes")
    im = Image.open(io.BytesIO(v))
    im.load()
    return im


def crop(im, x0, y0, x1, y1):
    W, H = im.size
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


def limit(n):
    return min(n, 8) if PROBE else n


# ---------------------------------------------------------------- charts

CHARTX_SUB = {"bar_chart": "bar", "bar_chart_num": "bar", "3D-Bar": "bar", "histogram": "bar", "line_chart": "line",
              "line_chart_num": "line", "pie_chart": "pie", "rings": "pie", "box": "box"}


def chartx(s):
    from PIL import Image
    z = zipfile.ZipFile(hf("InternScience/ChartX", "ChartX_png.zip"))
    by = defaultdict(list)
    for n in z.namelist():
        p = n.split("/")
        if len(p) == 4 and p[2] == "png" and n.lower().endswith(".png"):
            by[p[1]].append(n)
    print("chartx types", {k: len(v) for k, v in by.items()})
    rng = random.Random(1)
    per = max(1, limit(CAP) // len(by) + 1)
    for t in sorted(by):
        xs = by[t][:]
        rng.shuffle(xs)
        for n in xs[:per]:
            s.save(Image.open(io.BytesIO(z.read(n))), "chart", n, CHARTX_SUB.get(t), {"orig_type": t})


def plotqa(s):
    p = hf("achang/plot_qa", hf_files("achang/plot_qa", r"data/test-00000-")[0])
    for k, r in enumerate(parquet_rows(p, ["image"], limit=limit(CAP) * 2)):
        if s.full("chart"):
            break
        s.save(img_of(r["image"]), "chart", str(k))


def chartqa(s):
    p = hf("HuggingFaceM4/ChartQA", hf_files("HuggingFaceM4/ChartQA", r"data/test-")[0])
    seen = set()
    for k, r in enumerate(parquet_rows(p, ["image"])):
        b = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        h = hashlib.md5(b).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if s.full("chart") or (PROBE and len(seen) > 10):
            break
        s.save(img_of(b), "chart", h[:12])


UBPMC_URL = "https://www.dropbox.com/s/w0j0rxund06y04f/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1.zip?dl=1"
UBPMC_SUB = {"line": "line", "vertical bar": "bar", "horizontal bar": "bar", "scatter": "scatter",
             "vertical box": "box", "pie": "pie"}


def ubpmc(s):
    from PIL import Image
    z = zipfile.ZipFile(get(UBPMC_URL, os.path.join(TMP, "ubpmc_test.zip")))
    names = z.namelist()
    imgs = {os.path.splitext(os.path.basename(n))[0]: n for n in names if "/chart_images/" in n
            and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    by = defaultdict(list)
    for n in names:
        if "/final_full_GT/" in n and "/annotations_JSON/" in n and n.endswith(".json"):
            stem = os.path.splitext(os.path.basename(n))[0]
            try:
                t = json.loads(z.read(n))["task1"]["output"]["chart_type"]
            except Exception:  # noqa: BLE001
                continue
            if stem in imgs:
                by[t.strip().lower()].append(stem)
    print("ubpmc types", {k: len(v) for k, v in by.items()})
    rng = random.Random(4)
    chart_types = [t for t in by if t not in ("map", "venn")]
    per = limit(CAP) // max(1, len(chart_types)) + 1
    for t in sorted(by):
        xs = sorted(by[t])
        rng.shuffle(xs)
        label = "map" if t == "map" else None if t == "venn" else "chart"
        if label is None:
            continue
        for stem in xs[:(limit(CAP) if label == "map" else per)]:
            s.save(Image.open(io.BytesIO(z.read(imgs[stem]))), label, stem.split("___")[0],
                   UBPMC_SUB.get(t), {"orig_type": t})


SCICAP = {"graph plot": ("chart", None), "bar chart": ("chart", "bar"), "scatterplot": ("chart", "scatter"),
          "node diagram": ("diagram", None), "table": ("table_image", None), "photo": ("photo", None)}


def scicap(s):
    from PIL import Image
    meta = json.load(open(hf("CrowdAILab/scicap", "hidden-test.json"), encoding="utf-8"))
    types = {m["file_name"]: m.get("figure_type", "") for m in meta["images"]}
    print("scicap figure types", Counter(types.values()))
    if PROBE:
        from httpzip import HttpFile
        z = zipfile.ZipFile(HttpFile("https://huggingface.co/datasets/CrowdAILab/scicap/resolve/main/img-hide_test.zip"))
    else:
        z = zipfile.ZipFile(hf("CrowdAILab/scicap", "img-hide_test.zip"))
    by = defaultdict(list)
    for n in z.namelist():
        t = types.get(os.path.basename(n), "").lower()
        if t in SCICAP:
            by[t].append(n)
    rng = random.Random(5)
    for t in sorted(by):
        xs = sorted(by[t])
        rng.shuffle(xs)
        label, sub = SCICAP[t]
        cap = limit(CAP) // 3 if label == "chart" else limit(CAP)
        for n in xs[:cap]:
            s.save(Image.open(io.BytesIO(z.read(n))), label, os.path.basename(n), sub, {"orig_type": t})


# ---------------------------------------------------------------- diagrams

def ai2d(s):
    seen = set()
    for f in hf_files("lmms-lab/ai2d", r"\.parquet$"):
        for r in parquet_rows(hf("lmms-lab/ai2d", f), ["image"]):
            b = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
            h = hashlib.md5(b).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            s.save(img_of(b), "diagram", h[:12])
            if s.full("diagram") or (PROBE and len(seen) > 8):
                return


def qzhou_flowchart(s):
    files = hf_files("Kingsoft-LLM/QZhou-Flowchart-QA", r"\.parquet$")
    small = [f for f in files if "Y_swimlane_timeline_flowchart/" in f or "Colorful_diverse" in f
             or "X_and_Y_swimlane" in f or "Black_white" in f]
    per = limit(CAP) // max(1, len(small)) + 1
    for f in small:
        style = f.split("/")[0]
        n = 0
        for r in parquet_rows(hf("Kingsoft-LLM/QZhou-Flowchart-QA", f), ["image", "image_id"], limit=per):
            s.save(img_of(r["image"]), "diagram", f"{style}:{r['image_id']}", extra={"orig_type": style})
            n += 1


# ---------------------------------------------------------------- photos

def openimages(s):
    p = hf("finedet/openimages", "data/test-00000-of-00045.parquet")
    for k, r in enumerate(parquet_rows(p, ["image"], limit=limit(CAP) + 50)):
        s.save(img_of(r["image"]), "photo", str(k))
        if s.full("photo"):
            break


def coco_ccby(s):
    from PIL import Image
    p = hf("phiyodr/coco2017", hf_files("phiyodr/coco2017", r"validation-.*\.parquet$")[0])
    rows = [r for r in parquet_rows(p, ["license", "coco_url", "image_id"]) if int(r["license"]) in (4, 7, 8)]
    print("coco val2017 images with licence 4/7/8:", len(rows), Counter(int(r["license"]) for r in rows))
    random.Random(6).shuffle(rows)
    rows = rows[:limit(CAP) + 40]

    def one(r):
        try:
            return r, fetch_bytes(r["coco_url"])
        except Exception:  # noqa: BLE001
            return r, None
    with ThreadPoolExecutor(8) as ex:
        for r, b in ex.map(one, rows):
            if b:
                s.save(Image.open(io.BytesIO(b)), "photo", str(r["image_id"]), extra={"licence_id": int(r["license"])})


def imagenette(s):
    from PIL import Image
    t = tarfile.open(get("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", os.path.join(TMP, "inette.tgz")))
    ms = [m for m in t.getmembers() if m.name.endswith(".JPEG")]
    random.Random(2).shuffle(ms)
    for m in ms[:limit(CAP)]:
        s.save(Image.open(t.extractfile(m)), "photo", m.name)


# ---------------------------------------------------------------- screenshots

def rico(s):
    p = hf("bevaya/RICO-Screen2Words", "data/test-00000-of-00002.parquet")
    per = Counter()
    for r in parquet_rows(p, ["image", "app_package_name", "screenId"]):
        app = r.get("app_package_name") or str(r.get("screenId"))
        if per[app] >= 3:
            continue
        per[app] += 1
        s.save(img_of(r["image"]), "screenshot", app)
        if s.full("screenshot"):
            break


def screenspot(s):
    from PIL import Image
    if PROBE:
        from httpzip import HttpFile
        z = zipfile.ZipFile(HttpFile("https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2/resolve/main/screenspotv2_image.zip"))
    else:
        z = zipfile.ZipFile(hf("OS-Copilot/ScreenSpot-v2", "screenspotv2_image.zip"))
    names = sorted(n for n in z.namelist() if n.lower().endswith((".png", ".jpg")))
    random.Random(7).shuffle(names)
    for n in names[:limit(CAP)]:
        base = os.path.basename(n)
        s.save(Image.open(io.BytesIO(z.read(n))), "screenshot", base, extra={"orig_type": base.split("_")[0]})


def websight(s):
    f = [x for x in hf_files("HuggingFaceM4/WebSight", r"\.parquet$") if "v0.1" in x] or hf_files("HuggingFaceM4/WebSight", r"\.parquet$")
    p = hf("HuggingFaceM4/WebSight", f[0])
    for k, r in enumerate(parquet_rows(p, ["image"], limit=limit(CAP))):
        im = img_of(r["image"])
        w, h = im.size
        top = random.Random(k).randint(0, max(0, h - int(0.7 * w)))
        s.save(im.crop((0, top, w, min(h, top + int(0.7 * w)))), "screenshot", str(k))


# ---------------------------------------------------------------- logos and signatures

def logodet(s):
    p = hf("axonstan/LogoDet-3K", hf_files("axonstan/LogoDet-3K", r"test.*\.parquet$")[0])
    per = Counter()
    for r in parquet_rows(p, limit=8000):
        comp = str(r.get("company_name"))
        if per[comp] >= 2:
            continue
        b = r.get("bbox")
        im = img_of(r.get("image_path") or r.get("image"))
        if b and len(b) == 4:
            c = crop(im, *b)
            if c is None:
                continue
            im = c
        if s.save(im, "logo", comp, extra={"orig_type": str(r.get("industry_name"))}):
            per[comp] += 1
        if s.full("logo"):
            break


def signverod(s):
    for f in hf_files("ondrs/signverod", r"\.parquet$"):
        for k, r in enumerate(parquet_rows(hf("ondrs/signverod", f))):
            o = r["objects"]
            im = None
            for b, c in zip(o["bbox"], o["category"]):
                if c != 1:
                    continue
                im = im or img_of(r["image"])
                x, y, w, h = b
                cc = crop(im, x, y, x + w, y + h)
                if cc is not None:
                    s.save(cc, "signature", str(r.get("image_id", k)))
            if s.full("signature"):
                return


def rf100_signatures(s):
    from PIL import Image
    from huggingface_hub import snapshot_download
    root = snapshot_download("LibreYOLO/signatures-xc8up", repo_type="dataset", local_dir=os.path.join(TMP, "rf100sig"))
    imgs = sorted(p for p in __import__("glob").glob(os.path.join(root, "*", "images", "*")) if p.lower().endswith((".jpg", ".png")))
    for ip in imgs:
        lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        stem = os.path.basename(ip).split("_jpg.rf.")[0].split("_png.rf.")[0]
        for line in open(lp):
            v = line.split()
            if len(v) < 5:
                continue
            cx, cy, w, h = (float(x) for x in v[1:5])
            c = crop(im, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            if c is not None:
                s.save(c, "signature", stem)
        if s.full("signature"):
            return


def cedar_signatures(s):
    from PIL import Image
    files = [f for f in hf_files("T0KII/signatures-dataset", r"\.png$")]
    random.Random(9).shuffle(files)
    for f in files[:limit(CAP)]:
        m = re.match(r"(original|forgeries)_(\d+)_", os.path.basename(f))
        s.save(Image.open(hf("T0KII/signatures-dataset", f)), "signature", m.group(2) if m else f)


# ---------------------------------------------------------------- stamps

STAMP_SUBS = {"stamps_jsam": "stamp_detection_jsam", "stamps_yolo": "yolo_stamp_classify",
              "stamps_marcos": "stamp_detectation_marcos", "stamps_stampa": "stamp_detection_stampa"}


def mapo80(s, prefix):
    from PIL import Image
    for sp in ("test", "val", "train"):
        z = zipfile.ZipFile(hf("mapo80/stamps", f"{sp}.zip"))
        names = set(z.namelist())
        for lab in sorted(x for x in names if x.endswith(".txt") and "labels/" in x and os.path.basename(x).startswith(prefix)):
            stem = os.path.splitext(os.path.basename(lab))[0]
            img = next((x for x in (lab.replace("labels/", "images/", 1)[:-4] + e for e in (".jpg", ".png", ".jpeg"))
                        if x in names), None)
            rows = [l.split() for l in z.read(lab).decode().splitlines() if l.strip()]
            if not img or not rows:
                continue
            im = Image.open(io.BytesIO(z.read(img))).convert("RGB")
            W, H = im.size
            for r in rows:
                if len(r) < 5:
                    continue
                cx, cy, w, h = (float(v) for v in r[1:5])
                c = crop(im, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
                if c is not None:
                    s.save(c, "stamp", stem)
            if s.full("stamp"):
                return


def hyperbots_stamps(s):
    from PIL import Image
    meta = [json.loads(l) for l in open(hf("hyperbots-inc/synthetic-stamped-documents", "data/test/metadata.jsonl"),
                                        encoding="utf-8") if l.strip()]
    for m in meta[:limit(len(meta))]:
        im = Image.open(hf("hyperbots-inc/synthetic-stamped-documents", "data/test/" + m["file_name"])).convert("RGB")
        for st in m.get("stamps", []):
            b = st.get("bbox")
            if b and len(b) == 4:
                c = crop(im, *b)
                if c is not None:
                    s.save(c, "stamp", m["id"], extra={"orig_type": st.get("type")})


# ---------------------------------------------------------------- tables

def doclaynet_tables(s):
    p = hf("docling-project/DocLayNet-v1.1", hf_files("docling-project/DocLayNet-v1.1", r"data/val-00000-")[0])
    per = Counter()
    for r in parquet_rows(p, ["image", "bboxes", "category_id", "metadata"], batch=8):
        doc = (r["metadata"] or {}).get("original_filename", "")
        im = None
        for b, c in zip(r["bboxes"] or [], r["category_id"] or []):
            if int(c) != 9 or per[doc] >= 3:
                continue
            im = im or img_of(r["image"])
            x, y, w, h = b
            cc = crop(im, x, y, x + w, y + h)
            if cc is not None and s.save(cc, "table_image", doc, extra={"orig_type": r["metadata"].get("doc_category")}):
                per[doc] += 1
        if s.full("table_image"):
            return


def pubtabnet(s):
    p = hf("apoidea/pubtabnet-html", "data/validation-00000-of-00001.parquet")
    for r in parquet_rows(p, ["image", "imgid"], limit=limit(CAP)):
        s.save(img_of(r["image"]), "table_image", str(r["imgid"]))


def publaynet_tables(s):
    p = hf("creative-graphic-design/PubLayNet", hf_files("creative-graphic-design/PubLayNet", r"data/validation-00000-")[0])
    for k, r in enumerate(parquet_rows(p, batch=8)):
        anns = r.get("annotations") or r.get("objects") or []
        if isinstance(anns, dict):
            anns = [dict(zip(anns, v)) for v in zip(*anns.values())]
        im = None
        n = 0
        for a in anns:
            if int(a.get("category_id", -1)) != 4 or n >= 2:
                continue
            im = im or img_of(r["image"])
            x, y, w, h = a["bbox"]
            c = crop(im, x, y, x + w, y + h)
            if c is not None and s.save(c, "table_image", str(r.get("image_id", k))):
                n += 1
        if s.full("table_image"):
            return


# ---------------------------------------------------------------- Wikimedia Commons (permissive files only)

WM_CATS = {
    "chart": [("Bar_charts", "bar"), ("Line_charts", "line"), ("Pie_charts", "pie"), ("Scatterplots", "scatter"),
              ("Box_plots", "box"), ("Histograms", "bar"), ("Area_charts", None)],
    "diagram": [("Flowcharts", None), ("Flow_charts", None), ("Block_diagrams", None), ("Circuit_diagrams", None),
                ("UML_diagrams", None), ("Organizational_charts", None), ("Network_diagrams", None),
                ("Technical_drawings", None)],
    "map": [("Topographic_maps", None), ("Choropleth_maps", None), ("Locator_maps", None), ("Road_maps", None),
            ("Political_maps", None), ("Thematic_maps", None)],
    "logo": [("SVG_logos", None), ("Logos", None), ("Wordmarks", None), ("Text_logos", None)],
    "signature": [("SVG_signatures", None), ("Signatures", None), ("Autographs", None)],
    "screenshot": [("Screenshots", None), ("Screenshots_of_web_browsers", None),
                   ("Screenshots_of_free_software", None), ("Screenshots_of_GNU/Linux", None)],
    "stamp": [("Rubber_stamp_impressions", None), ("Stamp_impressions", None), ("Ink_stamps", None)],
}
WM_PER_CLASS = 150


def wm_ok(lic):
    l = (lic or "").strip().lower()
    if any(t in l for t in ("sa", "nc", "nd", "gfdl", "gpl", "fair use", "copyright")):
        return l.startswith(("public domain", "pd")) and "sa" not in l.split()
    return l.startswith(("public domain", "pd", "cc0", "cc by", "cc-by"))


def wm_api(params, state):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"format": "json", **params})
    while True:
        wait = 1.0 - (time.time() - state["last"])
        if wait > 0:
            time.sleep(wait)
        state["last"] = time.time()
        try:
            return json.loads(fetch_bytes(url, tries=6))
        except Exception as e:  # noqa: BLE001
            print("wikimedia api", type(e).__name__, str(e)[:100], flush=True)
            state["errors"] += 1
            if state["errors"] > 40:
                raise
            time.sleep(20)


def wm_files(cat, state, want):
    out, cont = [], {}
    while len(out) < want * 4:
        d = wm_api({"action": "query", "generator": "categorymembers", "gcmtitle": "Category:" + cat,
                    "gcmtype": "file", "gcmlimit": "50", "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
                    "iiurlwidth": "512", "iiextmetadatafilter": "LicenseShortName", **cont}, state)
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value")
            if ii.get("mime") in ("image/png", "image/jpeg", "image/svg+xml", "image/gif") and wm_ok(lic) and ii.get("thumburl"):
                out.append((p["title"], ii["thumburl"], lic))
        if "continue" not in d:
            break
        cont = {k: v for k, v in d["continue"].items() if k != "continue"}
    return out


def wikimedia(s):
    from PIL import Image
    state = {"last": 0.0, "errors": 0}
    log = {}
    t0 = time.time()
    budget = 60 * 70
    for label, cats in WM_CATS.items():
        want = limit(WM_PER_CLASS)
        files = []
        for cat, sub in cats:
            if time.time() - t0 > budget:
                break
            try:
                fs = wm_files(cat, state, want)
            except Exception as e:  # noqa: BLE001
                fs = []
                print("wikimedia category failed", cat, e, flush=True)
            log[cat] = len(fs)
            files += [(f, sub) for f in fs]
        random.Random(10).shuffle(files)
        seen = set()
        got = 0
        for (title, url, lic), sub in files:
            if got >= want or time.time() - t0 > budget:
                break
            if title in seen:
                continue
            seen.add(title)
            wait = 0.5 - (time.time() - state["last"])
            if wait > 0:
                time.sleep(wait)
            state["last"] = time.time()
            try:
                im = Image.open(io.BytesIO(fetch_bytes(url, tries=4)))
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    bg.alpha_composite(im)
                    im = bg
                if s.save(im, label, title, sub if label == "chart" else None, {"licence": lic, "orig_type": title[:80]}):
                    got += 1
            except Exception as e:  # noqa: BLE001
                print("wikimedia file failed", title[:60], type(e).__name__, flush=True)
        print("wikimedia", label, "saved", got, "of", len(files), "candidates", flush=True)
    print("wikimedia permissive files per category:", log, flush=True)


# ---------------------------------------------------------------- CVIT (slow server)

DOCFIG = {"line graph": ("chart", "line"), "graph plots": ("chart", None), "bar plot": ("chart", "bar"),
          "scatter plot": ("chart", "scatter"), "histogram": ("chart", "bar"), "box plot": ("chart", "box"),
          "pie chart": ("chart", "pie"), "area chart": ("chart", None), "bubble chart": ("chart", None),
          "polar plot": ("chart", None), "radar chart": ("chart", None), "pareto chart": ("chart", None),
          "heat map": ("chart", None), "contour plot": ("chart", None), "surface plot": ("chart", None),
          "vector plot": ("chart", None), "confusion matrix": ("chart", None), "flow chart": ("diagram", None),
          "block diagram": ("diagram", None), "tree diagram": ("diagram", None), "natural image": ("photo", None),
          "table": ("table_image", None), "geographic map": ("map", None)}


def docfigure(s):
    from PIL import Image
    z = zipfile.ZipFile(get("https://cvit.iiit.ac.in/usodi/img/projects/classification/Docfig/DocFigure_dataset.zip",
                            os.path.join(TMP, "docfig.zip"), tries=30, budget_s=60 * (5 if PROBE else 150)))
    names = z.namelist()
    imgs = {os.path.basename(n): n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))}
    labels = {}
    for n in names:
        if not n.lower().endswith((".txt", ".csv")):
            continue
        for line in z.read(n).decode("utf-8", "replace").splitlines():
            parts = [x.strip() for x in re.split(r"[,\t]", line) if x.strip()]
            fn = next((x for x in parts if os.path.basename(x) in imgs), None)
            lab = next((x.lower() for x in parts if x.lower() in DOCFIG), None)
            if fn and lab:
                labels[os.path.basename(fn)] = lab
    print("docfigure labelled", len(labels), Counter(labels.values()), flush=True)
    by = defaultdict(list)
    for b, t in sorted(labels.items()):
        by[t].append(b)
    rng = random.Random(11)
    for t in sorted(by):
        label, sub = DOCFIG[t]
        xs = by[t][:]
        rng.shuffle(xs)
        cap = limit(CAP) // 6 if label == "chart" else limit(CAP) // 2 if label == "diagram" else limit(CAP)
        for b in xs[:cap]:
            try:
                s.save(Image.open(io.BytesIO(z.read(imgs[b]))), label, b, sub, {"orig_type": t})
            except Exception:  # noqa: BLE001
                pass


IIIT = {"natural_image": "photo", "logo": "logo", "signature": "signature", "table": "table_image"}


def iiitar13k(s):
    from PIL import Image
    page = fetch_bytes("https://cvit.iiit.ac.in/usodi/iiitar13k.php", tries=6, timeout=120).decode("utf-8", "replace")
    links = sorted(set(re.findall(r'href="([^"]+)"', page)))
    print("iiit-ar-13k page links:", [l for l in links if any(k in l.lower() for k in ("zip", "tar", "drive", "download", "data"))])
    cand = [urllib.parse.urljoin("https://cvit.iiit.ac.in/usodi/", l) for l in links
            if l.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".rar"))]
    if not cand:
        raise RuntimeError("no archive link on the IIIT-AR-13K page (links: %s)" % links[:40])
    print("iiit-ar-13k archives:", cand, flush=True)
    import xml.etree.ElementTree as ET
    for url in cand:
        if not url.lower().endswith(".zip"):
            continue
        zp = get(url, os.path.join(TMP, "iiit", os.path.basename(url)), tries=30, budget_s=60 * (5 if PROBE else 60))
        z = zipfile.ZipFile(zp)
        names = z.namelist()
        imgs = {os.path.splitext(os.path.basename(n))[0]: n for n in names if n.lower().endswith((".jpg", ".png", ".jpeg"))}
        xmls = [n for n in names if n.lower().endswith(".xml")]
        print(os.path.basename(url), len(names), "images", len(imgs), "xml", len(xmls), flush=True)
        random.Random(12).shuffle(xmls)
        for xn in xmls:
            stem = os.path.splitext(os.path.basename(xn))[0]
            if stem not in imgs:
                continue
            root = ET.fromstring(z.read(xn))
            objs = []
            for o in root.iter("object"):
                nm = (o.findtext("name") or "").strip().lower()
                bb = o.find("bndbox")
                if nm in IIIT and bb is not None:
                    objs.append((IIIT[nm], *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
            if not objs:
                continue
            im = Image.open(io.BytesIO(z.read(imgs[stem]))).convert("RGB")
            for lab, x0, y0, x1, y1 in objs:
                if not s.full(lab):
                    c = crop(im, x0, y0, x1, y1)
                    if c is not None:
                        s.save(c, lab, stem)
            if all(s.full(l) for l in IIIT.values()):
                return


# ---------------------------------------------------------------- hand-labelled real-document crops

def realdoc(s_dln, s_odb, out):
    labels = [json.loads(l) for l in open(os.path.join(HERE, "realdoc_labels.jsonl"), encoding="utf-8") if l.strip()]
    labels = [m for m in labels if m.get("label") in CLASSES]
    if PROBE:
        labels = labels[:6] + [m for m in labels if m["src"] == "omnidocbench"][:6]
    dln = defaultdict(list)
    for m in labels:
        if m["src"] == "doclaynet_test":
            dln[m["page_hash"]].append(m)
    if dln:
        for f in hf_files("docling-project/DocLayNet-v1.1", r"data/test-"):
            for r in parquet_rows(hf("docling-project/DocLayNet-v1.1", f), ["image", "metadata"], batch=8):
                ph = (r["metadata"] or {}).get("page_hash")
                if ph not in dln:
                    continue
                im = img_of(r["image"])
                for m in dln.pop(ph):
                    c = crop(im, *m["bbox"])
                    if c is not None:
                        s_dln.save(c, m["label"], m["doc"], extra={"rid": m["id"], "orig_type": m["type"]})
                if not dln:
                    break
            if not dln:
                break
    print("doclaynet labelled crops not found:", sum(len(v) for v in dln.values()), flush=True)
    for m in labels:
        if m["src"] != "omnidocbench":
            continue
        from PIL import Image
        im = Image.open(hf("opendatalab/OmniDocBench", "images/" + os.path.basename(m["image"]))).convert("RGB")
        c = crop(im, *m["bbox"])
        if c is not None:
            s_odb.save(c, m["label"], m["doc"], extra={"rid": m["id"], "orig_type": m["type"]})


# ---------------------------------------------------------------- main

FUNCS = {"chartx": chartx, "plotqa": plotqa, "chartqa": chartqa, "ubpmc": ubpmc, "scicap": scicap, "ai2d": ai2d,
         "qzhou_flowchart": qzhou_flowchart, "openimages": openimages, "coco_ccby": coco_ccby,
         "imagenette": imagenette, "rico": rico, "screenspot": screenspot, "websight": websight, "logodet": logodet,
         "signverod": signverod, "rf100_signatures": rf100_signatures, "cedar_signatures": cedar_signatures,
         "hyperbots_stamps": hyperbots_stamps, "doclaynet_tables": doclaynet_tables, "pubtabnet": pubtabnet,
         "publaynet_tables": publaynet_tables, "wikimedia": wikimedia, "docfigure": docfigure, "iiitar13k": iiitar13k}
for _k, _p in STAMP_SUBS.items():
    FUNCS[_k] = (lambda p: (lambda s: mapo80(s, p)))(_p)


def main():
    global PROBE
    group, out = sys.argv[1], sys.argv[2]
    PROBE = "--probe" in sys.argv
    os.makedirs(out, exist_ok=True)
    srcs = [k for k, v in SOURCES.items() if v[0] == group]
    if group == "realdoc":
        a, b = Sink(out, "realdoc_doclaynet"), Sink(out, "realdoc_omnidocbench")
        a.cap = b.cap = 100000
        try:
            realdoc(a, b, out)
            st = "ok"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            st = f"failed: {type(e).__name__}: {str(e)[:300]}"
        a.close(st)
        b.close(st)
        print("realdoc", st, a.count, b.count)
        return
    for src in srcs:
        s = Sink(out, src)
        t0 = time.time()
        try:
            FUNCS[src](s)
            st = "ok"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            st = f"failed: {type(e).__name__}: {str(e)[:300]}"
        s.close(st)
        print(f"{src}: {st}, {dict(s.count)}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
