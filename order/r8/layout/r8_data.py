"""Round 8, test B data: layout sets not covered in round 7. Writes OUT/pages.jsonl with
{id, img, W, H, gt: [[cls, x0, y0, x1, y1], ...] or None, src, eval_classes}.
  pod2017  : ICDAR-2017 POD, table subset re-released with full table boxes (github cndplab-founder/
             ICDAR2017_POD_dataset_supplement; the original Train/Test downloads are gone). Only Table boxes are
             annotated, so only the Table class is scored.
  docbank  : DocBank test pages (500K_test.json MS-COCO boxes, images read by HTTP range from the split
             image archive on the Hugging Face hub, liminghao1630/DocBank, Apache-2.0). Its text and equation
             boxes come from token labels and do not follow paragraph / display-formula granularity (inline
             math is labelled equation), so only Table, Picture and Caption are scored.
  real2    : real PDFs / scans without ground truth: World Bank documents (reports, papers), Zenodo slide decks
             (presentations with a CC-BY licence) and scanned forms (FUNSD, RVL-CDIP form class).
usage: r8_data.py SET OUT [MAX]"""
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pictures"))
UA = {"User-Agent": "layout-benchmark/1.0 (research test)"}
DOCBANK = {"table": "Table", "figure": "Picture", "caption": "Caption", "equation": "Formula", "footer": "Page-footer"}


def get(url, dest, tries=4, timeout=120):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            return os.path.getsize(dest) > 1000
        except Exception as e:  # noqa: BLE001
            print("get retry", url[:100], type(e).__name__, str(e)[:80], flush=True)
            time.sleep(4 + 4 * attempt)
    return False


def get_json(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("json retry", url[:100], type(e).__name__, flush=True)
            time.sleep(5 * (k + 1))
    return None


def pod2017(out, limit):
    repo = os.path.join(out, "_pod")
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/cndplab-founder/ICDAR2017_POD_dataset_supplement.git",
                        repo], check=True)
    res = []
    for split in ("TestSet", "TrainSet"):
        adir = os.path.join(repo, "Annotations", split)
        for f in sorted(os.listdir(adir)):
            if not f.endswith(".xml"):
                continue
            ip = os.path.join(repo, "Images", split, f[:-4] + ".jpg")
            if not os.path.exists(ip):
                continue
            root = ET.parse(os.path.join(adir, f)).getroot()
            gt = []
            for t in root.iter("table"):
                c = t.find("Coords")
                if c is None:
                    continue
                pts = [tuple(float(v) for v in p.split(",")) for p in c.get("points").split()]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                gt.append(["Table", min(xs), min(ys), max(xs), max(ys)])
            w, h = Image.open(ip).size
            res.append(dict(id=f"pod2017:{split}:{f[:-4]}", img=ip, W=w, H=h, gt=gt, src="pod2017",
                            eval_classes=["Table"]))
    random.Random(7).shuffle(res)
    return res[:limit]


def docbank(out, limit):
    import zipfile
    from httpzip import HttpFile
    from huggingface_hub import hf_hub_download
    root = os.path.join(out, "_hf")
    zp = hf_hub_download("liminghao1630/DocBank", "MSCOCO_Format_Annotation.zip", repo_type="dataset", local_dir=root)
    coco = json.loads(zipfile.ZipFile(zp).read("500K_test.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    print("docbank categories", cats, flush=True)
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    imgs = [im for im in coco["images"] if im["id"] in anns]
    random.Random(7).shuffle(imgs)
    imgs = imgs[:limit]
    del coco
    urls = ["https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.%03d" % i
            for i in range(1, 11)]
    z = zipfile.ZipFile(HttpFile(urls))
    names = {os.path.basename(n): n for n in z.namelist()}
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    res = []
    for k, im in enumerate(imgs):
        n = names.get(im["file_name"])
        if not n:
            continue
        ip = os.path.join(out, "img", f"docbank{k:05d}.jpg")
        open(ip, "wb").write(z.read(n))
        w, h = Image.open(ip).size
        sx, sy = w / im["width"], h / im["height"]
        gt = []
        for a in anns[im["id"]]:
            x, y, bw, bh = a["bbox"]
            name = cats.get(a["category_id"], "paragraph").lower()
            gt.append([DOCBANK.get(name, "Text"), x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])
        res.append(dict(id=f"docbank:{im['file_name']}", img=ip, W=w, H=h, gt=gt, src="docbank",
                        eval_classes=["Table", "Picture", "Caption"]))
        if k % 200 == 0:
            print("docbank pages", k, flush=True)
    return res


def render_pdf(p, src, out, res, per_pdf, limit):
    import pypdfium2 as pdfium
    try:
        doc = pdfium.PdfDocument(p)
    except Exception:  # noqa: BLE001
        return
    n = len(doc)
    idx = list(range(n)) if n <= per_pdf else sorted(random.Random(p).sample(range(n), per_pdf))
    for i in idx:
        if len(res) >= limit:
            return
        try:
            img = doc[i].render(scale=2).to_pil().convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        if max(img.size) > 4000:
            img.thumbnail((4000, 4000))
        ip = os.path.join(out, "img", f"{src}-{len(res):05d}.png")
        img.save(ip)
        res.append(dict(id=f"real:{src}:{os.path.basename(p)}:{i}", img=ip, W=img.size[0], H=img.size[1], gt=None,
                        src=src))


def real2(out, limit):
    pdf_dir = os.path.join(out, "_pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    per_pdf = int(os.environ.get("R8_PAGES_PER_PDF", "6"))
    quota = {"worldbank": limit * 35 // 100, "slides_zenodo": limit * 35 // 100, "scanned_forms": limit * 30 // 100}
    res = []
    # World Bank documents and reports (English PDFs, several document types)
    wb = []
    for docty in ("Annual Report", "Working Paper", "Country Economic Memorandum", "Project Appraisal Document",
                  "Policy Research Working Paper", "Brief", "Report", "Publication"):
        q = urllib.parse.urlencode({"format": "json", "rows": 20, "fl": "pdfurl,docty,display_title",
                                    "docty_exact": docty, "lang_exact": "English", "strdate": "2015-01-01"})
        d = get_json("https://search.worldbank.org/api/v3/wds?" + q)
        docs = list((d or {}).get("documents", {}).values())
        got = [x["pdfurl"] for x in docs if isinstance(x, dict) and str(x.get("pdfurl", "")).endswith(".pdf")]
        print("worldbank", docty, len(got), flush=True)
        wb += got[:12]
    part = []
    for k, u in enumerate(wb):
        dest = os.path.join(pdf_dir, f"wb-{k:03d}.pdf")
        if get(u, dest):
            render_pdf(dest, "worldbank", out, part, per_pdf, quota["worldbank"])
        time.sleep(0.5)
        if len(part) >= quota["worldbank"]:
            break
    res += part
    # Zenodo presentations (slide decks as PDF, CC-BY only)
    part = []
    for pg in range(1, 30):
        d = get_json("https://zenodo.org/api/records?" + urllib.parse.urlencode(
            {"q": "slides", "type": "presentation", "size": 25, "page": pg, "sort": "mostrecent"}))
        if not d:
            break
        for h in d.get("hits", {}).get("hits", []):
            lic = ((h.get("metadata") or {}).get("license") or {}).get("id", "")
            if not str(lic).startswith("cc-by") or any(t in lic for t in ("nc", "nd", "sa")):
                continue
            pdfs = [f for f in h.get("files", []) if f["key"].lower().endswith(".pdf") and f.get("size", 0) < 60e6]
            if not pdfs:
                continue
            dest = os.path.join(pdf_dir, f"zen-{h['id']}.pdf")
            if get(pdfs[0]["links"]["self"], dest):
                render_pdf(dest, "slides_zenodo", out, part, per_pdf, quota["slides_zenodo"])
            time.sleep(1.0)
            if len(part) >= quota["slides_zenodo"]:
                break
        if len(part) >= quota["slides_zenodo"]:
            break
    res += part
    # scanned forms: FUNSD (all 199) and RVL-CDIP "form" pages
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    part = []
    for f in ("data/test-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"):
        p = hf_hub_download("nielsr/funsd", f, repo_type="dataset", local_dir=os.path.join(out, "_hf"))
        for r in pq.read_table(p, columns=["id", "image"]).to_pylist():
            if len(part) >= quota["scanned_forms"] // 2:
                break
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            ip = os.path.join(out, "img", f"funsd-{len(part):05d}.png")
            img.save(ip)
            part.append(dict(id=f"real:funsd:{r['id']}", img=ip, W=img.size[0], H=img.size[1], gt=None, src="scanned_forms_funsd"))
    p = hf_hub_download("chainyo/rvl-cdip", "data/test-00000-of-00015.parquet", repo_type="dataset",
                        local_dir=os.path.join(out, "_hf"))
    pf = pq.ParquetFile(p)
    names = None
    for b in pf.iter_batches(batch_size=32, columns=["image", "label"]):
        for r in b.to_pylist():
            if len(part) >= quota["scanned_forms"]:
                break
            if r["label"] != 4:  # 'form' in the RVL-CDIP label order
                continue
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            ip = os.path.join(out, "img", f"rvlform-{len(part):05d}.png")
            img.save(ip)
            part.append(dict(id=f"real:rvlcdip_form:{len(part)}", img=ip, W=img.size[0], H=img.size[1], gt=None,
                             src="scanned_forms_rvlcdip"))
        if len(part) >= quota["scanned_forms"]:
            break
    res += part
    return res


def main():
    ds, out = sys.argv[1], sys.argv[2]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    os.makedirs(out, exist_ok=True)
    rows = {"pod2017": pod2017, "docbank": docbank, "real2": real2}[ds](out, lim)
    with open(os.path.join(out, "pages.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by[r["src"]] = by.get(r["src"], 0) + 1
    print(ds, "pages", len(rows), by, flush=True)


if __name__ == "__main__":
    main()
