"""Round 7 data: layout benchmarks with ground truth (D4LA test, M6Doc test, PubLayNet validation) and
real PDF pages without ground truth (docling.rs test corpus, recent arXiv papers, Federal Register
documents). Writes OUT/<set>/pages.jsonl with {id, img, W, H, gt: [[cls, x0, y0, x1, y1], ...], src}.
usage: r7_data.py SET OUT [MAX]"""
import io
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

D4LA = ["DocTitle", "ParaTitle", "ParaText", "ListText", "RegionTitle", "Date", "LetterHead", "LetterDear",
        "LetterSign", "Question", "OtherText", "RegionKV", "RegionList", "Abstract", "Author", "TableName", "Table",
        "Figure", "FigureName", "Equation", "Reference", "Footer", "PageHeader", "PageFooter", "Number", "Catalog",
        "PageNumber"]
D4LA_COARSE = {"Table": "Table", "Catalog": "Table", "Figure": "Picture", "TableName": "Caption",
               "FigureName": "Caption", "Equation": "Formula", "PageHeader": "Page-header", "Footer": "Page-footer",
               "PageFooter": "Page-footer", "PageNumber": "Page-footer"}
PUBLAYNET = {1: "Text", 2: "Text", 3: "Text", 4: "Table", 5: "Picture"}
UA = {"User-Agent": "layout-benchmark/1.0 (research test)"}


def save_img(img, path):
    img.save(path)
    return img.size


def yolo(ds, out, limit):
    from huggingface_hub import hf_hub_download
    repo = {"d4la": "juliozhao/D4LA", "m6doc": "juliozhao/M6Doc"}[ds]
    root = os.path.join(out, "_hf")
    lst = open(hf_hub_download(repo, "test.txt", repo_type="dataset", local_dir=root), encoding="utf-8").read().split()
    lst = [p.lstrip("./") for p in lst if p.strip()]
    random.Random(7).shuffle(lst)
    lst = lst[:limit]

    def one(rel):
        for attempt in range(4):
            try:
                ip = hf_hub_download(repo, rel, repo_type="dataset", local_dir=root)
                lab = rel.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
                lp = hf_hub_download(repo, lab, repo_type="dataset", local_dir=root)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return None
                time.sleep(2 + attempt * 3)
        w, h = Image.open(ip).size
        gt = []
        for line in open(lp, encoding="utf-8"):
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            if ds == "d4la":
                name = D4LA[c] if c < len(D4LA) else "Other"
                cls = D4LA_COARSE.get(name, "Text")
            else:
                cls = "Any"
            gt.append([cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        return dict(id=f"{ds}:{os.path.basename(rel)}", img=ip, W=w, H=h, gt=gt, src=ds)

    with ThreadPoolExecutor(8) as ex:
        return [r for r in ex.map(one, lst) if r]


def publaynet(out, limit):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    root = os.path.join(out, "_hf")
    path = hf_hub_download("creative-graphic-design/PubLayNet", "data/validation-00000-of-00007.parquet",
                           repo_type="dataset", local_dir=root)
    t = pq.read_table(path)
    print("publaynet schema:", t.schema, flush=True)
    rows = t.to_pylist()
    random.Random(7).shuffle(rows)
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    res = []
    for i, r in enumerate(rows[:limit]):
        im = r.get("image")
        data = im.get("bytes") if isinstance(im, dict) else im
        img = Image.open(io.BytesIO(data)).convert("RGB")
        ip = os.path.join(out, "img", f"plv{i:05d}.png")
        w, h = save_img(img, ip)
        anns = r.get("annotations") or r.get("objects") or []
        if isinstance(anns, dict):
            anns = [dict(zip(anns, v)) for v in zip(*anns.values())]
        gt = []
        for a in anns:
            cid = a.get("category_id")
            bb = a.get("bbox")
            if cid is None or not bb:
                continue
            x, y, bw, bh = bb
            gt.append([PUBLAYNET.get(int(cid), "Text"), x, y, x + bw, y + bh])
        res.append(dict(id=f"publaynet:{r.get('image_id', i)}", img=ip, W=w, H=h, gt=gt, src="publaynet"))
    os.remove(path)
    return res


def get(url, dest, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return os.path.getsize(dest) > 1000
        except Exception:  # noqa: BLE001
            time.sleep(3 + 3 * attempt)
    return False


def real_pdfs(out, limit):
    """Real PDFs without ground truth, rendered at 2x: docling.rs's own test corpus, recent arXiv
    papers across subject areas, and Federal Register documents."""
    import pypdfium2 as pdfium
    pdf_dir = os.path.join(out, "_pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    pdfs = []
    corpus = os.path.join(out, "_corpus")
    if not os.path.exists(corpus):
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2.84.0", "--filter=blob:none", "--sparse",
                        "https://github.com/docling-project/docling.rs.git", corpus], check=True)
        subprocess.run(["git", "-C", corpus, "sparse-checkout", "set", "tests/data/pdf/sources",
                        "tests/data/scanned/sources"], check=True)
    for d, _, fs in os.walk(os.path.join(corpus, "tests", "data")):
        for f in sorted(fs):
            if f.lower().endswith(".pdf"):
                pdfs.append(("docling_corpus", os.path.join(d, f)))
    cats = ["cs.CL", "cs.CV", "math.PR", "physics.optics", "q-fin.ST", "econ.EM", "stat.ME", "q-bio.NC",
            "astro-ph.GA", "cond-mat.mtrl-sci", "eess.SP", "cs.CR"]
    for cat in cats:
        url = ("http://export.arxiv.org/api/query?search_query=cat:%s&sortBy=submittedDate&max_results=12" % cat)
        feed = os.path.join(pdf_dir, f"arxiv-{cat}.xml")
        if not get(url, feed):
            continue
        import re
        for aid in re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", open(feed, encoding="utf-8").read())[:12]:
            dest = os.path.join(pdf_dir, "arxiv-" + aid.replace("/", "_") + ".pdf")
            if get("https://arxiv.org/pdf/" + aid, dest):
                pdfs.append(("arxiv", dest))
            time.sleep(1.0)
    for kind in ("RULE", "PRORULE", "NOTICE", "PRESDOCU"):
        url = ("https://www.federalregister.gov/api/v1/documents.json?per_page=30&order=newest"
               "&conditions[type][]=%s&fields[]=pdf_url&fields[]=document_number" % kind)
        js = os.path.join(pdf_dir, f"fr-{kind}.json")
        if not get(url, js):
            continue
        for doc in json.load(open(js, encoding="utf-8")).get("results", [])[:30]:
            if doc.get("pdf_url"):
                dest = os.path.join(pdf_dir, f"fr-{doc['document_number']}.pdf")
                if get(doc["pdf_url"], dest):
                    pdfs.append(("federal_register", dest))
                time.sleep(0.5)
    res = []
    per_pdf = int(os.environ.get("R7_PAGES_PER_PDF", "6"))
    for src, p in pdfs:
        try:
            doc = pdfium.PdfDocument(p)
        except Exception:  # noqa: BLE001
            continue
        n = len(doc)
        idx = list(range(n)) if n <= per_pdf else sorted(random.Random(p).sample(range(n), per_pdf))
        for i in idx:
            try:
                img = doc[i].render(scale=2).to_pil().convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            name = f"{src}-{len(res):05d}.png"
            ip = os.path.join(out, "img", name)
            w, h = save_img(img, ip)
            res.append(dict(id=f"real:{src}:{os.path.basename(p)}:{i}", img=ip, W=w, H=h, gt=None, src=src))
            if len(res) >= limit:
                return res
    return res


def main():
    ds, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    os.makedirs(out, exist_ok=True)
    if ds in ("d4la", "m6doc"):
        rows = yolo(ds, out, limit)
    elif ds == "publaynet":
        rows = publaynet(out, limit)
    else:
        rows = real_pdfs(out, limit)
    with open(os.path.join(out, "pages.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by[r["src"]] = by.get(r["src"], 0) + 1
    print(ds, "pages", len(rows), by, flush=True)


if __name__ == "__main__":
    main()
