"""Pick picture boxes from real document pages for hand labelling (round 8, test A):
DocLayNet v1.1 test pages (Picture boxes, spread over the six document categories) and OmniDocBench
(figure boxes, spread over its page sources, all languages). Neither dataset labels the picture type, so the crops
are labelled by eye from contact sheets (realdoc_sheets.py) and the labels are stored in
realdoc_labels.jsonl. Only box coordinates and labels are committed, never images.
usage: realdoc_select.py OUT.jsonl CACHE_DIR"""
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict

UA = {"User-Agent": "layout-benchmark/1.0 (research test)"}
DLN = "docling-project/DocLayNet-v1.1"
ODB = "opendatalab/OmniDocBench"
PER_DLN_CAT = 60
PER_ODB_SRC = 40
MIN_SIDE = 40
MIN_AREA = 6000


def get_json(url):
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def doclaynet(cache):
    path = os.path.join(cache, "dln_test_pictures.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    rows = []
    for off in range(0, 4999, 100):
        d = get_json(f"https://datasets-server.huggingface.co/rows?dataset={DLN}&config=default&split=test"
                     f"&offset={off}&length=100")
        for k, r in enumerate(d["rows"]):
            row = r["row"]
            md = row["metadata"]
            pics = []
            for b, c in zip(row["bboxes"] or [], row["category_id"] or []):
                x, y, w, h = b
                if int(c) == 7 and w >= MIN_SIDE and h >= MIN_SIDE and w * h >= MIN_AREA:
                    pics.append([round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)])
            if pics:
                rows.append({"row": off + k, "page_hash": md["page_hash"], "doc": md["original_filename"],
                             "cat": md["doc_category"], "W": row["image"]["width"], "H": row["image"]["height"],
                             "pics": pics})
        print("doclaynet rows", off + 100, "pages with pictures", len(rows), flush=True)
    json.dump(rows, open(path, "w", encoding="utf-8"))
    return rows


def omnidocbench(cache):
    path = os.path.join(cache, "OmniDocBench.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{ODB}/resolve/main/OmniDocBench.json", path)
    pages = json.load(open(path, encoding="utf-8"))
    rows = []
    for p in pages:
        pi = p["page_info"]
        att = pi.get("page_attribute", {})
        pics = []
        for d in p["layout_dets"]:
            if d.get("category_type") != "figure" or d.get("ignore"):
                continue
            xs, ys = d["poly"][0::2], d["poly"][1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if x1 - x0 >= MIN_SIDE and y1 - y0 >= MIN_SIDE and (x1 - x0) * (y1 - y0) >= MIN_AREA:
                pics.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
        if pics:
            rows.append({"image": pi["image_path"], "src_type": att.get("data_source"), "W": pi.get("width"),
                         "H": pi.get("height"), "pics": pics})
    return rows


def pick(rows, key, per, doc_key, rng):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    out = []
    for k in sorted(by, key=str):
        rs = by[k][:]
        rng.shuffle(rs)
        used_docs = defaultdict(int)
        used_pages = set()
        n = 0
        for max_per_doc in (2, 6):  # second pass fills categories with few documents
            for r in rs:
                if n >= per:
                    break
                if id(r) in used_pages or used_docs[r[doc_key]] >= max_per_doc:
                    continue
                used_pages.add(id(r))
                used_docs[r[doc_key]] += 1
                for b in rng.sample(r["pics"], min(2, len(r["pics"]))):
                    out.append((r, b))
                    n += 1
                    if n >= per:
                        break
    return out


def main():
    out, cache = sys.argv[1], sys.argv[2]
    os.makedirs(cache, exist_ok=True)
    rng = random.Random(8)
    items = []
    for r, b in pick(doclaynet(cache), "cat", PER_DLN_CAT, "doc", rng):
        items.append({"src": "doclaynet_test", "page_hash": r["page_hash"], "row": r["row"], "doc": r["doc"],
                      "type": r["cat"], "bbox": b})
    for r, b in pick(omnidocbench(cache), "src_type", PER_ODB_SRC, "image", rng):
        items.append({"src": "omnidocbench", "image": r["image"], "doc": r["image"].rsplit("_page", 1)[0],
                      "type": r["src_type"], "bbox": b})
    for i, it in enumerate(items):
        it["id"] = "rd" + hashlib.md5(json.dumps(it, sort_keys=True).encode()).hexdigest()[:10]
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    c = defaultdict(int)
    for it in items:
        c[(it["src"], it["type"])] += 1
    print(len(items), dict(c))


if __name__ == "__main__":
    main()
