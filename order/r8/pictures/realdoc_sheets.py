"""Contact sheets (4 x 4 numbered crops) of the boxes chosen by realdoc_select.py, for labelling by eye.
Images are fetched one by one (DocLayNet page images through the datasets-server rows API, OmniDocBench
page images from the dataset repo) into a local cache that is never committed.
usage: realdoc_sheets.py BOXES.jsonl CACHE_DIR SHEET_DIR"""
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw

UA = {"User-Agent": "layout-benchmark/1.0 (research test)"}
CELL = 300
GRID = 4


def fetch(url, tries=10):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            time.sleep(6 * (k + 1))
    raise RuntimeError(url)


def page_image(it, cache):
    key = it.get("page_hash") or os.path.basename(it["image"])
    p = os.path.join(cache, "pages", key + (".jpg" if it["src"] == "doclaynet_test" else ""))
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if it["src"] == "doclaynet_test":
            d = json.loads(fetch("https://datasets-server.huggingface.co/rows?dataset=docling-project/DocLayNet-v1.1"
                                 f"&config=default&split=test&offset={it['row']}&length=1"))
            row = d["rows"][0]["row"]
            assert row["metadata"]["page_hash"] == it["page_hash"]
            data = fetch(row["image"]["src"])
        else:
            data = fetch("https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/images/"
                         + urllib.request.quote(os.path.basename(it["image"])))
        open(p + ".part", "wb").write(data)
        os.replace(p + ".part", p)
    return p


def main():
    boxes, cache, sheets = sys.argv[1], sys.argv[2], sys.argv[3]
    items = [json.loads(x) for x in open(boxes, encoding="utf-8") if x.strip()]
    os.makedirs(sheets, exist_ok=True)

    first = {}
    for it in items:
        first.setdefault(it.get("page_hash") or it["image"], it)
    with ThreadPoolExecutor(3) as ex:
        paths = dict(zip(first, ex.map(lambda it: page_image(it, cache), first.values())))
    crops = []
    for it in items:
        im = Image.open(paths[it.get("page_hash") or it["image"]]).convert("RGB")
        x0, y0, x1, y1 = it["bbox"]
        crops.append(im.crop((int(max(0, x0)), int(max(0, y0)), int(min(im.width, x1)), int(min(im.height, y1)))))
    index = []
    per = GRID * GRID
    for s in range(0, len(items), per):
        sheet = Image.new("RGB", (GRID * CELL, GRID * (CELL + 24)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (it, c) in enumerate(zip(items[s:s + per], crops[s:s + per])):
            n = s + k
            c = c.copy()
            c.thumbnail((CELL - 8, CELL - 8))
            gx, gy = (k % GRID) * CELL, (k // GRID) * (CELL + 24)
            sheet.paste(c, (gx + 4 + (CELL - 8 - c.width) // 2, gy + 24 + (CELL - 8 - c.height) // 2))
            d.rectangle([gx, gy, gx + CELL - 1, gy + CELL + 23], outline=(160, 160, 160))
            d.text((gx + 6, gy + 5), f"#{n}  {it['src'][:3]}/{it['type'][:10]}", fill=(200, 0, 0))
            index.append({"n": n, "id": it["id"]})
        sheet.save(os.path.join(sheets, f"sheet_{s // per:03d}.png"))
    json.dump(index, open(os.path.join(sheets, "index.json"), "w"))
    print("sheets", (len(items) + per - 1) // per)


if __name__ == "__main__":
    main()
