"""DocLayNet test split (docling-project/DocLayNet-v1.1 parquet, CDLA-Permissive-1.0)."""
import io, os

DLN_REPO = "docling-project/DocLayNet-v1.1"
DLN_FILES = ["data/test-00000-of-00002-635b47e9044a436c.parquet", "data/test-00001-of-00002-b2396409212624ed.parquet"]
DLN_CATS = {1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item", 5: "Page-footer", 6: "Page-header", 7: "Picture",
            8: "Section-header", 9: "Table", 10: "Text", 11: "Title"}
DLN_DIR = os.environ.get("DLN_DIR", os.path.join(os.path.expanduser("~"), "dln"))


def fetch(fi):
    from huggingface_hub import hf_hub_download
    for _ in range(4):
        try:
            return hf_hub_download(DLN_REPO, DLN_FILES[fi], repo_type="dataset", local_dir=DLN_DIR)
        except Exception as e:
            err = e
    raise err


def rows(fi, keep):
    """Yield (row_index, page_id, PIL image, gt list [(cat, x0, y0, x1, y1)], doc_category) for rows where keep(row_index)."""
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(fetch(fi))
    r = 0
    fmt = None
    for batch in pf.iter_batches(batch_size=8, columns=["image", "bboxes", "category_id", "metadata"]):
        d = batch.to_pydict()
        for k in range(len(d["bboxes"])):
            ri = r
            r += 1
            if not keep(ri):
                continue
            img = Image.open(io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
            w, h = img.size
            bbs = d["bboxes"][k] or []
            if fmt is None and bbs:
                if any(b[2] < b[0] or b[3] < b[1] for b in bbs):
                    fmt = "xywh"
                elif any(b[0] + b[2] > w + 2 or b[1] + b[3] > h + 2 for b in bbs):
                    fmt = "xyxy"
                else:
                    fmt = "xywh"
                print("DocLayNet bbox format", fmt, "sample", bbs[:2], "image", w, h, flush=True)
            gt = []
            for b, c in zip(bbs, d["category_id"][k] or []):
                if fmt == "xyxy":
                    x0, y0, x1, y1 = b
                else:
                    x0, y0, x1, y1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
                gt.append((DLN_CATS.get(int(c), str(c)), x0, y0, x1, y1))
            md = d["metadata"][k] or {}
            pid = "dln:%d:%s" % (fi, md.get("page_hash") or ri)
            yield ri, pid, img, gt, md.get("doc_category"), w, h
