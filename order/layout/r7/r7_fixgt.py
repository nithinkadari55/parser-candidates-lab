"""Rebuild the ground truth of the D4LA and M6Doc page lists: their YOLO label lines are 4-corner
polygons (class x1 y1 x2 y2 x3 y3 x4 y4, normalised), not centre boxes. Reads PAGES.jsonl (from a
heron-r7 run), downloads only the label files and writes the fixed PAGES.jsonl in place.
usage: r7_fixgt.py SET PAGES.jsonl"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r7_data import D4LA, D4LA_COARSE  # noqa: E402


def main():
    from huggingface_hub import hf_hub_download
    ds, path = sys.argv[1], sys.argv[2]
    repo = {"d4la": "juliozhao/D4LA", "m6doc": "juliozhao/M6Doc"}[ds]
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    root = os.path.join(os.path.dirname(os.path.abspath(path)), "_labels")

    def one(r):
        rel = r["img"].replace("\\", "/").split("/_hf/", 1)[1]
        lab = rel.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        for attempt in range(5):
            try:
                lp = hf_hub_download(repo, lab, repo_type="dataset", local_dir=root)
                break
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    return None
                time.sleep(2 + 3 * attempt)
        w, h = r["W"], r["H"]
        gt = []
        for line in open(lp, encoding="utf-8"):
            v = line.split()
            if len(v) < 5:
                continue
            c = int(float(v[0]))
            xy = [float(x) for x in v[1:]]
            if len(xy) >= 8:
                xs, ys = xy[0::2], xy[1::2]
                box = [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]
            else:
                cx, cy, bw, bh = xy[:4]
                box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            if ds == "d4la":
                name = D4LA[c] if c < len(D4LA) else "Other"
                cls = D4LA_COARSE.get(name, "Text")
            else:
                cls = "Any"
            gt.append([cls] + box)
        r = dict(r)
        r["gt"] = gt
        return r

    with ThreadPoolExecutor(8) as ex:
        fixed = [r for r in ex.map(one, rows) if r]
    with open(path, "w", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r) + "\n")
    print(ds, "fixed", len(fixed), "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
