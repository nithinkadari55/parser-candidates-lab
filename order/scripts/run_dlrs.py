"""docling-rs 2.84.0 end to end (heron + OCR + docling.rs reading order) on page images.
usage: run_dlrs.py SHARD NSHARDS OUT.jsonl   (sets from DLRS_SETS, default odb,drdoc,dln)"""
import json, os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "layout"))
import datasets, run_heron
from PIL import Image


def convert(conv, path, iw, ih, W, H):
    t = time.perf_counter()
    doc = conv.convert(path).document.export_to_dict()
    sec = time.perf_counter() - t
    pg = doc["pages"]["1"]["size"]
    sx, sy = iw / pg["width"] * (W / iw), ih / pg["height"] * (H / ih)
    its = []
    for it in run_heron.items(doc):
        bx = run_heron.to_topleft(it.pop("bbox"), pg["height"])
        it["box"] = [round(bx[0] * sx, 1), round(min(bx[1], bx[3]) * sy, 1), round(bx[2] * sx, 1), round(max(bx[1], bx[3]) * sy, 1)]
        its.append(it)
    return sec, its


def main():
    k, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sets = os.environ.get("DLRS_SETS", "odb,drdoc,dln").split(",")
    conv = run_heron.setup()
    with open(out, "w", encoding="utf-8") as f:
        def emit(ds, pid, path, W, H):
            iw, ih = Image.open(path).size
            try:
                sec, its = convert(conv, path, iw, ih, W, H)
                f.write(json.dumps(dict(ds=ds, id=pid, W=W, H=H, sec=sec, items=its)) + "\n")
            except Exception as e:
                f.write(json.dumps(dict(ds=ds, id=pid, error=str(e)[:300])) + "\n")
            f.flush()
        for ds in ("odb", "drdoc"):
            if ds in sets:
                for p in datasets.shard(datasets.pages(ds), k, n):
                    emit(ds, p["id"], p["img"], p["W"], p["H"])
        if "dln" in sets:
            import dln
            step = int(os.environ.get("DLRS_DLN_STEP", "10"))
            with tempfile.TemporaryDirectory() as td:
                for fi in (0, 1):
                    for ri, pid, img, gt, cat, w, h in dln.rows(fi, lambda r: r % step == 0 and (r // step) % n == k):
                        pth = os.path.join(td, "p.png")
                        img.save(pth)
                        emit("dln", pid, pth, w, h)
    print("dlrs", k, n)


if __name__ == "__main__":
    main()
