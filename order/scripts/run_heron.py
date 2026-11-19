import json, os, sys, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odb

import docling_rs as d
from docling_rs import models as dm


def setup():
    root = os.environ.get("DLRS_CACHE", os.path.join(os.path.expanduser("~"), "dlrs"))
    dm.ensure_env(root)
    os.environ.setdefault("DOCLING_RS_FP32", "1")
    return d.DocumentConverter(do_table_structure=False)


def items(doc):
    def res(ref):
        _, kind, idx = ref.split("/")
        return doc[kind][int(idx)]
    out = []

    def walk(node, inside_graphic, furn):
        for ch in node.get("children", []):
            it = res(ch["$ref"])
            lab = it.get("label")
            pr = it.get("prov") or []
            if pr and (not inside_graphic or lab in ("caption", "footnote")):
                out.append(dict(label=lab, bbox=pr[0]["bbox"], page_no=pr[0]["page_no"],
                                furniture=furn or it.get("content_layer") == "furniture", text=(it.get("text") or "")[:80]))
            walk(it, inside_graphic or lab in ("picture", "table", "chart"), furn)
    walk(doc["body"], False, False)
    walk(doc["furniture"], False, True)
    return out


def to_topleft(b, ph):
    if str(b.get("coord_origin", "BOTTOMLEFT")).upper().endswith("BOTTOMLEFT"):
        return [b["l"], ph - b["t"], b["r"], ph - b["b"]]
    return [b["l"], b["t"], b["r"], b["b"]]


if __name__ == "__main__":
    k, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    names = [l.strip() for l in open(os.path.join(os.path.dirname(HERE), "data", "heron_subset.txt"), encoding="utf-8") if l.strip()]
    names = odb.shard(names, k, n)
    conv = setup()
    with open(out, "w", encoding="utf-8") as f:
        for nm in names:
            path = os.path.join(odb.D, "images", nm)
            iw, ih = Image.open(path).size
            t = time.perf_counter()
            try:
                doc = conv.convert(path).document.export_to_dict()
            except Exception as e:
                f.write(json.dumps(dict(img=nm, error=str(e)[:300])) + "\n")
                continue
            dt = time.perf_counter() - t
            pg = doc["pages"]["1"]["size"]
            sx, sy = iw / pg["width"], ih / pg["height"]
            its = []
            for it in items(doc):
                bx = to_topleft(it.pop("bbox"), pg["height"])
                it["box"] = [bx[0] * sx, min(bx[1], bx[3]) * sy, bx[2] * sx, max(bx[1], bx[3]) * sy]
                its.append(it)
            f.write(json.dumps(dict(img=nm, sec=dt, items=its)) + "\n")
            f.flush()
            print(nm, round(dt, 1), len(its), flush=True)
