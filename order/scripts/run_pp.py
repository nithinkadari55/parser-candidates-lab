import json, os, sys, time
import numpy as np, cv2, onnxruntime as ort
from PIL import Image

PPDIR = os.environ.get("PP_DIR", os.path.join(os.path.expanduser("~"), "ppdlv3"))
LABELS = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image", "image", "inline_formula", "number", "paragraph_title", "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"]
_sess = None


def sess():
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
        _sess = ort.InferenceSession(os.path.join(PPDIR, "inference.onnx"), so, providers=["CPUExecutionProvider"])
    return _sess


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def infer(a):
    h, w = a.shape[:2]
    x = cv2.resize(a, (800, 800), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    o = sess().run(None, {"image": x, "im_shape": np.array([[800, 800]], np.float32), "scale_factor": np.array([[800 / h, 800 / w]], np.float32)})
    n = int(o[1][0])
    return o[0][:n]


def post(b, thr=0.5):
    b = b[(b[:, 1] > thr) & (b[:, 0] > -1)]
    b = b[np.argsort(-b[:, 1])]
    keep = []
    for i in range(len(b)):
        ok = True
        for j in keep:
            v = iou(b[i, 2:6], b[j, 2:6])
            if (b[i, 0] == b[j, 0] and v > 0.6) or v > 0.98:
                ok = False
                break
        if ok:
            keep.append(i)
    b = b[keep]
    return b[np.argsort(b[:, 6])]


def items(a):
    return [dict(label=LABELS[int(r[0])], score=float(r[1]), box=[float(v) for v in r[2:6]], order=float(r[6])) for r in post(infer(a))]


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import odb
    k, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    pages = odb.shard([p for p in odb.pages() if len(p["boxes"]) >= 2], k, n)
    with open(out, "w", encoding="utf-8") as f:
        for p in pages:
            a = np.asarray(Image.open(os.path.join(odb.D, "images", p["img"])).convert("RGB"))
            t = time.perf_counter()
            its = items(a)
            f.write(json.dumps(dict(img=p["img"], sec=time.perf_counter() - t, items=its)) + "\n")
            f.flush()
    print("pp", k, n, len(pages))
