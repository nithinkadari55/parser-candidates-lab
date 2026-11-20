import os, sys, io, json, time, glob, urllib.request
import numpy as np
from PIL import Image
import paths
from paths import MODELS, OUT
import onnxruntime as ort
ort.set_default_logger_severity(3)

DLN_REPO = "docling-project/DocLayNet-v1.1"
DLN_FILES = ["data/test-00000-of-00002-635b47e9044a436c.parquet", "data/test-00001-of-00002-b2396409212624ed.parquet"]
DLN_CATS = {1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item", 5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header", 9: "Table", 10: "Text", 11: "Title"}
ODB = os.environ.get("ODB_DIR", os.path.expanduser("~/odb"))
D = os.path.join(MODELS, "det")


def so(threads):
    o = ort.SessionOptions()
    if threads: o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    return o


def fetch():
    from huggingface_hub import hf_hub_download
    os.makedirs(D, exist_ok=True)
    files = {
        "pico.onnx": "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet_layout_1x_table.onnx",
    }
    for fn, url in files.items():
        p = os.path.join(D, fn)
        if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    hf_hub_download("PaddlePaddle/PicoDet_layout_1x_table", "inference.yml", local_dir=os.path.join(D, "pico"))
    hf_hub_download("Xenova/table-transformer-detection", "onnx/model_fp16.onnx", local_dir=os.path.join(D, "tatrdet"))
    hf_hub_download("Xenova/table-transformer-detection", "config.json", local_dir=os.path.join(D, "tatrdet"))
    for fn in ("yolox_tiny.onnx", "yolox_l0.05_quantized.onnx", "label_map.json", "LICENSE.txt"):
        hf_hub_download("unstructuredio/yolo_x_layout", fn, local_dir=os.path.join(D, "yolox"))
    for r, d, fs in os.walk(D):
        for f in fs: print("det file", os.path.relpath(os.path.join(r, f), D), os.path.getsize(os.path.join(r, f)))


class Pico:
    def __init__(self, threads):
        import yaml
        cfg = yaml.safe_load(open(os.path.join(D, "pico", "inference.yml")))
        rs = [t for t in cfg["Preprocess"] if t["type"] == "Resize"][0]
        nm = [t for t in cfg["Preprocess"] if t["type"] == "NormalizeImage"][0]
        self.th, self.tw = rs["target_size"]
        self.mean, self.std = np.array(nm["mean"], np.float32), np.array(nm["std"], np.float32)
        self.s = ort.InferenceSession(os.path.join(D, "pico.onnx"), so(threads), providers=["CPUExecutionProvider"])
        self.inputs = [(i.name, i.shape) for i in self.s.get_inputs()]
        print("pico inputs", self.inputs, "outputs", [(o.name, o.shape) for o in self.s.get_outputs()], flush=True)

    def __call__(self, img):
        w, h = img.size
        x = np.asarray(img.resize((self.tw, self.th), Image.BICUBIC)).astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)
        feed = {}
        for n, sh in self.inputs:
            if n in ("image", "x", "images"): feed[n] = x
            elif n == "scale_factor": feed[n] = np.array([[self.th / h, self.tw / w]], np.float32)
            elif n == "im_shape": feed[n] = np.array([[self.th, self.tw]], np.float32)
        out = self.s.run(None, feed)
        bx = out[0].reshape(-1, out[0].shape[-1])
        dets = []
        for b in bx:
            if b.shape[0] >= 6 and b[1] > 0.05:
                dets.append(("Table", float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])))
        if not any(n == "scale_factor" for n, _ in self.inputs):
            dets = [(d[0], d[1], d[2] * w / self.tw, d[3] * h / self.th, d[4] * w / self.tw, d[5] * h / self.th) for d in dets]
        return dets


class TatrDet:
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, threads):
        self.s = ort.InferenceSession(os.path.join(D, "tatrdet", "onnx", "model_fp16.onnx"), so(threads), providers=["CPUExecutionProvider"])
        self.f16 = self.s.get_inputs()[0].type == "tensor(float16)"
        cfg = json.load(open(os.path.join(D, "tatrdet", "config.json")))
        self.id2 = {int(k): v for k, v in cfg["id2label"].items()}

    def __call__(self, img):
        w, h = img.size; sc = 800 / max(w, h)
        r = img.resize((int(round(sc * w)), int(round(sc * h))), Image.BILINEAR)
        x = ((np.asarray(r).astype(np.float32) / 255 - self.MEAN) / self.STD).transpose(2, 0, 1)[None].copy()
        lg, bx = self.s.run(None, {"pixel_values": x.astype(np.float16) if self.f16 else x})
        lg = lg[0].astype(np.float32); bx = bx[0].astype(np.float32)
        e = np.exp(lg - lg.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
        dets = []
        for pr, b in zip(p, bx):
            k = int(pr[:-1].argmax()); sc_ = float(pr[k])
            if sc_ < 0.05: continue
            cx, cy, bw, bh = b
            dets.append(("Table", sc_, float((cx - bw / 2) * w), float((cy - bh / 2) * h), float((cx + bw / 2) * w), float((cy + bh / 2) * h), self.id2.get(k, str(k))))
        return dets


class Yolox:
    def __init__(self, threads, fn):
        self.s = ort.InferenceSession(os.path.join(D, "yolox", fn), so(threads), providers=["CPUExecutionProvider"])
        self.iname = self.s.get_inputs()[0].name
        self.shape = (1024, 768)
        self.labels = {int(k): v for k, v in json.load(open(os.path.join(D, "yolox", "label_map.json"))).items()}

    def __call__(self, img):
        a = np.asarray(img)
        H, W = self.shape
        r = min(H / a.shape[0], W / a.shape[1])
        rs = np.asarray(img.resize((int(a.shape[1] * r), int(a.shape[0] * r)), Image.BILINEAR)).astype(np.float32)
        pad = np.full((H, W, 3), 114.0, np.float32); pad[:rs.shape[0], :rs.shape[1]] = rs
        x = pad[:, :, ::-1].transpose(2, 0, 1)[None].copy()
        out = self.s.run(None, {self.iname: x})[0][0]
        grids, strides = [], []
        for st in (8, 16, 32):
            hs, ws = H // st, W // st
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2)); strides.append(np.full((hs * ws, 1), st))
        g = np.concatenate(grids, 0); s_ = np.concatenate(strides, 0)
        out[:, :2] = (out[:, :2] + g) * s_; out[:, 2:4] = np.exp(out[:, 2:4]) * s_
        boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], 1) / r
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1); sc = scores.max(1)
        keep = sc > 0.1
        dets = []
        for c in np.unique(cls[keep]):
            idx = np.where(keep & (cls == c))[0]
            for i in nms(boxes[idx], sc[idx], 0.45):
                j = idx[i]
                dets.append((self.labels.get(int(c), str(c)), float(sc[j]), *[float(v) for v in boxes[j]]))
        return dets


def nms(b, s, thr):
    order = s.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(b[i, 0], b[order[1:], 0]); yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2]); yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def make(name, threads=0):
    return {"pico": lambda: Pico(threads), "tatrdet": lambda: TatrDet(threads), "yolox_tiny": lambda: Yolox(threads, "yolox_tiny.onnx"),
            "yolox_l": lambda: Yolox(threads, "yolox_l0.05_quantized.onnx")}[name]()


def dln_pages(k, n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    half = n // 2; fi = k % 2
    p = hf_hub_download(DLN_REPO, DLN_FILES[fi], repo_type="dataset", local_dir=os.path.expanduser("~/dln"))
    pf = pq.ParquetFile(p); r = 0
    for batch in pf.iter_batches(batch_size=8, columns=["image", "metadata"]):
        d = batch.to_pydict()
        for j in range(len(d["image"])):
            ri = r; r += 1
            if ri % half != k // 2: continue
            img = Image.open(io.BytesIO(d["image"][j]["bytes"])).convert("RGB")
            md = d["metadata"][j] or {}
            yield "dln", "dln:%d:%s" % (fi, md.get("page_hash") or ri), img, img.size[0], img.size[1]


def odb_pages(k, n):
    d = json.load(open(os.path.join(ODB, "OmniDocBench.json"), encoding="utf-8"))
    recs = [r for r in d if r["page_info"]["page_attribute"].get("language") == "english"]
    for i, rec in enumerate(recs):
        if i % n != k: continue
        pi = rec["page_info"]; p = os.path.join(ODB, "images", pi["image_path"])
        if not os.path.exists(p): continue
        img = Image.open(p).convert("RGB")
        yield "odb", "odb:" + os.path.basename(pi["image_path"]), img, float(pi["width"]), float(pi["height"])


def fetch_odb():
    from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(ODB, exist_ok=True)
    hf_hub_download("opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset", local_dir=ODB)
    d = json.load(open(os.path.join(ODB, "OmniDocBench.json"), encoding="utf-8"))
    en = [p["page_info"]["image_path"] for p in d if p["page_info"]["page_attribute"].get("language") == "english"]

    def get(n):
        for _ in range(4):
            try:
                hf_hub_download("opendatalab/OmniDocBench", "images/" + n, repo_type="dataset", local_dir=ODB); return 0
            except Exception:
                pass
        return 1
    with ThreadPoolExecutor(8) as ex:
        print("odb english", len(en), "failed", sum(ex.map(get, en)))


def main():
    if sys.argv[1] == "fetch": return fetch()
    if sys.argv[1] == "fetch_odb": return fetch_odb()
    name, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    m = make(name)
    os.makedirs(os.path.join(OUT, "r3det"), exist_ok=True)
    of = os.path.join(OUT, "r3det", f"det-{name}-{k}.jsonl")
    cnt = 0
    with open(of, "w", encoding="utf-8") as f:
        lim = int(os.environ.get("TB_LIMIT") or 0)
        for gen in (dln_pages(k, n), odb_pages(k, n)):
            c0 = 0
            for ds, pid, img, W, H in gen:
                c0 += 1
                if lim and c0 > lim: break
                t = time.perf_counter(); dets = m(img); sec = time.perf_counter() - t
                sx, sy = W / img.size[0], H / img.size[1]
                dets = [[d[0], round(d[1], 4), round(d[2] * sx, 1), round(d[3] * sy, 1), round(d[4] * sx, 1), round(d[5] * sy, 1)] + list(d[6:]) for d in dets]
                f.write(json.dumps(dict(ds=ds, id=pid, W=W, H=H, sec=sec, dets=dets)) + "\n"); cnt += 1
    print(name, k, n, "pages", cnt, flush=True)


if __name__ == "__main__":
    main()
