"""PP-DocLayout_plus-L ONNX (HF PaddlePaddle/PP-DocLayout_plus-L_onnx @feb7461) on the RF-DETR / heron pages, and an
ONNX-vs-native-Paddle parity check.

Pre/post-processing follow inference.yml and PaddleX 3.7.2 (object_detection/processors.py): RGB, cv2 resize to 800x800
(keep_ratio false, interp 2 = cubic), /255 with mean 0 / std 1, CHW; inputs image / im_shape / scale_factor; output rows
[class, score, x0, y0, x1, y1] in original pixels; DetPostProcess.apply = score threshold, layout_nms off (module default),
then 'filter_large_image' (drop 'image' boxes covering > 82% (portrait) / 93% of the page when more than one box).
usage: r4_pp.py run {dln,odb} SHARD NSHARDS | r4_pp.py parity N"""
import os, sys, json, time
from r4_common import *
import numpy as np

REPO, REV = "PaddlePaddle/PP-DocLayout_plus-L_onnx", "feb74619326f634e0e883218598096a3733ad9f7"
NATIVE, NATIVE_REV = "PaddlePaddle/PP-DocLayout_plus-L", "aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d"
LIMIT = int(os.environ.get("TB_LIMIT") or 0)


class PPPlus:
    def __init__(self):
        import yaml, onnxruntime as ort
        from huggingface_hub import hf_hub_download
        self.onnx = hf_hub_download(REPO, "inference.onnx", revision=REV)
        cfg = yaml.safe_load(open(hf_hub_download(REPO, "inference.yml", revision=REV)))
        rs = [t for t in cfg["Preprocess"] if t["type"] == "Resize"][0]
        nm = [t for t in cfg["Preprocess"] if t["type"] == "NormalizeImage"][0]
        assert rs["keep_ratio"] is False and rs["interp"] == 2, rs
        self.th, self.tw = rs["target_size"]
        self.mean, self.std = np.array(nm["mean"], np.float32), np.array(nm["std"], np.float32)
        self.labels = cfg["label_list"]
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
        self.s = ort.InferenceSession(self.onnx, so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.s.get_inputs()]
        print("pp-doclayout_plus-L inputs", [(i.name, i.shape) for i in self.s.get_inputs()], "outputs", [(o.name, o.shape) for o in self.s.get_outputs()],
              "labels", self.labels, "size", (self.th, self.tw), flush=True)

    def __call__(self, img, threshold=0.05):
        import cv2
        a = np.asarray(img.convert("RGB")); h, w = a.shape[:2]
        x = cv2.resize(a, (self.tw, self.th), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)
        feed = {}
        for n in self.inputs:
            if n == "im_shape": feed[n] = np.array([[self.th, self.tw]], np.float32)
            elif n == "scale_factor": feed[n] = np.array([[self.th / h, self.tw / w]], np.float32)
            else: feed[n] = x
        o = self.s.run(None, feed)
        det = o[0]
        if len(o) > 1 and np.asarray(o[1]).size == 1: det = det[: int(np.asarray(o[1]).reshape(-1)[0])]
        det = det[(det[:, 1] > threshold) & (det[:, 0] > -1)]
        if len(det) > 1:
            thr = 0.82 if w > h else 0.93
            ii = self.labels.index("image")
            keep = []
            for b in det:
                if int(b[0]) == ii:
                    x0, y0, x1, y1 = max(0, b[2]), max(0, b[3]), min(w, b[4]), min(h, b[5])
                    if (x1 - x0) * (y1 - y0) > thr * w * h: continue
                keep.append(b)
            if keep: det = np.array(keep)
        return [(self.labels[int(b[0])], float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])) for b in det]


def run(ds, k, n):
    m = PPPlus()
    import r3_detect as RD
    gen = RD.dln_pages(k, n) if ds == "dln" else RD.odb_pages(k, n)
    os.makedirs(R4OUT, exist_ok=True)
    of = os.path.join(R4OUT, f"ppplus-{ds}-{k}.jsonl"); cnt = 0
    with open(of, "w", encoding="utf-8") as f:
        for _, pid, img, W, H in gen:
            t = time.perf_counter(); dets = m(img); sec = time.perf_counter() - t
            sx, sy = W / img.size[0], H / img.size[1]
            dets = [[d[0], round(d[1], 4), round(d[2] * sx, 1), round(d[3] * sy, 1), round(d[4] * sx, 1), round(d[5] * sy, 1)] for d in sorted(dets, key=lambda d: -d[1])]
            f.write(json.dumps(dict(ds=ds, id=pid, W=W, H=H, sec=sec, dets=dets)) + "\n"); cnt += 1
            if cnt % 100 == 0: print(ds, k, cnt, round(sec, 3), "s/page", flush=True)
            if LIMIT and cnt >= LIMIT: break
    print("done", ds, k, cnt, flush=True)


def parity(N):
    """native PaddleX/PaddleOCR (paddle inference, same weights) vs our ONNX path on N pages (half DocLayNet, half OmniDocBench)"""
    from huggingface_hub import snapshot_download
    from paddleocr import LayoutDetection
    ndir = snapshot_download(NATIVE, revision=NATIVE_REV)
    nat = LayoutDetection(model_name="PP-DocLayout_plus-L", model_dir=ndir)
    ox = PPPlus()
    import r3_detect as RD, itertools
    pages = list(itertools.islice(RD.dln_pages(0, 6), N // 2)) + list(itertools.islice(RD.odb_pages(0, 1), N - N // 2))
    tmp = os.path.join(R4OUT, "parity_img.png"); os.makedirs(R4OUT, exist_ok=True)
    tot_n = tot_o = mat = 0; sdiff = []; ious = []; rows = []; tn = to = 0.0
    for _, pid, img, W, H in pages:
        img.save(tmp)
        t = time.perf_counter(); r = list(nat.predict(tmp, batch_size=1, threshold=0.3, layout_nms=False))[0]; tn += time.perf_counter() - t
        nb = [(b["label"], float(b["score"]), *map(float, b["coordinate"])) for b in r["boxes"]]
        t = time.perf_counter(); ob = ox(img, threshold=0.3); to += time.perf_counter() - t
        used = set(); m_ = 0
        for o in ob:
            best, bj = 0.0, -1
            for j, nbx in enumerate(nb):
                if j in used or nbx[0] != o[0]: continue
                v = iou(o[2:6], nbx[2:6])
                if v > best: best, bj = v, j
            if bj >= 0 and best >= 0.95:
                used.add(bj); m_ += 1; sdiff.append(abs(o[1] - nb[bj][1])); ious.append(best)
        tot_n += len(nb); tot_o += len(ob); mat += m_
        rows.append(dict(id=pid, native=len(nb), onnx=len(ob), matched=m_))
    res = dict(pages=len(pages), native_boxes=tot_n, onnx_boxes=tot_o, matched_iou95_same_label=mat,
               matched_of_native=mat / max(1, tot_n), matched_of_onnx=mat / max(1, tot_o),
               max_score_diff=max(sdiff) if sdiff else None, mean_score_diff=float(np.mean(sdiff)) if sdiff else None,
               mean_iou=float(np.mean(ious)) if ious else None, native_ms=1000 * tn / len(pages), onnx_ms=1000 * to / len(pages), threshold=0.3)
    print(json.dumps(res, indent=1))
    json.dump(dict(summary=res, pages=rows), open(os.path.join(R4OUT, "pp_parity.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else: parity(int(sys.argv[2]))
