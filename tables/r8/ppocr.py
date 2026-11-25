"""(Copied from nithinkadari-ms/ci-lab-eval media/ocr/ppocr.py, round 4.)
PP-OCR text detection + CTC recognition, ported from docling.rs 2.84 (crates/docling-pdf/src/ocr_det.rs,
ocr_prep.rs, ocr.rs): RapidOCR's DB post-processing (thresh .3, box_thresh .5, unclip 1.6 by the rectangle rule,
shorter side up to 736, longer side capped at 2000, BGR input), rotated-rectangle crops (bicubic, replicated
border, tall crops turned), recognition at height 48 with RGB planes normalised to [-1, 1], greedy CTC with
index 0 = blank and a trailing space class, mean emitted-character probability as confidence, and the 0.5
text_score docling uses to drop page-OCR lines."""
import math

import cv2
import numpy as np
import onnxruntime as ort

LIMIT_SIDE = 736
MAX_SIDE = 2000
THRESH, BOX_THRESH, UNCLIP = 0.3, 0.5, 1.6
MIN_SIZE = 3.0
REC_H = 48
TEXT_SCORE = 0.5


def session(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.enable_cpu_mem_arena = False
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def det_size(w, h, cap=MAX_SIDE):
    ratio = LIMIT_SIDE / min(w, h) if min(w, h) < LIMIT_SIDE else 1.0
    if cap > 0 and max(w, h) * ratio > cap:
        ratio = cap / max(w, h)

    def r32(v):
        return int(math.floor(int(v) / 32.0 + 0.5) * 32)
    return r32(w * ratio), r32(h * ratio)


def order_corners(pts):
    pts = sorted(pts, key=lambda p: p[0])
    left, right = sorted(pts[:2], key=lambda p: p[1]), sorted(pts[2:], key=lambda p: p[1])
    return [left[0], right[0], right[1], left[1]]


def _box_score(prob, quad):
    h, w = prob.shape
    q = np.array(quad, dtype=np.float32)
    x0, x1 = int(np.clip(np.floor(q[:, 0].min()), 0, w - 1)), int(np.clip(np.ceil(q[:, 0].max()), 0, w - 1))
    y0, y1 = int(np.clip(np.floor(q[:, 1].min()), 0, h - 1)), int(np.clip(np.ceil(q[:, 1].max()), 0, h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    q[:, 0] -= x0
    q[:, 1] -= y0
    cv2.fillPoly(mask, [q.astype(np.int32)], 1)
    return float(cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0])


def db_boxes(prob, dest_w, dest_h):
    h, w = prob.shape
    seg = (prob > THRESH).astype(np.uint8)
    mask = cv2.dilate(seg, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts[:1000]:
        rect = cv2.minAreaRect(c)
        (cw, ch) = rect[1]
        if min(cw, ch) < MIN_SIZE:
            continue
        quad = order_corners([tuple(p) for p in cv2.boxPoints(rect)])
        score = _box_score(prob, quad)
        if score < BOX_THRESH:
            continue
        per = 2 * (cw + ch)
        if per < 1e-6:
            continue
        d = cw * ch * UNCLIP / per
        big = (rect[0], (cw + 2 * d, ch + 2 * d), rect[2])
        if min(cw, ch) + 2 * d < MIN_SIZE + 2:
            continue
        q = order_corners([tuple(p) for p in cv2.boxPoints(big)])
        q = [(float(np.clip(round(x / w * dest_w), 0, dest_w)), float(np.clip(round(y / h * dest_h), 0, dest_h)))
             for x, y in q]
        rw = math.floor(math.dist(q[0], q[1]))
        rh = math.floor(math.dist(q[0], q[3]))
        if rw <= 3 or rh <= 3:
            continue
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        out.append({"quad": q, "box": [min(xs), min(ys), max(xs), max(ys)], "score": score})
    out.sort(key=lambda b: (b["box"][1], b["box"][0]))
    for i in range(len(out) - 1):
        j = i
        while j >= 0 and abs(out[j + 1]["box"][1] - out[j]["box"][1]) < 10 and out[j + 1]["box"][0] < out[j]["box"][0]:
            out[j], out[j + 1] = out[j + 1], out[j]
            j -= 1
    return out


class Detector:
    def __init__(self, path, threads=1, cap=MAX_SIDE):
        self.s = session(path, threads)
        self.name = self.s.get_inputs()[0].name
        self.cap = cap

    def __call__(self, rgb, region=False):
        """region=True: docling.rs det_crop_size for a layout-region crop (own pixels, sides rounded up to 32)."""
        H, W = rgb.shape[:2]
        if region:
            f = min(1.0, self.cap / max(W, H)) if self.cap > 0 else 1.0
            rw, rh = max(1, math.ceil(W * f / 32.0)) * 32, max(1, math.ceil(H * f / 32.0)) * 32
        else:
            rw, rh = det_size(W, H, self.cap)
        if rw <= 0 or rh <= 0:
            return []
        im = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rw, rh) != (W, H) else rgb
        x = (im[:, :, ::-1].astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        prob = self.s.run(None, {self.name: np.ascontiguousarray(x)})[0][0, 0]
        return db_boxes(prob, W, H)


def crop(rgb, quad):
    q = np.array(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w <= 0 or h <= 0:
        return None
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32))
    out = cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h / w >= 1.5:
        out = np.rot90(out)
    return np.ascontiguousarray(out)


def load_dict(path):
    return [""] + [l.rstrip("\r\n") for l in open(path, encoding="utf-8")] + [" "]


def dict_from_yml(path):
    import yaml
    y = yaml.safe_load(open(path, encoding="utf-8"))
    chars = y["PostProcess"]["character_dict"]
    return [""] + [str(c) for c in chars] + [" "]


class Recognizer:
    """Paddle CTC recognizer (PP-OCRv6 small/medium, en_PP-OCRv5 mobile) with docling.rs preprocessing."""

    def __init__(self, path, chars, threads=1):
        self.s = session(path, threads)
        self.name = self.s.get_inputs()[0].name
        self.chars = chars
        self.checked = False

    def one(self, rgb):
        h, w = rgb.shape[:2]
        if h == 0 or w == 0:
            return "", 0.0
        nw = int(min(max(round(w * REC_H / h), 8), 16384))
        im = cv2.resize(rgb, (nw, REC_H), interpolation=cv2.INTER_LINEAR)
        x = (im.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        p = self.s.run(None, {self.name: np.ascontiguousarray(x)})[0][0]
        if not self.checked:
            if p.shape[-1] != len(self.chars):
                raise RuntimeError(f"class count {p.shape[-1]} != dict {len(self.chars)}")
            self.checked = True
        if p.min() < 0 or abs(float(p[0].sum()) - 1.0) > 0.05:
            e = np.exp(p - p.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
        idx = p.argmax(axis=1)
        out, conf, prev = [], [], 0
        for t, c in enumerate(idx):
            if c != prev and c != 0:
                out.append(self.chars[c])
                conf.append(float(p[t, c]))
            prev = c
        return "".join(out).strip(), (sum(conf) / len(conf) if conf else 0.0)

    def __call__(self, crops):
        return [self.one(c) for c in crops]
