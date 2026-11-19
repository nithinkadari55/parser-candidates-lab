"""ONNX layout detectors with their own pre/post-processing. Every model returns raw detections
[(native_label, score, x0, y0, x1, y1)] in the pixel space of the input image."""
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

HOME = os.path.expanduser("~")
DOCLING17 = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture", "Section-header", "Table",
             "Text", "Title", "Document Index", "Code", "Checkbox-Selected", "Checkbox-Unselected", "Form", "Key-Value Region"]
PPL23 = ["paragraph_title", "image", "text", "number", "abstract", "content", "figure_title", "formula", "table", "table_title",
         "reference", "doc_title", "footnote", "header", "algorithm", "footer", "seal", "chart_title", "chart", "formula_number",
         "header_image", "footer_image", "aside_text"]
PPV3 = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title", "figure_title", "footer",
        "footer_image", "footnote", "formula_number", "header", "header_image", "image", "inline_formula", "number",
        "paragraph_title", "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"]

PATHS = {
    "heron": os.environ.get("HERON_ONNX", os.path.join(HOME, "dlrs", "models", "layout_heron.onnx")),
    "heron_int8": os.environ.get("HERON8_ONNX", os.path.join(HOME, "dlrs", "models", "layout_heron_int8.onnx")),
    "egret": os.environ.get("EGRET_ONNX", os.path.join(HOME, "lmodels", "egret", "model.onnx")),
    "ppl": os.environ.get("PPL_ONNX", os.path.join(HOME, "lmodels", "ppl", "pp-doclayout-l.onnx")),
    "ppv3": os.environ.get("PPV3_ONNX", os.path.join(HOME, "ppdlv3", "inference.onnx")),
}


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class RTDetrHF:
    """docling heron (RT-DETRv2) and egret (D-FINE) exports: pixel_values 640x640, /255, no normalize,
    PIL bilinear resize; outputs logits [1,Q,C] and pred_boxes [1,Q,4] (cx,cy,w,h normalised)."""

    def __init__(self, name):
        self.name = name
        self.s = session(PATHS[name])
        self.inp = self.s.get_inputs()[0].name
        self.outs = [o.name for o in self.s.get_outputs()]
        self.labels = DOCLING17

    def __call__(self, img, min_score=0.05, topk=150):
        w, h = img.size
        x = np.asarray(img.convert("RGB").resize((640, 640), Image.BILINEAR), dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        o = dict(zip(self.outs, self.s.run(None, {self.inp: x})))
        logits = o.get("logits")
        boxes = o.get("pred_boxes")
        if logits is None or boxes is None:
            arrs = list(o.values())
            boxes = next(a for a in arrs if a.ndim == 3 and a.shape[-1] == 4)
            logits = next(a for a in arrs if a.ndim == 3 and a.shape[-1] != 4)
        logits, boxes = logits[0], boxes[0]
        q, c = logits.shape
        sc = 1 / (1 + np.exp(-logits.reshape(-1)))
        top = np.argsort(-sc)[:q]
        out = []
        for idx in top:
            s = float(sc[idx])
            if s < min_score:
                break
            qi, ci = divmod(int(idx), c)
            cx, cy, bw, bh = boxes[qi]
            out.append((self.labels[ci] if ci < len(self.labels) else str(ci), s,
                        float((cx - bw / 2) * w), float((cy - bh / 2) * h), float((cx + bw / 2) * w), float((cy + bh / 2) * h)))
        return out[:topk]


class PaddleDet:
    """PaddleDetection RT-DETR exports (PP-DocLayout-L, PP-DocLayoutV3): inputs image / im_shape /
    scale_factor, RGB, plain resize to size x size (bicubic), /255; output [N, 6 or 7] =
    class, score, x0, y0, x1, y1 (, reading-order index for V3) in original pixels."""

    def __init__(self, name):
        import cv2
        self.cv2 = cv2
        self.name = name
        self.s = session(PATHS[name])
        self.inputs = {i.name: i for i in self.s.get_inputs()}
        self.size = 800 if name == "ppv3" else 640
        self.labels = PPV3 if name == "ppv3" else PPL23

    def __call__(self, img, min_score=0.05, topk=150):
        a = np.asarray(img.convert("RGB"))
        h, w = a.shape[:2]
        x = self.cv2.resize(a, (self.size, self.size), interpolation=self.cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        feed = {}
        for n in self.inputs:
            if n == "image" or n.startswith("x") or n == "images":
                feed[n] = x
            elif n == "im_shape":
                feed[n] = np.array([[self.size, self.size]], np.float32)
            elif n == "scale_factor":
                feed[n] = np.array([[self.size / h, self.size / w]], np.float32)
        o = self.s.run(None, feed)
        det = o[0]
        if len(o) > 1 and np.asarray(o[1]).size == 1:
            det = det[: int(np.asarray(o[1]).reshape(-1)[0])]
        out = []
        for r in det:
            ci, s = int(r[0]), float(r[1])
            if ci < 0 or s < min_score:
                continue
            item = (self.labels[ci] if ci < len(self.labels) else str(ci), s, float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            if det.shape[1] > 6:
                item = item + (float(r[6]),)
            out.append(item)
        out.sort(key=lambda t: -t[1])
        return out[:topk]


def load(name):
    return RTDetrHF(name) if name in ("heron", "heron_int8", "egret") else PaddleDet(name)


# -------------------------------------------------------------------------------------------
# Unified classes (DocLayNet's 11 minus the List-item / Text split): the evaluation maps every
# native label set onto these. None = dropped.
UNI = ["Caption", "Footnote", "Formula", "Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
MAP_DOCLING = {"Caption": "Caption", "Footnote": "Footnote", "Formula": "Formula", "List-item": "Text", "Page-footer": "Page-footer",
               "Page-header": "Page-header", "Picture": "Picture", "Section-header": "Section-header", "Table": "Table", "Text": "Text",
               "Title": "Title", "Document Index": "Table", "Code": "Text", "Checkbox-Selected": None, "Checkbox-Unselected": None,
               "Form": None, "Key-Value Region": None}
MAP_PP = {"paragraph_title": "Section-header", "doc_title": "Title", "text": "Text", "abstract": "Text", "content": "Text",
          "reference": "Text", "reference_content": "Text", "algorithm": "Text", "aside_text": "Text", "vertical_text": "Text",
          "image": "Picture", "chart": "Picture", "seal": "Picture", "figure_title": "Caption", "table_title": "Caption",
          "chart_title": "Caption", "formula": "Formula", "display_formula": "Formula", "table": "Table", "footnote": "Footnote",
          "vision_footnote": "Footnote", "header": "Page-header", "header_image": "Page-header", "footer": "Page-footer",
          "footer_image": "Page-footer", "number": "@position", "formula_number": None, "inline_formula": None}
# docling-rs JSON labels (lower snake case)
MAP_DLRS = {"caption": "Caption", "footnote": "Footnote", "formula": "Formula", "list_item": "Text", "page_footer": "Page-footer",
            "page_header": "Page-header", "picture": "Picture", "chart": "Picture", "section_header": "Section-header", "table": "Table",
            "text": "Text", "title": "Title", "document_index": "Table", "code": "Text", "checkbox_selected": None,
            "checkbox_unselected": None, "form": None, "key_value_region": None, "paragraph": "Text", "reference": "Text"}


def unify(model, label, y0, y1, H):
    m = MAP_DOCLING if model in ("heron", "heron_int8", "egret") else MAP_DLRS if model == "dlrs" else MAP_PP
    u = m.get(label)
    if u == "@position":
        u = "Page-header" if (y0 + y1) / 2 < H / 2 else "Page-footer"
    return u


# unified -> labels the reading-order code understands
TO_RO = {"Caption": "caption", "Footnote": "footnote", "Formula": "formula", "Page-footer": "page_footer",
         "Page-header": "page_header", "Picture": "picture", "Section-header": "section_header", "Table": "table",
         "Text": "text", "Title": "section_header"}
