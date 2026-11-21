"""docling's picture classifier (DocumentFigureClassifier-v2.5, picture_classifier.onnx from docling.rs models-v1)
on the same test crops, preprocessed as docling.rs enrich.rs does (224x224 bilinear, RGB/255, its mean/std),
26 classes mapped to the 9 used here. usage: pics_docling.py PICSDIR OUT.json"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pics_data import CLASSES  # noqa: E402

URL = "https://github.com/docling-project/docling.rs/releases/download/models-v1/picture_classifier.onnx"
D26 = ["logo", "photograph", "icon", "engineering_drawing", "line_chart", "bar_chart", "other", "table", "flow_chart",
       "screenshot_from_computer", "signature", "screenshot_from_manual", "geographical_map", "pie_chart",
       "page_thumbnail", "stamp", "music", "calendar", "qr_code", "bar_code", "full_page_image", "scatter_plot",
       "chemistry_structure", "topographical_map", "crossword_puzzle", "box_plot"]
MAP = {"logo": "logo", "photograph": "photo", "engineering_drawing": "diagram", "line_chart": "chart",
       "bar_chart": "chart", "table": "table_image", "flow_chart": "diagram", "screenshot_from_computer": "screenshot",
       "signature": "signature", "screenshot_from_manual": "screenshot", "pie_chart": "chart", "stamp": "stamp",
       "scatter_plot": "chart", "chemistry_structure": "diagram", "box_plot": "chart"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.47853944, 0.4732864, 0.47434163], dtype=np.float32)


def main():
    pics, out = sys.argv[1], sys.argv[2]
    import onnxruntime as ort
    from PIL import Image
    from sklearn.metrics import classification_report, f1_score
    p = os.path.join(os.path.dirname(os.path.abspath(out)), "picture_classifier.onnx")
    if not os.path.exists(p):
        urllib.request.urlretrieve(URL, p)
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    s = ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    man = json.load(open(os.path.join(pics, "manifest.json")))["items"]

    def run(m):
        im = Image.open(m["path"]).convert("RGB").resize((224, 224), Image.BILINEAR)
        x = ((np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        t = time.perf_counter()
        lg = s.run(None, {name: np.ascontiguousarray(x)})[0][0]
        return int(lg.argmax()), (time.perf_counter() - t) * 1000
    te = [m for m in man if m["split"] == "test"]
    dom = [m for m in man if m["split"] == "domain"]
    preds, ms, raw = [], [], []
    for m in te:
        k, t = run(m)
        raw.append(D26[k])
        preds.append(CLASSES.index(MAP.get(D26[k], "other")))
        ms.append(t)
    dpred = [CLASSES.index(MAP.get(D26[run(m)[0]], "other")) for m in dom]
    lab = [CLASSES.index(m["label"]) for m in te]
    ms.sort()
    res = {"kind": "docling_v2.5", "mb": os.path.getsize(p) / 1e6, "macro_f1": float(f1_score(lab, preds, average="macro")),
           "report": classification_report(lab, preds, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                           output_dict=True, zero_division=0),
           "ms_p50": ms[len(ms) // 2], "ms_p95": ms[int(0.95 * (len(ms) - 1))], "pred": preds,
           "test_ids": [m["id"] for m in te], "labels": lab, "domain_pred": dpred,
           "raw26": {c: raw.count(c) for c in set(raw)}}
    json.dump(res, open(out, "w"))
    print("docling macro-F1", res["macro_f1"])


if __name__ == "__main__":
    main()
