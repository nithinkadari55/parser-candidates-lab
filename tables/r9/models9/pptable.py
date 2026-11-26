"""PaddleOCR 3.x table recognition v2 pipeline (PaddleX) with SLANet_plus as the wired and wireless structure model,
RT-DETR-L wired / wireless table cell detectors and PP-LCNet_x1_0_table_cls (273 MB; the default SLANeXt structure
models are 365 MB each, over the size rule). No layout, no document preprocessing, no table orientation model; our
words are the pipeline's OCR result (use_ocr_model=False). Output: the pipeline's html (structure + its cell text from
our words) and its cell boxes when they line up one-to-one with the html cells."""
import os, sys, time, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geo9 import html_cells


def html_tokens(h):
    """an html table string -> (structure tokens, cell texts)"""
    import lxml.html
    root = lxml.html.fromstring(h)
    tab = root if root.tag == "table" else root.find(".//table")
    toks, texts = [], []
    if tab is None: return toks, texts
    for tr in tab.iter("tr"):
        toks.append("<tr>")
        for td in tr:
            if td.tag not in ("td", "th"): continue
            cs = int(re.sub(r"\D", "", td.get("colspan") or "1") or 1); rs = int(re.sub(r"\D", "", td.get("rowspan") or "1") or 1)
            if cs > 1 or rs > 1:
                toks += ["<td", (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else ""), ">", "</td>"]
            else:
                toks += ["<td>", "</td>"]
            texts.append(re.sub(r"\s+", " ", td.text_content() or "").strip())
        toks.append("</tr>")
    return toks, texts


class OcrRes(dict):
    pass


class PPTable:
    def __init__(self, variant, wdir):
        from paddleocr import TableRecognitionPipelineV2
        d = lambda m: os.path.join(wdir, m)
        self.pipe = TableRecognitionPipelineV2(
            table_classification_model_name="PP-LCNet_x1_0_table_cls", table_classification_model_dir=d("PP-LCNet_x1_0_table_cls"),
            wired_table_structure_recognition_model_name="SLANet_plus", wired_table_structure_recognition_model_dir=d("SLANet_plus"),
            wireless_table_structure_recognition_model_name="SLANet_plus", wireless_table_structure_recognition_model_dir=d("SLANet_plus"),
            wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det", wired_table_cells_detection_model_dir=d("RT-DETR-L_wired_table_cell_det"),
            wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det", wireless_table_cells_detection_model_dir=d("RT-DETR-L_wireless_table_cell_det"),
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=False, use_ocr_model=False,
            device="cpu", cpu_threads=int(os.environ.get("R9_THREADS", "4")), enable_mkldnn=False)
        self.weights = [os.path.join(d(m), "inference.pdiparams") for m in ("PP-LCNet_x1_0_table_cls", "SLANet_plus", "RT-DETR-L_wired_table_cell_det",
                                                                              "RT-DETR-L_wireless_table_cell_det")]

    def __call__(self, im, words):
        bgr = np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])
        boxes = np.array([[w["bbox"][0], w["bbox"][1], w["bbox"][2], w["bbox"][3]] for w in words], dtype=np.float32).reshape(-1, 4)
        polys = [np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32) for b in boxes]
        ocr = OcrRes(rec_boxes=boxes.astype(np.int32) if len(boxes) else np.zeros((0, 4), np.int32), rec_texts=[w["text"] for w in words],
                     rec_scores=[1.0] * len(words), rec_polys=polys, dt_polys=polys, textline_orientation_angles=[-1] * len(words))
        t = time.perf_counter()
        out = list(self.pipe.predict(bgr, use_ocr_model=False, overall_ocr_res=ocr, use_table_orientation_classify=False,
                                     use_ocr_results_with_table_cells=False))
        ms = (time.perf_counter() - t) * 1000
        res = out[0] if out else None
        tl = (res.get("table_res_list") if res is not None else None) or []
        if not tl: return [], ms, {"n_tables": 0}
        t0 = tl[0]
        toks, texts = html_tokens(t0.get("pred_html") or "")
        cb = t0.get("cell_box_list")
        cb = [list(map(float, b)) for b in (cb.tolist() if hasattr(cb, "tolist") else (cb or []))]
        aligned = len(cb) == len(texts) and len(cb) > 0
        cells = html_cells(toks, cb if aligned else None, texts)
        return cells, ms, {"aligned_boxes": aligned, "n_cells_html": len(texts), "n_boxes": len(cb), "own_text_only": not aligned}
