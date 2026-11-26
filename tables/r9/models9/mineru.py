"""MinerU 4.2.3 table stack with the MinerU-4 ONNX bundle (22.9 MB), as mineru/backend/analysis/pdf/tables.py runs it:
PP-LCNet table classifier -> SLANet-plus (wireless, always) -> if wired, or wireless with score < 0.9, the UNet
wired model and MinerU's own wired/wireless choice. Our words are MinerU's OCR result. The chosen model's cells
(boxes + logic points) are returned."""
import os, sys, time, html
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geo9 import logic_cells


class MinerU:
    def __init__(self, variant, wdir):
        from mineru.model.table.cls.paddle_table_cls import PaddleTableClsModel
        from mineru.model.table.rec.slanet_plus.main import PaddleTableModel
        from mineru.model.table.rec.unet_table.main import UnetTableModel
        from mineru.model.runtime.contracts import AtomicModelName
        self.AM = AtomicModelName
        p = lambda f: os.path.join(wdir, f)
        self.cls = PaddleTableClsModel(model_path=p("PP-LCNet_x1_0_table_cls.onnx"), small_backend="onnx")
        self.wireless = PaddleTableModel(None, model_path=p("slanet-plus.onnx"), small_backend="onnx")
        self.wired = UnetTableModel(None, model_path=p("unet.onnx"), small_backend="onnx")
        self.weights = [p("PP-LCNet_x1_0_table_cls.onnx"), p("slanet-plus.onnx"), p("unet.onnx")]

    def __call__(self, im, words):
        rgb = np.asarray(im.convert("RGB"))
        ocr = [[[[w["bbox"][0], w["bbox"][1]], [w["bbox"][2], w["bbox"][1]], [w["bbox"][2], w["bbox"][3]], [w["bbox"][0], w["bbox"][3]]],
                html.escape(w["text"]), 1.0] for w in words]
        task = {"table_img": rgb, "wired_table_img": rgb, "ocr_result": ocr, "table_res": {}}
        t = time.perf_counter()
        self.cls.batch_predict([task])
        self.wireless.batch_predict([task])
        lab = task["table_res"].get("cls_label"); score = float(task["table_res"].get("cls_score") or 0.0)
        use_wired = lab == self.AM.WiredTable or (lab == self.AM.WirelessTable and score < 0.9)
        chosen = "wireless"; meta = None
        if use_wired:
            meta = self.wired.predict(rgb, ocr, task["table_res"].get("html", "") or "", return_metadata=True)
            if isinstance(meta, dict): chosen = meta.get("selected_model", "wireless")
        ms = (time.perf_counter() - t) * 1000
        info = {"cls": str(lab), "cls_score": score, "chosen": chosen}
        if chosen == "wired" and meta and meta.get("wired_cell_bboxes") is not None:
            b, lp = meta["wired_cell_bboxes"], meta["wired_logic_points"]
        else:
            b, lp = task.get("wireless_cell_bboxes"), task.get("wireless_logic_points")
        if b is None or lp is None: return [], ms, info
        b, lp = list(b), list(lp)
        info["n_boxes"], info["n_logic"] = len(b), len(lp)
        return logic_cells(b, lp), ms, info
