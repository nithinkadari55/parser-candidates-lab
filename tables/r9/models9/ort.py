"""ONNX Runtime candidates: TATR (reference), RapidTable SLANet en / ch / plus, PaddlePaddle SLANet / SLANet_plus
official ONNX, AE3-TL. Each __call__(im RGB PIL, words [{text, bbox}]) -> (cells, model ms, info)."""
import os, sys, time, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geo9 import logic_cells, box4

THREADS = int(os.environ.get("R9_THREADS", "4"))


class Tatr:
    """the harness's TATR v1.1 fp16 with the round-3/5 pre/post-processing (bench_models.TATR16 + tatr8 padding)"""
    weights = []

    def __init__(self, variant, wdir):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
        import bench_models as BM
        self.BM = BM; self.M = BM.TATR16(THREADS)
        self.weights = [os.path.join(BM.MODELS, "tatr", BM.TATR16.FILE)]

    def __call__(self, im, words):
        BM = self.BM; PAD = 20
        im2, (ox, oy) = BM.prep_image(im, "pad", PAD)
        toks = BM.chunks(dict(words=[dict(text=w["text"], bbox=list(w["bbox"])) for w in words]), PAD, ox, oy)
        cells, tm = self.M(im2, toks)
        for c in cells: c["bbox"] = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
        return cells, tm * 1000, {}


def ocr_tuple(words):
    """our words -> RapidTable's ocr_results tuple (boxes as 4 points, texts, scores)"""
    if not words: return None
    boxes = np.array([[[w["bbox"][0], w["bbox"][1]], [w["bbox"][2], w["bbox"][1]], [w["bbox"][2], w["bbox"][3]], [w["bbox"][0], w["bbox"][3]]] for w in words], np.float32)
    return (boxes, tuple(w["text"] for w in words), tuple(1.0 for _ in words))


class Rapid:
    """RapidTable 3.0.2 (rapid_table.RapidTable, use_ocr=False): structure tokens -> logic points + cell boxes"""
    FILES = {"rapid_en": ("ppstructure_en", "en_ppstructure_mobile_v2_SLANet.onnx"), "rapid_zh": ("ppstructure_zh", "ch_ppstructure_mobile_v2_SLANet.onnx"),
             "rapid_plus": ("slanet_plus", "slanet-plus.onnx"),
             "paddle_slanet": ("ppstructure_en", "../paddle_onnx/SLANet/inference.onnx"), "paddle_slanet_plus": ("slanet_plus", "../paddle_onnx/SLANet_plus/inference.onnx")}

    def __init__(self, variant, wdir):
        from rapid_table import RapidTable, RapidTableInput, ModelType
        mt, fn = self.FILES[variant]
        path = os.path.normpath(os.path.join(wdir, fn))
        self.weights = [path]
        if variant.startswith("paddle_"):
            self._patch_chars(os.path.join(os.path.dirname(path), "inference.yml"))
        cfg = RapidTableInput(model_type=ModelType(mt), model_dir_or_path=path, use_ocr=False,
                              engine_cfg={"intra_op_num_threads": THREADS, "inter_op_num_threads": 1})
        self.eng = RapidTable(cfg)

    @staticmethod
    def _patch_chars(yml):
        """the official PaddleX ONNX files carry the character list in inference.yml, not in the ONNX metadata"""
        import yaml
        from rapid_table.inference_engine.onnxruntime import main as M
        chars = yaml.safe_load(open(yml, encoding="utf-8"))["PostProcess"]["character_dict"]
        cls = M.OrtInferSession
        orig = cls.get_character_list

        def get_character_list(self, key="character"):
            try:
                v = orig(self, key)
                if v: return v
            except Exception:
                pass
            return list(chars)
        cls.get_character_list = get_character_list

    def __call__(self, im, words):
        a = np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])
        t = time.perf_counter(); res = self.eng(a); ms = (time.perf_counter() - t) * 1000
        if not res.cell_bboxes or res.cell_bboxes[0] is None or not len(res.logic_points): return [], ms, {}
        boxes, lp = res.cell_bboxes[0], res.logic_points[0]
        return logic_cells(list(boxes), list(lp)), ms, {"n_boxes": len(boxes), "n_logic": len(lp)}


class Ae3:
    """AE3-TL (HF APEXE3HQ/ae3-tl, pinned): the library's EngineOrt on the crop as a page, with our words; the
    detected table overlapping the crop most is the answer (the crop is docling's table box + 4 pt)"""

    def __init__(self, variant, wdir):
        repo = os.path.join(wdir, "repo")
        sys.path.insert(0, os.path.join(repo, "src"))
        os.environ["AE3_WEIGHTS_DIR"] = os.path.join(repo, "weights"); os.environ["AE3_THREADS"] = str(THREADS)
        os.environ.setdefault("AE3_ONNX_PROVIDER", "cpu")
        from ae3_tl.serve.settings import load_settings
        from ae3_tl.backends import create_engine
        from ae3_tl.image import words as WM
        self.WM = WM
        self.eng = create_engine(load_settings())
        self.weights = [os.path.join(repo, "weights", f) for f in os.listdir(os.path.join(repo, "weights")) if f.endswith(".onnx")]

    def __call__(self, im, words):
        """AE3-TL reads whole pages: the crop is pasted, at its own scale, at the top left of a white page of Letter
        proportions with a 6% margin (same pixels, page context the model was trained on); boxes are moved back"""
        import json
        from PIL import Image
        W, H = im.size
        PW = int(max(W / 0.88, H * 0.7727 / 0.88)); PH = int(PW / 0.7727); ox, oy = int(0.06 * PW), int(0.06 * PH)
        page = Image.new("RGB", (PW, PH), "white"); page.paste(im.convert("RGB"), (ox, oy))
        ws = self.WM.parse(json.dumps([dict(text=w["text"], bbox=[w["bbox"][0] + ox, w["bbox"][1] + oy, w["bbox"][2] + ox, w["bbox"][3] + oy]) for w in words]).encode()) if words else []
        t = time.perf_counter(); pred = self.eng.predict(page, ws, page.size); ms = (time.perf_counter() - t) * 1000
        tabs = list(getattr(pred, "tables", []) or [])
        if not tabs: return [], ms, {"n_tables": 0}

        def ov(t):
            b = t["bbox"]; ix = max(0, min(ox + W, b[2]) - max(ox, b[0])); iy = max(0, min(oy + H, b[3]) - max(oy, b[1]))
            return ix * iy
        best = max(tabs, key=ov)
        cells = []
        for c in best["cells"]:
            rs, cs = c["row_nums"], c["column_nums"]
            if not rs or not cs: continue
            b = box4(c["bbox"])
            cells.append(dict(r0=min(rs), r1=max(rs) + 1, c0=min(cs), c1=max(cs) + 1, bbox=[b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy],
                              header=bool(c.get("header")), text=c.get("cell_text", "")))
        return cells, ms, {"n_tables": len(tabs)}
