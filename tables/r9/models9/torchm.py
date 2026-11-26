"""PyTorch / Paddle candidates run from their official code on CPU: MuTabNet (ICDAR 2026 release), LORE-TSR
(official code, checkpoints linked from lgwsgc/LORE-TSR), TRACE (naver-ai/trace), TableMaster (PaddleOCR 2.7
TableStructurer with the PDF-Extract-Kit weights). Each __call__(im RGB PIL, words) -> (cells, model ms, info)."""
import os, sys, time, tempfile, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from geo9 import html_cells, box4, logic_cells

THREADS = int(os.environ.get("R9_THREADS", "4"))
CODE = os.environ.get("R9_CODE", os.path.join(os.path.expanduser("~"), "r9code"))


def _torch_cpu():
    import torch
    torch.set_num_threads(THREADS)
    torch.cuda.synchronize = lambda *a, **k: None
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    orig = torch.load

    def load(f, *a, **k):
        k["map_location"] = "cpu"; k.setdefault("weights_only", False)
        return orig(f, *a, **k)
    torch.load = load
    for n, t in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if not hasattr(np, n): setattr(np, n, t)
    return torch


class MuTab:
    """MuTabNet at release commit c45a54c: mmdet init_detector + model.predict(path); html tokens, one box and one
    text per cell token"""

    def __init__(self, variant, wdir):
        torch = _torch_cpu()
        repo = os.path.join(CODE, "mutabnet"); sys.path.insert(0, repo); os.chdir(repo)
        import mutab  # noqa: F401  (registers the model)
        from mmcv import Config
        from mmdet.apis import init_detector
        conf = Config.fromfile(os.path.join(repo, "configs", "pubtabnet.py" if variant == "mutab_ptn" else "fintabnet.py"))
        self.weights = [os.path.join(wdir, "model.pth")]
        self.model = init_detector(conf, checkpoint=self.weights[0], device="cpu")
        self.model.eval(); self.torch = torch
        self.tmp = tempfile.mkdtemp()

    def __call__(self, im, words):
        p = os.path.join(self.tmp, "crop.png"); im.convert("RGB").save(p)
        t = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.predict(p)
        ms = (time.perf_counter() - t) * 1000
        html = out.get("html"); cell = out.get("cell") or []; bbox = out.get("bbox")
        toks = html.split(",") if isinstance(html, str) else list(html or [])
        toks = ["<td></td>" if t.startswith("<eb") else t for t in toks]
        boxes = [list(map(float, b)) for b in (bbox if bbox is not None else [])]
        texts = [c if isinstance(c, str) else "".join(c) for c in cell]
        cells = html_cells(toks, boxes, texts)
        return cells, ms, {"n_tokens": len(toks), "n_boxes": len(boxes), "n_texts": len(texts)}


class TableMaster:
    """PaddleOCR 2.7 ppstructure TableStructurer, table_algorithm TableMaster, max_len 480 (PaddleOCR's TableMaster
    inference settings), weights from PDF-Extract-Kit models/TabRec/TableMaster"""

    def __init__(self, variant, wdir):
        import paddleocr  # noqa: F401  (puts ppocr / ppstructure on sys.path)
        pdir = os.path.dirname(paddleocr.__file__)
        if pdir not in sys.path: sys.path.insert(0, pdir)
        from ppstructure.utility import init_args
        from ppstructure.table.predict_structure import TableStructurer
        args = init_args().parse_args([])
        args.table_algorithm = "TableMaster"; args.table_model_dir = os.path.join(wdir, "table_structure_tablemaster_infer")
        args.table_char_dict_path = os.path.join(wdir, "table_master_structure_dict.txt"); args.table_max_len = 480
        args.use_gpu = False; args.cpu_threads = THREADS; args.enable_mkldnn = False; args.benchmark = False; args.use_onnx = False
        self.ts = TableStructurer(args)
        self.weights = [os.path.join(args.table_model_dir, f) for f in ("inference.pdiparams", "inference.pdmodel")]

    def __call__(self, im, words):
        bgr = np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])
        t = time.perf_counter(); res, _ = self.ts(bgr); ms = (time.perf_counter() - t) * 1000
        if res is None: return [], ms, {}
        toks, boxes = res
        toks = [t for t in toks if t not in ("<html>", "<body>", "<table>", "</table>", "</body>", "</html>")]
        boxes = [list(map(float, np.asarray(b).reshape(-1))) for b in boxes]
        return html_cells(toks, boxes), ms, {"n_tokens": len(toks), "n_boxes": len(boxes)}


LORE_ARGS = {
    "lore_wireless": ("ctdet_mid", "table_mid", "resfpnhalf_18", ["--upper_left", "--tsfm_layers", "4", "--stacking_layers", "4", "--wiz_2dpe", "--wiz_detect",
                      "--wiz_stacking", "--vis_thresh_corner", "0.3", "--vis_thresh", "0.2", "--scores_thresh", "0.2", "--nms"]),
    "lore_wtw": ("ctdet", "table", "dla_34", ["--tsfm_layers", "4", "--stacking_layers", "4", "--wiz_4ps", "--wiz_detect", "--wiz_rev", "--wiz_stacking",
                 "--vis_thresh_corner", "0.3", "--vis_thresh", "0.2", "--scores_thresh", "0.2", "--nms"]),
    "lore_ptn": ("ctdet_small", "table_small", "dla_34", ["--tsfm_layers", "3", "--stacking_layers", "3", "--wiz_2dpe", "--wiz_detect", "--wiz_stacking",
                 "--vis_thresh_corner", "0.3", "--vis_thresh", "0.35", "--scores_thresh", "0.35", "--nms"]),
}


class Lore:
    """official LORE-TSR code (AlibabaResearch/AdvancedLiterateMachinery, DocumentUnderstanding/LORE-TSR) with the
    official demo settings per checkpoint; DCNv2 replaced by torchvision's deform_conv2d (same weights); the
    detector's show_results hook is replaced to capture the boxes and logical coordinates it would write"""

    def __init__(self, variant, wdir):
        _torch_cpu()
        src = os.path.join(CODE, "lore", "src"); sys.path.insert(0, os.path.join(src, "lib")); sys.path.insert(0, src); os.chdir(src)
        from opts import opts
        from detectors.detector_factory import detector_factory
        task, ds, arch, extra = LORE_ARGS[variant]
        ck = self._find(wdir)
        self.tmp = tempfile.mkdtemp()
        argv = [task, "--dataset", ds, "--arch", arch, "--K", "3000", "--MK", "5000", "--gpus", "-1", "--debug", "1", "--convert_onnx", "0",
                "--demo", self.tmp, "--demo_name", "r9", "--demo_dir", os.path.join(self.tmp, "vis"), "--output_dir", self.tmp + "/",
                "--load_model", ck["model"], "--load_processor", ck["processor"]] + extra
        opt = opts().init(argv)
        self.det = detector_factory[opt.task](opt); self.opt = opt
        self.weights = [ck["model"], ck["processor"]]
        cap = {}

        def show_results(debugger, image, results, corner, image_name, logi=None):
            cap["results"] = results; cap["logi"] = logi
        self.det.show_results = show_results; self.cap = cap

    @staticmethod
    def _find(wdir):
        out = {}
        for root, _, files in os.walk(wdir):
            for f in files:
                if f.startswith("model") and f.endswith(".pth"): out.setdefault("model", os.path.join(root, f))
                if f.startswith("processor") and f.endswith(".pth"): out.setdefault("processor", os.path.join(root, f))
        if len(out) != 2: raise RuntimeError(f"LORE checkpoint files not found in {wdir}: {out}")
        return out

    def __call__(self, im, words):
        import cv2
        p = os.path.join(self.tmp, "crop.png"); cv2.imwrite(p, np.asarray(im.convert("RGB"))[:, :, ::-1])
        self.cap.clear()
        t = time.perf_counter(); self.det.run(self.opt, p); ms = (time.perf_counter() - t) * 1000
        res, logi = self.cap.get("results"), self.cap.get("logi")
        if res is None or logi is None: return [], ms, {}
        logi = np.asarray(logi.cpu() if hasattr(logi, "cpu") else logi)
        if logi.ndim == 1: logi = logi[None]
        boxes, lps = [], []
        for m, b in enumerate(res[1]):
            if b[8] > self.opt.vis_thresh and m < len(logi):
                boxes.append(list(map(float, b[:8]))); lps.append([int(v) for v in logi[m][:4]])
        return logic_cells(boxes, lps), ms, {"n": len(boxes)}


class Trace:
    """TRACE (naver-ai/trace) test_net with configs/trace_base.json (canvas 1600, mag 10, thresholds 0.3 / 0.2); the
    detected table overlapping the crop most is the answer"""

    def __init__(self, variant, wdir):
        torch = _torch_cpu()
        repo = os.path.join(CODE, "trace"); sys.path.insert(0, repo); os.chdir(repo)
        import imgproc, postprocessor
        from model import TraceModel
        from collections import OrderedDict
        net = TraceModel()
        sd = torch.load(os.path.join(wdir, "model.pth"))
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
        sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
        net.load_state_dict(sd); net.eval()
        self.net, self.torch, self.imgproc, self.post = net, torch, imgproc, postprocessor
        self.args = types.SimpleNamespace(canvas_size=1600, mag_ratio=10, threshold1=0.3, threshold2=0.2)
        self.weights = [os.path.join(wdir, "model.pth")]

    def __call__(self, im, words):
        img = np.asarray(im.convert("RGB"))
        t = time.perf_counter()
        r, ratio, _ = self.imgproc.resize_aspect_ratio(img, self.args.canvas_size, mag_ratio=self.args.mag_ratio)
        x = self.torch.from_numpy(self.imgproc.normalizeMeanVariance(r)).permute(2, 0, 1).unsqueeze(0)
        with self.torch.no_grad():
            y = self.net(x)
            if isinstance(y, tuple): y = y[0]
        res = self.post.run(y[0].cpu().numpy(), self.args, ratio)
        ms = (time.perf_counter() - t) * 1000
        if not res: return [], ms, {"n_tables": 0}
        W, H = im.size

        def ov(tb):
            x0, y0, w, h = tb["rect"]
            return max(0, min(W, x0 + w) - max(0, x0)) * max(0, min(H, y0 + h) - max(0, y0))
        best = max(res, key=ov)
        boxes = [list(map(float, c["quad"])) for c in best["cells"]]
        lps = [[c["row_range"][0], c["row_range"][1], c["col_range"][0], c["col_range"][1]] for c in best["cells"]]
        return logic_cells(boxes, lps), ms, {"n_tables": len(res)}
