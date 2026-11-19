import sys, os, json, time, copy, argparse, traceback
import numpy as np
from PIL import Image
import paths
from paths import SETS, MODELS, OUT
import onnxruntime as ort
ort.set_default_logger_severity(3)

ALL_SETS = ["ptn", "ftn", "pt1m", "icdar", "pt2", "scitsr", "tabrec", "rot"]


def so(threads):
    o = ort.SessionOptions()
    if threads: o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    return o


def set_ids(setname):
    ids = json.load(open(os.path.join(SETS, setname, "index.json")))
    lim = int(os.environ.get("TB_LIMIT") or 0)
    return ids[:lim] if lim else ids


def shard_ids(setname, shard, nshards):
    return [t for i, t in enumerate(set_ids(setname)) if i % nshards == shard]


def load_gt(setname, tid):
    return json.load(open(os.path.join(SETS, setname, tid + ".json"), encoding="utf-8"))


def chunks(gt, pad, ox=None, oy=None):
    ox = -pad if ox is None else ox; oy = -pad if oy is None else oy
    if "words" in gt:
        src = [dict(text=w["text"], bbox=w["bbox"], gi=w.get("gi")) for w in gt["words"]]
    else:
        src = [dict(text=c["text"], bbox=c["bbox"][:4], gi=i) for i, c in enumerate(gt["cells"]) if c["text"].strip()]
    toks = []
    for t in src:
        x0, y0, x1, y1 = t["bbox"]
        toks.append(dict(text=t["text"], bbox=[x0 - ox, y0 - oy, x1 - ox, y1 - oy], gi=t["gi"]))
    toks.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for i, t in enumerate(toks): t["span_num"] = i; t["line_num"] = 0; t["block_num"] = 0
    return toks


def prep_image(im, mode, pad):
    if mode == "trim":
        g = np.asarray(im.convert("L")) < 245
        ys, xs = np.where(g)
        if len(xs):
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            return padimg(im.crop((x0, y0, x1, y1)), pad), (x0 - pad, y0 - pad)
    return padimg(im, pad), (-pad, -pad)


def padimg(im, pad):
    if pad <= 0: return im
    W, H = im.size
    n = Image.new("RGB", (W + 2 * pad, H + 2 * pad), (255, 255, 255)); n.paste(im, (pad, pad)); return n


class TATR:
    FILE = "model_quantized.onnx"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, threads):
        import tatr_lib as TL
        self.TL = TL; self.keep_text = False
        self.sess = ort.InferenceSession(os.path.join(MODELS, "tatr", self.FILE), so(threads), providers=["CPUExecutionProvider"])
        self.f16 = self.sess.get_inputs()[0].type == "tensor(float16)"

    def prep(self, im):
        w, h = im.size; s = 1000 / max(w, h)
        r = im.resize((int(round(s * w)), int(round(s * h))))
        return ((np.asarray(r).astype(np.float32) / 255 - self.MEAN) / self.STD).transpose(2, 0, 1)[None].copy()

    def __call__(self, im, toks):
        x = self.prep(im)
        t = time.perf_counter()
        lg, bx = self.sess.run(None, {"pixel_values": x.astype(np.float16) if self.f16 else x})
        lg, bx = lg.astype(np.float32), bx.astype(np.float32)
        tm = time.perf_counter() - t
        objs = self.TL.outputs_to_objects(lg[0], bx[0], im.size)
        cells, _ = self.TL.run_post(objs, copy.deepcopy(toks))
        out = []
        for c in cells:
            rs, cs = c["row_nums"], c["column_nums"]
            if not rs or not cs: continue
            out.append(dict(r0=min(rs), r1=max(rs) + 1, c0=min(cs), c1=max(cs) + 1, bbox=list(map(float, c["bbox"])), header=bool(c.get("column header"))))
            if self.keep_text: out[-1]["text"] = c.get("cell text", "")
        return out, tm


class SLANet:
    def __init__(self, threads):
        import yaml
        from rapid_table.table_structure.pp_structure.pre_process import TablePreprocess
        from rapid_table.table_structure.pp_structure.post_process import TableLabelDecode
        from rapid_table.table_matcher.main import TableMatch
        from rapid_table.utils.typings import ModelType
        self.sess = ort.InferenceSession(os.path.join(MODELS, "slanet1m", "slanet_1m.onnx"), so(threads), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        chars = yaml.safe_load(open(os.path.join(MODELS, "slanet1m", "inference.yml")))["PostProcess"]["character_dict"]
        self.pre = TablePreprocess(max_len=488)
        self.post = TableLabelDecode(list(chars), {"model_type": ModelType.PPSTRUCTURE_EN})
        self.tm = TableMatch()

    def __call__(self, im, toks):
        a = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        imgs, shapes = self.pre([a])
        x = np.stack(imgs).astype(np.float32)
        t = time.perf_counter()
        bbox_preds, struct_probs = self.sess.run(None, {self.iname: x})
        tm = time.perf_counter() - t
        structs, boxes = self.post(bbox_preds, struct_probs, shapes, [a])
        toks_ = structs[0][0]
        lp = self.tm.decode_logic_points([structs[0]])[0] if len(boxes[0]) else []
        hdr = 0; inh = False
        for tk in toks_:
            if tk == "<thead>": inh = True
            elif tk == "</thead>": inh = False
            elif tk == "</tr>" and inh: hdr += 1
        out = []
        for b, (r0, r1, c0, c1) in zip(boxes[0], lp):
            b = np.asarray(b, float)
            xs, ys = b[0::2], b[1::2]
            out.append(dict(r0=int(r0), r1=int(r1) + 1, c0=int(c0), c1=int(c1) + 1, bbox=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], header=int(r0) < hdr))
        return out, tm


class TCN:
    PAD = 0

    def __init__(self, threads):
        from onnxtr.models import table_predictor
        from onnxtr.models.table_structure import tablecenternet
        from onnxtr.models.engine import EngineConfig
        o = so(threads)
        self.pred = table_predictor(arch=tablecenternet(model_path=os.path.join(MODELS, "tcn", "model.onnx"), engine_cfg=EngineConfig(providers=["CPUExecutionProvider"], session_options=o)), assume_straight_pages=True)

    def __call__(self, im, toks):
        W, H = im.size
        a = np.asarray(im)
        wb = np.array([[t["bbox"][0] / W, t["bbox"][1] / H, t["bbox"][2] / W, t["bbox"][3] / H] for t in toks], dtype=np.float32).reshape(-1, 4)
        t = time.perf_counter()
        res = self.pred.predict_regions([a], [{"class_names": ["Table"], "boxes": [np.array([0, 0, 1, 1], np.float32)]}], word_boxes=[wb])
        tm = time.perf_counter() - t
        tables = res[0]
        if not tables: return [], tm
        tb = max(tables, key=lambda x: len(x["cells"]))
        out = []
        for c in tb["cells"]:
            g = np.asarray(c["geometry"], float).reshape(-1, 2)
            out.append(dict(r0=int(c["row_start"]), r1=int(c["row_end"]) + 1, c0=int(c["col_start"]), c1=int(c["col_end"]) + 1,
                            bbox=[float(g[:, 0].min() * W), float(g[:, 1].min() * H), float(g[:, 0].max() * W), float(g[:, 1].max() * H)], header=False))
        return out, tm


class POTATR:
    PAD = 20
    FILE = "potatr.onnx"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, threads):
        import json as J
        from potatr import postprocess as PP
        import torch
        torch.set_num_threads(max(1, threads or 2))
        self.PP = PP; self.torch = torch
        self.classes = J.load(open(os.path.join(MODELS, "potatr", "potatr_v1.json")))["classes"]
        self.sess = ort.InferenceSession(os.path.join(MODELS, "potatr", self.FILE), so(threads), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def __call__(self, im, toks):
        w, h = im.size; s = 1000 / max(w, h)
        r = im.resize((int(round(s * w)), int(round(s * h))), Image.BILINEAR)
        x = ((np.asarray(r).astype(np.float32) / 255 - self.MEAN) / self.STD).transpose(2, 0, 1)[None].copy()
        t = time.perf_counter()
        lg, bx, ed = self.sess.run(None, {self.iname: x})
        tm = time.perf_counter() - t
        tables = self.PP.potatr_outputs_to_tables_cells(self.torch.from_numpy(lg[0]), self.torch.from_numpy(bx[0]), copy.deepcopy(toks), (w, h), self.classes)
        if not tables: return [], tm
        cells = max(tables, key=len)
        out = []
        for c in cells:
            rs, cs = c.get("row_nums") or [], c.get("column_nums") or []
            if not rs or not cs: continue
            out.append(dict(r0=min(rs), r1=max(rs) + 1, c0=min(cs), c1=max(cs) + 1, bbox=list(map(float, c["bbox"])), header=bool(c.get("header") or c.get("column header"))))
        return out, tm


class POTATR8(POTATR):
    FILE = "potatr_int8.onnx"


class TFV2:
    PAD = 0
    CELL = {"fcel", "ecel", "ched", "rhed", "srow"}

    def __init__(self, threads):
        import torch, torchvision.transforms as T
        from transformers import AutoTokenizer
        from docling_ibm_models.tableformer_v2 import TableFormerV2
        torch.set_num_threads(max(1, threads or 4))
        d = os.path.join(MODELS, "tfv2")
        self.torch = torch
        self.model = TableFormerV2.from_pretrained(d).eval()
        self.tok = AutoTokenizer.from_pretrained(d)
        self.tf = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.names = {i: self.tok.decode([i]).strip().strip("<>") for i in range(self.model.config.vocab_size)}

    def generate(self, x):
        with self.torch.no_grad():
            o = self.model.generate(images=x, tokenizer=self.tok, max_length=512)
        ids = o["generated_ids"][0].tolist()
        bb = o.get("predicted_bboxes")
        bb = bb[0].numpy() if bb is not None and bb.numel() else np.zeros((0, 4))
        return ids, bb

    def __call__(self, im, toks):
        W, H = im.size
        x = self.tf(im).unsqueeze(0)
        t = time.perf_counter()
        ids, bb = self.generate(x)
        tm = time.perf_counter() - t
        return otsl_cells([self.names[i] for i in ids], bb, W, H, self.CELL), tm


def otsl_cells(tags, bb, W, H, cell_tags):
    tags = [t for t in tags if t not in ("pad", "[UNK]", "start", "end")]
    rows, cur = [], []
    for t in tags:
        if t == "nl": rows.append(cur); cur = []
        else: cur.append(t)
    if cur: rows.append(cur)
    if not rows: return []
    C = max(len(r) for r in rows)
    grid = [r + [""] * (C - len(r)) for r in rows]
    out = []; k = 0
    for ri, row in enumerate(grid):
        for ci, tg in enumerate(row):
            if tg not in cell_tags: continue
            box = None
            if k < len(bb):
                b = bb[k]; box = [float(b[0] * W), float(b[1] * H), float(b[2] * W), float(b[3] * H)]
            k += 1
            cs = 1
            while ci + cs < C and grid[ri][ci + cs] == "lcel": cs += 1
            rs = 1
            while ri + rs < len(grid) and grid[ri + rs][ci] == "ucel": rs += 1
            out.append(dict(r0=ri, r1=ri + rs, c0=ci, c1=ci + cs, bbox=box or [0, 0, 0, 0], header=tg == "ched"))
    return out


class TATR32(TATR):
    FILE = "model.onnx"


class TATR16(TATR):
    FILE = "model_fp16.onnx"


class TATR4(TATR):
    FILE = "model_int4.onnx"


def _tfv2q(threads):
    from tfv2_onnx import TFV2Q
    return TFV2Q(threads)


MODELS_ = {"tatr": TATR, "tatr32": TATR32, "tatr16": TATR16, "tatr4": TATR4, "slanet1m": SLANet, "tcn": TCN, "potatr": POTATR, "potatr8": POTATR8, "tfv2": TFV2, "tfv2q": _tfv2q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pad", type=int, default=-1); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    base, _, mode = a.model.partition("-")
    M = MODELS_[base](a.threads)
    if a.pad < 0: a.pad = 4 if mode == "trim" else getattr(M, "PAD", 20)
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    for s in a.sets.split(","):
        of = os.path.join(OUT, "pred", f"{a.model}_{s}_s{a.shard}.jsonl")
        n = 0
        with open(of, "w", encoding="utf-8") as f:
            for tid in shard_ids(s, a.shard, a.nshards):
                gt = load_gt(s, tid)
                im, (ox, oy) = prep_image(Image.open(os.path.join(SETS, s, tid + ".png")).convert("RGB"), mode or "pad", a.pad)
                toks = chunks(gt, a.pad, ox, oy)
                rec = dict(id=tid)
                try:
                    t = time.perf_counter()
                    cells, tm = M(im, toks)
                    rec["ms_total"] = (time.perf_counter() - t) * 1000; rec["ms_model"] = tm * 1000
                    for c in cells: c["bbox"] = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
                    rec["cells"] = cells
                except Exception as e:
                    rec["cells"] = []; rec["err"] = repr(e)[:300]
                    traceback.print_exc()
                f.write(json.dumps(rec) + "\n"); n += 1
        print("done", a.model, s, a.shard, n, flush=True)


if __name__ == "__main__":
    main()
