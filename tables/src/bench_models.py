import sys, os, json, time, copy, argparse, traceback
import numpy as np
from PIL import Image
import paths
from paths import SETS, MODELS, OUT
import onnxruntime as ort
ort.set_default_logger_severity(3)

ALL_SETS = ["ptn", "ftn", "pt1m", "icdar"]


def so(threads):
    o = ort.SessionOptions()
    if threads: o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    return o


def set_ids(setname):
    return json.load(open(os.path.join(SETS, setname, "index.json")))


def shard_ids(setname, shard, nshards):
    return [t for i, t in enumerate(set_ids(setname)) if i % nshards == shard]


def load_gt(setname, tid):
    return json.load(open(os.path.join(SETS, setname, tid + ".json"), encoding="utf-8"))


def chunks(gt, pad):
    if "words" in gt:
        src = [dict(text=w["text"], bbox=w["bbox"], gi=w.get("gi")) for w in gt["words"]]
    else:
        src = [dict(text=c["text"], bbox=c["bbox"][:4], gi=i) for i, c in enumerate(gt["cells"]) if c["text"].strip()]
    toks = []
    for t in src:
        x0, y0, x1, y1 = t["bbox"]
        toks.append(dict(text=t["text"], bbox=[x0 + pad, y0 + pad, x1 + pad, y1 + pad], gi=t["gi"]))
    toks.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for i, t in enumerate(toks): t["span_num"] = i; t["line_num"] = 0; t["block_num"] = 0
    return toks


def padimg(im, pad):
    if pad <= 0: return im
    W, H = im.size
    n = Image.new("RGB", (W + 2 * pad, H + 2 * pad), (255, 255, 255)); n.paste(im, (pad, pad)); return n


class TATR:
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, threads):
        import tatr_lib as TL
        self.TL = TL; self.keep_text = False
        self.sess = ort.InferenceSession(os.path.join(MODELS, "tatr", "model_quantized.onnx"), so(threads), providers=["CPUExecutionProvider"])

    def prep(self, im):
        w, h = im.size; s = 1000 / max(w, h)
        r = im.resize((int(round(s * w)), int(round(s * h))))
        return ((np.asarray(r).astype(np.float32) / 255 - self.MEAN) / self.STD).transpose(2, 0, 1)[None].copy()

    def __call__(self, im, toks):
        x = self.prep(im)
        t = time.perf_counter()
        lg, bx = self.sess.run(None, {"pixel_values": x})
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


MODELS_ = {"tatr": TATR, "slanet1m": SLANet}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pad", type=int, default=20); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    M = MODELS_[a.model](a.threads)
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    for s in a.sets.split(","):
        of = os.path.join(OUT, "pred", f"{a.model}_{s}_s{a.shard}.jsonl")
        n = 0
        with open(of, "w", encoding="utf-8") as f:
            for tid in shard_ids(s, a.shard, a.nshards):
                gt = load_gt(s, tid)
                im = padimg(Image.open(os.path.join(SETS, s, tid + ".png")).convert("RGB"), a.pad)
                toks = chunks(gt, a.pad)
                rec = dict(id=tid)
                try:
                    t = time.perf_counter()
                    cells, tm = M(im, toks)
                    rec["ms_total"] = (time.perf_counter() - t) * 1000; rec["ms_model"] = tm * 1000
                    for c in cells: c["bbox"] = [c["bbox"][0] - a.pad, c["bbox"][1] - a.pad, c["bbox"][2] - a.pad, c["bbox"][3] - a.pad]
                    rec["cells"] = cells
                except Exception as e:
                    rec["cells"] = []; rec["err"] = repr(e)[:300]
                    traceback.print_exc()
                f.write(json.dumps(rec) + "\n"); n += 1
        print("done", a.model, s, a.shard, n, flush=True)


if __name__ == "__main__":
    main()
