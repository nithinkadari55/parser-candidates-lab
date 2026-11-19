import json, os, re
import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(ROOT, "models_ref")


def _session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class PPFormula:
    """PP-FormulaNet_plus-S, community ONNX conversion; pre/post-processing follows PaddleX formula_recognition."""

    SIZE = 384
    MEAN, STD = 0.7931, 0.1738

    def __init__(self, threads=4):
        from tokenizers import Tokenizer
        d = os.path.join(RD, "PP_FormulaNet_plus_S")
        cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        self.tok = Tokenizer.from_str(json.dumps(cfg["PostProcess"]["character_dict"]["fast_tokenizer_file"]))
        self.sess = _session(os.path.join(d, "PP-FormulaNet_plus-S_infer.onnx"), threads)
        self.inp = self.sess.get_inputs()[0].name
        outs = self.sess.get_outputs()
        self.out = next((o.name for o in outs if "int64" in o.type), outs[0].name)

    @staticmethod
    def crop_margin(img):
        import cv2
        data = np.array(img.convert("L")).astype(np.uint8)
        mx, mn = data.max(), data.min()
        if mx == mn:
            return img
        data = (data - mn) / (mx - mn) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        a, b, w, h = cv2.boundingRect(cv2.findNonZero(gray))
        return img.crop((a, b, w + a, h + b))

    def preprocess(self, path):
        import cv2
        arr = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = self.crop_margin(Image.fromarray(arr).convert("RGB"))
        w, h = img.size
        short, long = (w, h) if w <= h else (h, w)
        ns, nl = self.SIZE, int(self.SIZE * long / short)
        nw, nh = (ns, nl) if w <= h else (nl, ns)
        img = img.resize((nw, nh), resample=2)
        img.thumbnail((self.SIZE, self.SIZE))
        dw, dh = self.SIZE - img.width, self.SIZE - img.height
        pw, ph = dw // 2, dh // 2
        img = np.array(ImageOps.expand(img, (pw, ph, dw - pw, dh - ph)))
        x = (img.astype("float32") * (1 / 255.0) - np.float32(self.MEAN)) / np.float32(self.STD)
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return g[None, None, :, :].astype(np.float32)

    @staticmethod
    def normalize(s):
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = r"[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = []
        for x in re.findall(text_reg, s):
            pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
            matches = re.findall(pattern, x[0])
            for m in matches:
                if m not in ["\\operatorname", "\\mathrm", "\\text", "\\mathbf"] and m.strip() != "":
                    s = s.replace(m, m + "XXXXXXX")
                    s = s.replace(" ", "")
                    names.append(s)
        if len(names) > 0:
            s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s.replace("XXXXXXX", " ")

    def post_process(self, text):
        from ftfy import fix_text
        text = re.compile(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}").sub(lambda m: m.group(1), text).replace('"', "")
        text = fix_text(text)
        return self.normalize(text)

    def __call__(self, path):
        x = self.preprocess(path)
        ids = np.asarray(self.sess.run([self.out], {self.inp: x})[0])[0].astype(np.int64)
        end = np.argwhere(ids == 2)
        runaway = len(end) == 0
        if not runaway:
            ids = ids[: int(end[0][0]) + 1]
        text = self.tok.decode(ids.tolist(), skip_special_tokens=True)
        return self.post_process(text), int(len(ids)), runaway


class UniRec:
    """UniRec-0.1B ONNX; follows OpenOCR tools/infer_unirec_onnx.py (greedy, KV cache, cross K/V once)."""

    MAX_W, MAX_H, DIV = 960, 1408, 64
    MAX_NEW = 1024

    def __init__(self, threads=4):
        d = os.path.join(RD, "unirec")
        self.enc = _session(os.path.join(d, "unirec_encoder.onnx"), threads)
        self.dec = _session(os.path.join(d, "unirec_decoder.onnx"), threads)
        mp = json.load(open(os.path.join(d, "unirec_tokenizer_mapping.json"), encoding="utf-8"))
        self.id2tok = {int(k): v for k, v in mp["id_to_token"].items()}
        sp = mp["special_tokens"]
        self.bos, self.eos, self.pad = sp["bos_token_id"], sp["eos_token_id"], sp["pad_token_id"]
        self.past_names = [i for i in self.dec.get_inputs() if i.name.startswith("past_")]
        self.layers = len([i for i in self.past_names if i.name.startswith("past_key_")])
        shp = self.past_names[0].shape
        self.heads = shp[1] if isinstance(shp[1], int) else 6
        self.hdim = shp[3] if isinstance(shp[3], int) else 128

    def preprocess(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        ar = w / h
        if w > self.MAX_W or h > self.MAX_H:
            if (self.MAX_W / self.MAX_H) >= ar:
                nh = self.MAX_H
                nw = int(nh * ar)
            else:
                nw = self.MAX_W
                nh = int(nw / ar)
        else:
            nw, nh = w, h
        fw = max(int(nw // self.DIV * self.DIV), 64)
        fh = max(int(nh // self.DIV * self.DIV), 64)
        img = img.resize((fw, fh), resample=Image.BICUBIC)
        x = np.array(img, dtype=np.float32)[:, :, :3] / 255.0
        x = (x - 0.5) / 0.5
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def clean(self, ids):
        text = "".join(self.id2tok.get(i, f"<unk_{i}>") for i in ids)
        text = text.replace("Ġ", " ").replace("Ċ", "\n")
        text = text.replace("<|bos|>", "").replace("<|eos|>", "").replace("<|pad|>", "")
        for p, r in [(r"-<\|sn\|>", ""), (r" <\|sn\|>", " "), (r"<\|sn\|>", " "), (r"<\|unk\|>", ""), (r"<s>", ""),
                     (r"</s>", ""), (r"\uffff", ""), (r"_{4,}", "___"), (r"\.{4,}", "...")]:
            text = re.sub(p, r, text)
        t = text.strip()
        t = re.sub(r"(\\\])\s*\(\d+\)\s*$", r"\1", t)
        for a, b in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$")):
            if t.startswith(a) and t.endswith(b) and len(t) >= len(a) + len(b):
                t = t[len(a):len(t) - len(b)].strip()
                break
        return t

    def __call__(self, path):
        x = self.preprocess(path)
        _, ck, cv = self.enc.run(None, {"pixel_values": x})[:3]
        past = {}
        for i in range(self.layers):
            past[f"past_key_{i}"] = np.zeros((1, self.heads, 0, self.hdim), np.float32)
            past[f"past_value_{i}"] = np.zeros((1, self.heads, 0, self.hdim), np.float32)
        ids = [self.bos]
        runaway = True
        for step in range(self.MAX_NEW):
            feeds = {"input_ids": np.array([[ids[-1]]], np.int64),
                     "position_ids": np.array([[self.pad + 1 + step]], np.int64),
                     "cross_k": ck, "cross_v": cv, **past}
            outs = self.dec.run(None, feeds)
            nxt = int(np.argmax(outs[0][0, -1, :]))
            for i in range(self.layers):
                past[f"past_key_{i}"] = outs[1 + 2 * i]
                past[f"past_value_{i}"] = outs[2 + 2 * i]
            ids.append(nxt)
            if nxt == self.eos:
                runaway = False
                break
        return self.clean(ids), len(ids) - 1, runaway


class DoctrFormula:
    """docTR FormulaNet through OnnxTR's own formula_predictor (pre-processing and post-processing unchanged)."""

    def __init__(self, threads=4, int8=False):
        import onnxruntime as ort
        from onnxtr.models import formula_predictor, formulanet
        from onnxtr.models.engine import EngineConfig
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cfg = EngineConfig(providers=["CPUExecutionProvider"], session_options=so)
        path = os.path.join(RD, "doctr_formulanet_int8" if int8 else "doctr_formulanet", "model.onnx")
        self.pred = formula_predictor(arch=formulanet(model_path=path, engine_cfg=cfg), batch_size=1)
        self.eos = len(self.pred.model.vocab)

    def __call__(self, path):
        img = np.array(Image.open(path).convert("RGB"))
        batch = self.pred.pre_processor([img])[0]
        out = self.pred.model(batch, return_model_output=True)
        latex, _ = out["preds"][0]
        idx = np.argmax(out["out_map"][0], axis=-1)
        ends = np.flatnonzero(idx == self.eos)
        n = int(ends[0]) if ends.size else len(idx)
        return latex, n, ends.size == 0


class MonkeyOnnx:
    """MonkeyOCRv2-S formula model exported to ONNX here (encoder + decoder without KV cache, decoder split so the
    cross-attention K/V is computed once). Pre/post-processing re-implemented from the official processor."""

    MEAN = np.array((0.48145466, 0.4578275, 0.40821073), np.float32)
    STD = np.array((0.26862954, 0.26130258, 0.27577711), np.float32)
    MAX_NEW = 1536

    def __init__(self, threads=4, variant="int8"):
        from tokenizers import Tokenizer
        d = os.path.join(ROOT, "models_monkey", "onnx_" + variant)
        self.enc = _session(os.path.join(d, "encoder_model.onnx"), threads)
        self.st = _session(os.path.join(d, "split", "dec_static.onnx"), threads)
        self.dy = _session(os.path.join(d, "split", "dec_dyn.onnx"), threads)
        self.st_out = [o.name for o in self.st.get_outputs()]
        self.dy_in = [i.name for i in self.dy.get_inputs()]
        self.tok = Tokenizer.from_file(os.path.join(ROOT, "models_monkey", "unimernet_tiny", "tokenizer.json"))

    def preprocess(self, path):
        with Image.open(path) as im:
            img = PPFormula.crop_margin(im.convert("RGB"))
        img = img.resize((672, 196), Image.Resampling.BICUBIC)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - self.MEAN) / self.STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def __call__(self, path):
        from ftfy import fix_text
        h = self.enc.run(None, {"pixel_values": self.preprocess(path)})[0]
        feeds = dict(zip(self.st_out, self.st.run(None, {"encoder_hidden_states": h})))
        if "encoder_hidden_states" in self.dy_in:
            feeds["encoder_hidden_states"] = h
        ids = [0]
        runaway = True
        for step in range(self.MAX_NEW):
            feeds["input_ids"] = np.array([ids], dtype=np.int64)
            logits = self.dy.run(None, feeds)[0]
            nxt = 2 if step == self.MAX_NEW - 1 else int(logits[0, -1].argmax())
            ids.append(nxt)
            if nxt == 2:
                runaway = step == self.MAX_NEW - 1
                break
        text = fix_text(self.tok.decode(ids[1:], skip_special_tokens=True))
        return text, len(ids) - 2, runaway
