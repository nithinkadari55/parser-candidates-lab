import os, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models")
MODELS = {
    "p2t15_fp32": ("breezedeus__pix2text-mfr-1.5/encoder_model.onnx", "breezedeus__pix2text-mfr-1.5/decoder_model.onnx", "breezedeus__pix2text-mfr-1.5/tokenizer.json"),
    "p2t10_int8_mathorium": ("mathorium__pix2text-mfr-quantized/onnx/encoder_model.onnx", "mathorium__pix2text-mfr-quantized/onnx/decoder_model_int8.onnx", "mathorium__pix2text-mfr-quantized/tokenizer.json"),
    "p2t15_int8_own": ("own_int8_mfr15/encoder_model.onnx", "own_int8_mfr15/decoder_model.onnx", "own_int8_mfr15/tokenizer.json"),
}
MAX_NEW = 512


class MFR:
    def __init__(self, name, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        e, d, t = (os.path.join(MD, x) for x in MODELS[name])
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(e, so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(d, so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(t)
        self.start = 2 if "p2t10" in name else 1

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        im = img.convert("RGB").resize((384, 384), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]
        h = self.enc.run(None, {"pixel_values": x})[0]
        ids = [self.start]
        runaway = True
        for _ in range(MAX_NEW):
            logits = self.dec.run(None, {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": h})[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == 2:
                runaway = False
                break
            ids.append(nxt)
        return self.tok.decode(ids[1:], skip_special_tokens=True), len(ids) - 1, runaway


class MFRFast(MFR):
    """Same graphs, decoder split into encoder-only part (cross-attn K/V, computed once) and per-step part."""

    def __init__(self, name, threads=4):
        super().__init__(name, threads)
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        d = os.path.join(MD, "split_" + name)
        self.st = ort.InferenceSession(os.path.join(d, "dec_static.onnx"), so, providers=["CPUExecutionProvider"])
        self.dy = ort.InferenceSession(os.path.join(d, "dec_dyn.onnx"), so, providers=["CPUExecutionProvider"])
        self.st_out = [o.name for o in self.st.get_outputs()]
        self.dy_in = [i.name for i in self.dy.get_inputs()]

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        im = img.convert("RGB").resize((384, 384), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]
        h = self.enc.run(None, {"pixel_values": x})[0]
        feeds = dict(zip(self.st_out, self.st.run(None, {"encoder_hidden_states": h})))
        if "encoder_hidden_states" in self.dy_in:
            feeds["encoder_hidden_states"] = h
        ids = [self.start]
        runaway = True
        for _ in range(MAX_NEW):
            feeds["input_ids"] = np.array([ids], dtype=np.int64)
            logits = self.dy.run(None, feeds)[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == 2:
                runaway = False
                break
            ids.append(nxt)
        return self.tok.decode(ids[1:], skip_special_tokens=True), len(ids) - 1, runaway
