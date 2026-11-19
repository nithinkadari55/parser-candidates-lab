"""Precision variants of Pix2Text MFR 1.5 (fp16, int16 QDQ, int4 weight-only). Each goes to models/<variant>/."""
import json, os, shutil, sys, traceback
import numpy as np
import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models")
SRC = os.path.join(MD, "breezedeus__pix2text-mfr-1.5")
PARTS = ["encoder_model.onnx", "decoder_model.onnx"]


def out_dir(name):
    d = os.path.join(MD, name)
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(SRC, "tokenizer.json"), d)
    return d


def fp16():
    from onnxconverter_common import float16
    d = out_dir("p2t15_fp16")
    for p in PARTS:
        m = float16.convert_float_to_float16(onnx.load(os.path.join(SRC, p)), keep_io_types=True)
        onnx.save(m, os.path.join(d, p))


def calib_feeds():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from mfr import MFR
    from PIL import Image
    m = MFR("p2t15_fp32", 4)
    gt = json.load(open(os.path.join(ROOT, "gt.json"), encoding="utf-8"))
    enc, dec = [], []
    for k in gt:
        if k.startswith("n"):
            continue
        im = Image.open(os.path.join(ROOT, "crops", k + ".png")).convert("RGB").resize((384, 384), Image.BICUBIC)
        x = ((np.asarray(im, dtype=np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        enc.append({"pixel_values": x})
        h = m.enc.run(None, {"pixel_values": x})[0]
        ids = [1]
        for step in range(200):
            if step % 3 == 0:
                dec.append({"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": h})
            logits = m.dec.run(None, {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": h})[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == 2:
                break
            ids.append(nxt)
    return {"encoder_model.onnx": enc, "decoder_model.onnx": dec}


def int16():
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, feeds):
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    feeds = calib_feeds()
    d = out_dir("p2t15_int16")
    for p in PARTS:
        pre = os.path.join(d, p + ".pre.onnx")
        quant_pre_process(os.path.join(SRC, p), pre, skip_symbolic_shape=True)
        quantize_static(pre, os.path.join(d, p), Reader(feeds[p]), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt16, weight_type=QuantType.QInt16, per_channel=False,
                        calibrate_method=CalibrationMethod.MinMax, extra_options={"UseQDQContribOps": True})
        os.remove(pre)
        for f in os.listdir(d):
            if f.startswith("augmented") or f.endswith(".data"):
                os.remove(os.path.join(d, f))


def int4(bs, mix):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.quantization import QuantType, quantize_dynamic
    name = f"p2t15_int4_b{bs}" + ("_int8rest" if mix else "")
    d = out_dir(name)
    for p in PARTS:
        q = MatMulNBitsQuantizer(onnx.load(os.path.join(SRC, p)), block_size=bs, is_symmetric=True)
        q.process()
        dst = os.path.join(d, p)
        q.model.save_model_to_file(dst, use_external_data_format=False)
        if mix:
            tmp = dst + ".nb.onnx"
            os.replace(dst, tmp)
            quantize_dynamic(tmp, dst, op_types_to_quantize=["Gather", "Conv"], weight_type=QuantType.QInt8)
            os.remove(tmp)


if __name__ == "__main__":
    status = {}
    for name, fn in [("p2t15_fp16", fp16), ("p2t15_int16", int16), ("p2t15_int4_b32", lambda: int4(32, False)),
                     ("p2t15_int4_b128", lambda: int4(128, False)), ("p2t15_int4_b128_int8rest", lambda: int4(128, True))]:
        try:
            fn()
            status[name] = "ok"
        except Exception as e:
            traceback.print_exc()
            status[name] = f"failed: {type(e).__name__}: {str(e)[:400]}"
            shutil.rmtree(os.path.join(MD, name), ignore_errors=True)
    json.dump(status, open(os.path.join(MD, "variants_status.json"), "w"), indent=1)
    print(json.dumps(status, indent=1))
