import hashlib, json, os, shutil, sys
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(ROOT, "models_ref")
PP = ("x3zvawq/paddleocr-js-onnx", "51c2133b5a7ea27b795fa8c400fdbfbd5337dd6a",
      ["PP_FormulaNet_plus_S/PP-FormulaNet_plus-S_infer.onnx", "PP_FormulaNet_plus_S/config.json", "PP_FormulaNet_plus_S/inference.yml"])
UR = ("topdu/unirec_0_1b_onnx", "e8f55f72013d544cbd6c27c576946ac63da247ed",
      ["unirec_encoder.onnx", "unirec_decoder.onnx", "unirec_tokenizer_mapping.json"])
FN = ("Felix92/onnxtr-formulanet", "b1ef3bee0dc424480fe5afd833338c605ebec68d", ["model.onnx", "config.json", "README.md"])
EXPECT = {"doctr_formulanet/model.onnx": "3d2c732e4de80a1375ada58e827de51b9ba7d387210471bb568808abe0c8c043","unirec/unirec_encoder.onnx": "eb6b5d38f16c4f1abc39bab47e8d4ec83103b3246f642bde599edf6634289392",
          "unirec/unirec_decoder.onnx": "334326c877856d7595614424f1e15d6b1aab6ba24c26c769f09ab38901d5021c"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


os.makedirs(RD, exist_ok=True)
for f in PP[2]:
    hf_hub_download(PP[0], f, revision=PP[1], local_dir=RD)
for f in UR[2]:
    hf_hub_download(UR[0], f, revision=UR[1], local_dir=os.path.join(RD, "unirec"))
for f in FN[2]:
    hf_hub_download(FN[0], f, revision=FN[1], local_dir=os.path.join(RD, "doctr_formulanet"))
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
src = os.path.join(RD, "doctr_formulanet", "model.onnx")
os.makedirs(os.path.join(RD, "doctr_formulanet_int8"), exist_ok=True)
dst = os.path.join(RD, "doctr_formulanet_int8", "model.onnx")
pre = dst + ".pre.onnx"
try:
    quant_pre_process(src, pre, skip_symbolic_shape=True)
    print("quant_pre_process ok")
except Exception as e:
    print("quant_pre_process failed, quantizing the raw graph:", type(e).__name__, str(e)[:300])
    pre = src
quantize_dynamic(pre, dst, weight_type=QuantType.QInt8, per_channel=False)
if pre != src:
    os.remove(pre)
for d in [os.path.join(RD, ".cache"), os.path.join(RD, "unirec", ".cache"), os.path.join(RD, "doctr_formulanet", ".cache")]:
    shutil.rmtree(d, ignore_errors=True)
info = {}
for dp, _, fs in os.walk(RD):
    for f in fs:
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, RD).replace("\\", "/")
        info[rel] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
for rel, h in EXPECT.items():
    if info[rel]["sha256"] != h:
        sys.exit("hash mismatch " + rel)
json.dump(info, open(os.path.join(RD, "models_info.json"), "w"), indent=1, sort_keys=True)
print(json.dumps(info, indent=1, sort_keys=True))
