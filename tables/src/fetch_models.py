"""Download model files into $TB_WORK/models (never committed; cached by actions/cache)."""
import os, sys, hashlib, shutil, urllib.request
from paths import MODELS, TATR_SRC
from huggingface_hub import hf_hub_download

TATR_COMMIT = "16d124f616109746b7785f03085100f1f6247575"
POTATR_COMMIT = "fdb8f57c5124ff27ad7c12bcb976b48a615f1131"
TFV2_REV = "e41ba5da7ef69004f4b329168791c2987abf917c"
SHA = {
    "tatr/model_quantized.onnx": "83841e84bedb3af591a07888cf518333c88055ae0ab7bbd936c9289c5d8da82b",
    "slanet1m/slanet_1m.onnx": "8a8aa31bf964c1c05039f02814e5f425a354a37552eaca8e6d5dc513048759f5",
    "potatr/potatr.onnx": "499b5c090a6737037b1789a253773032842a89d5c540116fee5f00a7b43354a6",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def place(src, rel):
    dst = os.path.join(MODELS, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst): shutil.copy(src, dst)
    if rel in SHA: assert sha(dst) == SHA[rel], (rel, sha(dst))
    return dst


def main(which):
    which = which.split("-")[0]
    if which.startswith("tatr"):
        X = "Xenova/table-transformer-structure-recognition-v1.1-all"
        place(hf_hub_download(X, "onnx/model_quantized.onnx"), "tatr/model_quantized.onnx")
        if which in ("tatr32", "tatr4"): place(hf_hub_download(X, "onnx/model.onnx"), "tatr/model.onnx")
        if which == "tatr16": place(hf_hub_download(X, "onnx/model_fp16.onnx"), "tatr/model_fp16.onnx")
        q = os.path.join(MODELS, "tatr", "model_int4.onnx")
        if which == "tatr4" and not os.path.exists(q):
            import onnx
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            qz = MatMulNBitsQuantizer(onnx.load(os.path.join(MODELS, "tatr", "model.onnx")), block_size=32, is_symmetric=True)
            qz.process()
            qz.model.save_model_to_file(q, use_external_data_format=False)
        for f in sorted(os.listdir(os.path.join(MODELS, "tatr"))): print("tatr file", f, os.path.getsize(os.path.join(MODELS, "tatr", f)), sha(os.path.join(MODELS, "tatr", f)))
        os.makedirs(TATR_SRC, exist_ok=True)
        for fn in ("inference.py", "grits.py"):
            dst = os.path.join(TATR_SRC, fn)
            if not os.path.exists(dst):
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/microsoft/table-transformer/{TATR_COMMIT}/src/{fn}", dst)
    elif which == "grits":
        os.makedirs(TATR_SRC, exist_ok=True)
        dst = os.path.join(TATR_SRC, "grits.py")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/microsoft/table-transformer/{TATR_COMMIT}/src/grits.py", dst)
    elif which == "slanet1m":
        place(hf_hub_download("bdatdo0601/slanet-1m-onnx", "slanet_1m.onnx"), "slanet1m/slanet_1m.onnx")
        place(hf_hub_download("bdatdo0601/slanet-1m-onnx", "inference.yml"), "slanet1m/inference.yml")
    elif which == "tcn":
        place(hf_hub_download("Felix92/onnxtr-tablecenternet", "model.onnx"), "tcn/model.onnx")
        print("tcn sha256", sha(os.path.join(MODELS, "tcn", "model.onnx")))
    elif which in ("potatr", "potatr8"):
        place(hf_hub_download("opticsWolf/potatr-onnx", "potatr.onnx"), "potatr/potatr.onnx")
        cfg = os.path.join(MODELS, "potatr", "potatr_v1.json")
        if not os.path.exists(cfg):
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/kensho-technologies/POTATR/{POTATR_COMMIT}/configs/model/potatr_v1.json", cfg)
        q = os.path.join(MODELS, "potatr", "potatr_int8.onnx")
        if which == "potatr8" and not os.path.exists(q):
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(os.path.join(MODELS, "potatr", "potatr.onnx"), q, weight_type=QuantType.QInt8)
        for f in sorted(os.listdir(os.path.join(MODELS, "potatr"))): print("potatr file", f, os.path.getsize(os.path.join(MODELS, "potatr", f)))
    elif which in ("tfv2", "tfv2q"):
        from huggingface_hub import snapshot_download
        snapshot_download("docling-project/TableFormerV2", revision=TFV2_REV, local_dir=os.path.join(MODELS, "tfv2"))
        if which == "tfv2q":
            import tfv2_onnx
            tfv2_onnx.export()
    elif which == "tableformer":
        import docling_rs
        docling_rs.download_models()
    print("models ok", which)


if __name__ == "__main__":
    for w in sys.argv[1:]: main(w)
