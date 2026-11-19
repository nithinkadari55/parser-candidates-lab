"""Download model files into $TB_WORK/models (never committed; cached by actions/cache)."""
import os, sys, hashlib, shutil, urllib.request
from paths import MODELS, TATR_SRC
from huggingface_hub import hf_hub_download

TATR_COMMIT = "16d124f616109746b7785f03085100f1f6247575"
SHA = {
    "tatr/model_quantized.onnx": "83841e84bedb3af591a07888cf518333c88055ae0ab7bbd936c9289c5d8da82b",
    "slanet1m/slanet_1m.onnx": "8a8aa31bf964c1c05039f02814e5f425a354a37552eaca8e6d5dc513048759f5",
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
    if which == "tatr":
        place(hf_hub_download("Xenova/table-transformer-structure-recognition-v1.1-all", "onnx/model_quantized.onnx"), "tatr/model_quantized.onnx")
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
    elif which == "tableformer":
        import docling_rs
        docling_rs.download_models()
    print("models ok", which)


if __name__ == "__main__":
    for w in sys.argv[1:]: main(w)
