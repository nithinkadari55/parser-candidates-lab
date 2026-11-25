"""Round 8 model files that tables/src/fetch_models.py does not fetch: PP-LCNet_x1_0_doc_ori, the official ONNX export
(PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx, model card: Apache-2.0; the repo has no separate LICENSE file), pinned."""
import os, sys, hashlib, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common8 import *
from huggingface_hub import hf_hub_download

REPO, REV = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx", "7330ab7039123e46af2dc03154b9969aa412c61d"
SHA = "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92"

if __name__ == "__main__":
    d = os.path.join(os.environ.get("R8_MODELS", os.path.join(WORK, "models")), "pplcnet"); os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "inference.onnx")
    if not os.path.exists(dst): shutil.copy(hf_hub_download(REPO, "inference.onnx", revision=REV), dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    assert h == SHA, h
    shutil.copy(hf_hub_download(REPO, "README.md", revision=REV), os.path.join(d, "README.md"))
    lic = [l.strip() for l in open(os.path.join(d, "README.md"), encoding="utf-8") if l.startswith("license:")]
    print("pplcnet", os.path.getsize(dst), h, lic, flush=True)
