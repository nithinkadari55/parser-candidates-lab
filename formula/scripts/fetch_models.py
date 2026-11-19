import hashlib, json, os, subprocess, sys, urllib.request
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models")
P15_REPO, P15_REV = "breezedeus/pix2text-mfr-1.5", "1cef9f0bdcd6a4c63df7de1311fb0894593340cc"
EXPECT = {
    "breezedeus__pix2text-mfr-1.5/encoder_model.onnx": "080a3f660f08bc9ebcacdd96e34be6b6400f8c7e62d7cd0dd8251badc37f610b",
    "breezedeus__pix2text-mfr-1.5/decoder_model.onnx": "917deb98e91a0453c5f234f58a0f32f9fb037de8527c7eb4ed394daf9e692f2a",
    "breezedeus__pix2text-mfr-1.5/tokenizer.json": "4ffbeb2143e6a38324bb6111b7a8109530d38a076a8439aa5777535f0a32758a",
}
LOCAL_OWN = {
    "own_int8_mfr15/encoder_model.onnx": "1b9d39354f2d479191923d621bed43ac4aff7abdbe997d9badd3b6154651e453",
    "own_int8_mfr15/decoder_model.onnx": "55c6cd3ec1659d6d35991d3c8ca1df003eafb150957ab3080e5706321dc9e395",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(MD, exist_ok=True)
    snapshot_download(P15_REPO, revision=P15_REV, local_dir=os.path.join(MD, "breezedeus__pix2text-mfr-1.5"),
                      allow_patterns=["*.onnx", "*.json", "README.md"])
    for rel, h in EXPECT.items():
        got = sha(os.path.join(MD, rel))
        if got != h:
            sys.exit(f"hash mismatch {rel}: {got}")
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "quant.py"),
                           os.path.join(MD, "breezedeus__pix2text-mfr-1.5"), os.path.join(MD, "own_int8_mfr15")])
    for name, dec in [("p2t15_int8_own", "own_int8_mfr15/decoder_model.onnx")]:
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "split_dec.py"), os.path.join(MD, dec),
                               os.path.join(MD, "split_" + name)])
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "build_variants.py")])
    info = {}
    for dp, _, fs in os.walk(MD):
        for f in fs:
            if f.endswith((".onnx", "tokenizer.json")):
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, MD).replace("\\", "/")
                info[rel] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
    for rel, h in LOCAL_OWN.items():
        info[rel]["same_as_local_laptop_build"] = info[rel]["sha256"] == h
    json.dump(info, open(os.path.join(MD, "models_info.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(info, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
