"""Round 9 candidate weights: where each comes from (pinned where the host allows), downloaded into $R9_W/<model>/.
Weights never leave the job (Actions cache only). usage: weights9.py MODEL [MODEL...]  (prints sizes and sha256)"""
import os, sys, json, hashlib, shutil, subprocess, urllib.request, zipfile, time

W = os.environ.get("R9_W", os.path.join(os.path.expanduser("~"), "r9w"))

HF = "https://huggingface.co"
MODELS = {
    # name: list of (url-or-spec, local relative path)
    "ae3": [("hf:APEXE3HQ/ae3-tl@d48bccd05164f70bdb6442476c6d4da93569ab08", "repo")],
    "mutab_ptn": [("https://github.com/JG1VPP/MuTabNet/releases/download/ICDAR2026/mutab_2f242f2_pubtabnet_mutual_w300_im520_global_html1_run1.pth", "model.pth")],
    "mutab_ftn": [("https://github.com/JG1VPP/MuTabNet/releases/download/ICDAR2026/mutab_2f242f2_fintabnet_mutual_w300_im520_global_html1_run1.pth", "model.pth")],
    "tablemaster": [(f"{HF}/opendatalab/PDF-Extract-Kit/resolve/72672acfc5552d95a6d002a6f2d36fb752d2965c/models/TabRec/TableMaster/{f}", f)
                    for f in ("table_structure_tablemaster_infer/inference.pdiparams", "table_structure_tablemaster_infer/inference.pdiparams.info",
                              "table_structure_tablemaster_infer/inference.pdmodel", "table_master_structure_dict.txt")],
    "rapid": [(f"https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0/{f}", f)
              for f in ("en_ppstructure_mobile_v2_SLANet.onnx", "ch_ppstructure_mobile_v2_SLANet.onnx", "slanet-plus.onnx")],
    "paddle_onnx": [(f"{HF}/PaddlePaddle/SLANet_onnx/resolve/94c4873d01ee3409d47801c8eef299aee9831b71/{f}", "SLANet/" + f) for f in ("inference.onnx", "inference.yml")]
                   + [(f"{HF}/PaddlePaddle/SLANet_plus_onnx/resolve/7dbe640e127602bf506815e822c09758de73c482/{f}", "SLANet_plus/" + f) for f in ("inference.onnx", "inference.yml")],
    "pptable": [(f"{HF}/PaddlePaddle/{m}/resolve/{rev}/{f}", f"{m}/{f}") for m, rev in (
        ("SLANet_plus", "bae6e5f8c3c4e7da0c0b7639fdf3228fe76184e2"), ("RT-DETR-L_wired_table_cell_det", "e2bd53c06b3a815d86acbf5c6779dada58819cfe"),
        ("RT-DETR-L_wireless_table_cell_det", "25ca86356a601c877476bb0dcc5fd09153d9d64d"), ("PP-LCNet_x1_0_table_cls", "2fa6323e7dab88fa883081db1460995f46af2922"))
        for f in ("inference.json", "inference.pdiparams", "inference.yml", "config.json")],
    "mineru": [(f"{HF}/opendatalab/MinerU-4_models_onnx/resolve/de70e917d417d8bdee7b985f9431d52b35efcd23/Table/{f}", f)
               for f in ("PP-LCNet_x1_0_table_cls.onnx", "unet.onnx", "slanet-plus.onnx")],
    "lore_wireless": [("gdrive:1cBaewRwlZF1tIZovT49HpJZ5wlb3nSCw", "ckpt")],
    "lore_ptn": [("gdrive:1hg5R42u_6xaoO-6Ft18Ctu86HB_N2Bzu", "ckpt")],
    "lore_wtw": [("gdrive:1n33c9jmGmjSfRbheleE1pqiIXBb_BCEw", "ckpt")],
    "trace_base": [("gdrive:1TbzFE0YGtqlfN9cgAJP59nnEzo1dOcjJ", "model.pth")],
    "trace_wtw": [("gdrive:1KmVwLgbcHmEXwGKVeYfVIqF0yjTf0UKI", "model.pth")],
    "tableseq": [("https://zenodo.org/records/21597715/files/checkpoints.zip?download=1", "checkpoints.zip")],
}


def get(url, dst, tries=4):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "table-bench/1.0"})
            with urllib.request.urlopen(req, timeout=900) as r, open(dst + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            os.replace(dst + ".tmp", dst); return
        except Exception as e:
            print("download retry", k, repr(e)[:160], flush=True); time.sleep(5 + 15 * k)
    raise RuntimeError("download failed: " + url)


def fetch(name):
    d = os.path.join(W, name); os.makedirs(d, exist_ok=True)
    done = os.path.join(d, ".done")
    if os.path.exists(done): return d
    for spec, rel in MODELS[name]:
        dst = os.path.join(d, rel)
        if spec.startswith("hf:"):
            from huggingface_hub import snapshot_download
            repo, rev = spec[3:].split("@")
            snapshot_download(repo, revision=rev, local_dir=dst)
        elif spec.startswith("gdrive:"):
            out = dst + ".download"
            subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", "-q", "-O", out, "https://drive.google.com/uc?id=" + spec[7:]], check=True)
            if zipfile.is_zipfile(out) and not rel.endswith((".pth", ".pt")):
                os.makedirs(dst, exist_ok=True); zipfile.ZipFile(out).extractall(dst); os.remove(out)
            else:
                os.replace(out, dst)
        else:
            get(spec, dst)
    open(done, "w").write("ok")
    return d


def report(name):
    d = os.path.join(W, name); rows = []
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            if f.startswith(".") or "/.cache/" in p.replace("\\", "/"): continue
            sz = os.path.getsize(p)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest() if sz < 3e9 else "-"
            rows.append((os.path.relpath(p, d), sz, h))
    for r in sorted(rows): print(f"  {name}: {r[0]} {r[1] / 1e6:.2f} MB sha256 {r[2][:16]}", flush=True)
    return rows


if __name__ == "__main__":
    for m in sys.argv[1:]:
        try:
            fetch(m); report(m)
        except Exception as e:
            print("FAILED", m, repr(e)[:300], flush=True)
