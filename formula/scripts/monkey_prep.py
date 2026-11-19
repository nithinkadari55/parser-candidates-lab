"""Download MonkeyOCRv2-S formula weights (internal testing only), strip training state, measure fp32/fp16 sizes,
export ONNX (encoder; decoder without KV cache), check parity with PyTorch, build INT8 and split decoders."""
import hashlib, json, os, subprocess, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, "models_monkey")
REPOS = [("wanderkid/unimernet_tiny", "3f09ac4b1cd583be47ea20a7d7daef839473028a", ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"], "unimernet_tiny"),
         ("zenosai/MonkeyOCRv2-S", "8efa4f856af5d38030d83565c8a79b36e5454ce7", ["config.json", "model.safetensors"], "monkeyocrv2_vit"),
         ("zhd36/monkeyocrv2_s_formula", "0dfb2643a0c89791f687650570641124d8b20bdd", ["monkeyocrv2_s_formula.pth"], "raw")]
INFO = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download():
    from huggingface_hub import hf_hub_download
    for repo, rev, files, sub in REPOS:
        for f in files:
            hf_hub_download(repo, f, revision=rev, local_dir=os.path.join(WD, sub))


def strip():
    import torch
    raw = os.path.join(WD, "raw", "monkeyocrv2_s_formula.pth")
    INFO["checkpoint_bytes"] = os.path.getsize(raw)
    INFO["checkpoint_sha256"] = sha(raw)
    ck = torch.load(raw, map_location="cpu", weights_only=False)
    INFO["checkpoint_keys"] = sorted(ck) if isinstance(ck, dict) else str(type(ck))
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    INFO["n_tensors"] = len(sd)
    INFO["n_params"] = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    INFO["prefixes"] = sorted({".".join(k.split(".")[:3]) for k in sd})[:10]
    p32 = os.path.join(WD, "model_only_fp32.pth")
    torch.save({"model": sd}, p32)
    p16 = os.path.join(WD, "model_only_fp16.pth")
    torch.save({"model": {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}}, p16)
    INFO["model_only_fp32_bytes"] = os.path.getsize(p32)
    INFO["model_only_fp16_bytes"] = os.path.getsize(p16)
    INFO["vit_safetensors_bytes"] = os.path.getsize(os.path.join(WD, "monkeyocrv2_vit", "model.safetensors"))
    os.remove(p16)
    os.remove(raw)
    del ck


def export():
    import torch
    from monkey_torch import MonkeyTorch
    mt = MonkeyTorch(4)
    vedm = mt.model.model.model
    vedm.eval()

    class Enc(torch.nn.Module):
        def forward(self, pixel_values):
            h = vedm.encoder(pixel_values, return_dict=True)[0]
            if vedm.encoder.config.hidden_size != vedm.decoder.config.hidden_size and vedm.decoder.config.cross_attention_hidden_size is None:
                h = vedm.enc_to_dec_proj(h)
            return h

    class Dec(torch.nn.Module):
        def forward(self, input_ids, encoder_hidden_states):
            return vedm.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False, return_dict=True).logits

    from PIL import Image
    img = os.path.join(ROOT, "crops", "f08_eq3_teds.png")
    with Image.open(img) as im:
        x = mt.proc(im.convert("RGB"))
    x = torch.stack([x], 0)
    od = os.path.join(WD, "onnx_fp32")
    os.makedirs(od, exist_ok=True)
    with torch.no_grad():
        h = Enc()(x)
        ids = torch.tensor([[0, 5, 6]], dtype=torch.long)
        torch.onnx.export(Enc(), (x,), os.path.join(od, "encoder_model.onnx"), input_names=["pixel_values"],
                          output_names=["encoder_hidden_states"], opset_version=17, dynamo=False)
        torch.onnx.export(Dec(), (ids, h), os.path.join(od, "decoder_model.onnx"), input_names=["input_ids", "encoder_hidden_states"],
                          output_names=["logits"], dynamic_axes={"input_ids": {1: "seq"}}, opset_version=17, dynamo=False)
    INFO["onnx_fp32_bytes"] = sum(os.path.getsize(os.path.join(od, f)) for f in os.listdir(od))


def quant_split():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    od, qd = os.path.join(WD, "onnx_fp32"), os.path.join(WD, "onnx_int8")
    os.makedirs(qd, exist_ok=True)
    for f in ("encoder_model.onnx", "decoder_model.onnx"):
        quantize_dynamic(os.path.join(od, f), os.path.join(qd, f), weight_type=QuantType.QInt8, per_channel=False)
    INFO["onnx_int8_bytes"] = sum(os.path.getsize(os.path.join(qd, f)) for f in os.listdir(qd))
    for d in (od, qd):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "split_dec.py"), os.path.join(d, "decoder_model.onnx"), os.path.join(d, "split")])


def parity():
    from monkey_torch import MonkeyTorch
    from refmodels import MonkeyOnnx
    mt = MonkeyTorch(4)
    res = {}
    gt = json.load(open(os.path.join(ROOT, "gt.json"), encoding="utf-8"))
    imgs = [os.path.join(ROOT, "crops", k + ".png") for k in gt if not k.startswith("n")]
    ref = {p: mt(p) for p in imgs}
    for v in ("fp32", "int8"):
        mo = MonkeyOnnx(4, v)
        same = 0
        for p in imgs:
            same += mo(p)[0] == ref[p][0]
        res[v] = f"{same}/{len(imgs)} identical to PyTorch"
    INFO["onnx_parity_own12"] = res
    INFO["torch_own12_preds"] = {os.path.basename(p): ref[p][0] for p in imgs}


if __name__ == "__main__":
    os.makedirs(WD, exist_ok=True)
    for step in (download, strip, export, quant_split, parity):
        t = time.time()
        try:
            step()
            INFO[step.__name__] = f"ok {round(time.time() - t)}s"
        except Exception as e:
            traceback.print_exc()
            INFO[step.__name__] = f"failed: {type(e).__name__}: {str(e)[:500]}"
            if step in (download, strip):
                break
    json.dump(INFO, open(os.path.join(WD, "monkey_info.json"), "w"), indent=1)
    print(json.dumps(INFO, indent=1))
