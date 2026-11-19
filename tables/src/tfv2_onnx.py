import os, sys, time
import numpy as np
import paths
from paths import MODELS

D = os.path.join(MODELS, "tfv2")
O = os.path.join(MODELS, "tfv2_onnx")


def export():
    import torch
    from docling_ibm_models.tableformer_v2 import TableFormerV2
    os.makedirs(O, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)
    m = TableFormerV2.from_pretrained(D).eval()

    class Enc(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, images): return s.m.encode_images(images)["last_hidden_state"]

    class Dec(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, input_ids, memory):
            L = input_ids.shape[1]
            tgt = s.m.input_embedding(input_ids) + s.m.positional_encoding[:, :L, :]
            mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
            dec, _ = s.m.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=mask, tgt_key_padding_mask=None, past_key_values=None, use_cache=False)
            return s.m.output_projection(dec), dec

    class Box(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, cell_embs, memory):
            idx = torch.zeros(cell_embs.shape[0], dtype=torch.long)
            return s.m.bbox_head(cell_embs, memory, idx)

    x = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        mem = Enc(m)(x)
        ids = torch.tensor([[2, 5, 5, 9, 5]])
        torch.onnx.export(Enc(m), (x,), os.path.join(O, "enc.onnx"), input_names=["images"], output_names=["memory"], opset_version=17, dynamo=False)
        torch.onnx.export(Dec(m), (ids, mem), os.path.join(O, "dec.onnx"), input_names=["input_ids", "memory"], output_names=["logits", "hidden"],
                          dynamic_axes={"input_ids": {1: "L"}, "logits": {1: "L"}, "hidden": {1: "L"}}, opset_version=17, dynamo=False)
        ce = torch.randn(3, m.config.embed_dim)
        torch.onnx.export(Box(m), (ce, mem), os.path.join(O, "box.onnx"), input_names=["cell_embs", "memory"], output_names=["boxes"],
                          dynamic_axes={"cell_embs": {0: "N"}, "boxes": {0: "N"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for n in ("enc", "dec", "box"):
        quantize_dynamic(os.path.join(O, n + ".onnx"), os.path.join(O, n + "_int8.onnx"), weight_type=QuantType.QInt8)
    for f in sorted(os.listdir(O)):
        print("tfv2 onnx", f, os.path.getsize(os.path.join(O, f)), flush=True)


class TFV2Q:
    PAD = 0
    CELL = {"fcel", "ecel", "ched", "rhed", "srow"}
    SUFFIX = "_int8"

    def __init__(self, threads):
        import onnxruntime as ort, json
        from bench_models import so
        if not os.path.exists(os.path.join(O, "dec" + self.SUFFIX + ".onnx")): export()
        mk = lambda n: ort.InferenceSession(os.path.join(O, n + self.SUFFIX + ".onnx"), so(threads), providers=["CPUExecutionProvider"])
        self.enc, self.dec, self.box = mk("enc"), mk("dec"), mk("box")
        tj = json.load(open(os.path.join(D, "tokenizer.json"), encoding="utf-8"))
        self.names = {t["id"]: t["content"].strip("<>") for t in tj.get("added_tokens", [])}
        cfg = json.load(open(os.path.join(D, "config.json")))
        self.cells_ids = set(cfg["data_cells"]); self.eos = cfg["eos_token_id"]; self.bos = 2
        import torchvision.transforms as T
        self.tf = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, im, toks):
        from bench_models import otsl_cells
        W, H = im.size
        x = self.tf(im).unsqueeze(0).numpy()
        t = time.perf_counter()
        mem = self.enc.run(None, {"images": x})[0]
        ids = [self.bos]
        for _ in range(511):
            lg, _h = self.dec.run(None, {"input_ids": np.array([ids], np.int64), "memory": mem})
            nx = int(lg[0, -1].argmax()); ids.append(nx)
            if nx == self.eos: break
        lg, hid = self.dec.run(None, {"input_ids": np.array([ids], np.int64), "memory": mem})
        pos = [i for i, v in enumerate(ids) if v in self.cells_ids]
        bb = self.box.run(None, {"cell_embs": hid[0, pos].astype(np.float32), "memory": mem})[0] if pos else np.zeros((0, 4))
        tm = time.perf_counter() - t
        return otsl_cells([self.names.get(i, "") for i in ids], bb, W, H, self.CELL), tm


class TFV2F(TFV2Q):
    SUFFIX = ""


if __name__ == "__main__":
    export()
