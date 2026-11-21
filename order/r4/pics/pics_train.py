"""Gap 10: frozen backbone features + a small logistic head, exported to ONNX (backbone + head in one graph,
fp32 and dynamic int8), scored on the group-held-out test crops.
usage: pics_train.py mnv3|dinov2 PICSDIR OUT.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pics_data import CLASSES  # noqa: E402


def backbone(kind):
    import torch
    if kind == "mnv3":
        import timm
        m = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=True, num_classes=0).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        tf = timm.data.create_transform(**cfg)

        class F(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m = m

            def forward(s, x):
                return s.m(x)
        return F().eval(), tf, cfg
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    m = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    class F(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = m

        def forward(s, x):
            h = s.m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)

    def tf(pil):
        return torch.tensor(proc(images=pil, return_tensors="np")["pixel_values"][0])
    return F().eval(), tf, {"size": proc.crop_size}


def main():
    kind, pics, out = sys.argv[1], sys.argv[2], sys.argv[3]
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(4)
    man = json.load(open(os.path.join(pics, "manifest.json")))["items"]
    net, tf, cfg = backbone(kind)
    t0 = time.perf_counter()
    feats, xs = [], []
    with torch.no_grad():
        for k in range(0, len(man), 32):
            b = torch.stack([tf(Image.open(m["path"]).convert("RGB")) for m in man[k:k + 32]])
            feats.append(net(b).numpy())
    F = np.concatenate(feats)
    feat_s = time.perf_counter() - t0
    lab = np.array([CLASSES.index(m["label"]) if m["label"] in CLASSES else -1 for m in man])
    sp = np.array([m["split"] for m in man])
    grp = np.array([m["group"] for m in man])
    tr, te, dom = sp == "train", sp == "test", sp == "domain"
    sc = StandardScaler().fit(F[tr])
    Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
    best = None
    for C in (0.03, 0.1, 0.3, 1.0):
        fs = []
        for a, b in GroupKFold(3).split(Xtr, lab[tr], grp[tr]):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr[a], lab[tr][a])
            fs.append(f1_score(lab[tr][b], clf.predict(Xtr[b]), average="macro"))
        if best is None or np.mean(fs) > best[0]:
            best = (float(np.mean(fs)), C)
    clf = LogisticRegression(C=best[1], max_iter=5000).fit(Xtr, lab[tr])
    pred = clf.predict(Xte)
    res = {"kind": kind, "C": best[1], "cv_macro_f1": best[0], "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "feat_dim": int(F.shape[1]), "feature_seconds": feat_s,
           "sklearn": {"macro_f1": float(f1_score(lab[te], pred, average="macro")),
                       "report": classification_report(lab[te], pred, labels=list(range(len(CLASSES))),
                                                       target_names=CLASSES, output_dict=True, zero_division=0),
                       "confusion": confusion_matrix(lab[te], pred, labels=list(range(len(CLASSES)))).tolist()}}
    W = clf.coef_ / sc.scale_[None, :]
    bvec = clf.intercept_ - (clf.coef_ * (sc.mean_ / sc.scale_)[None, :]).sum(1)

    class Full(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.f = net
            s.lin = torch.nn.Linear(W.shape[1], W.shape[0])
            s.lin.weight.data = torch.tensor(W, dtype=torch.float32)
            s.lin.bias.data = torch.tensor(bvec, dtype=torch.float32)

        def forward(s, x):
            return s.lin(s.f(x))
    full = Full().eval()
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(out)), f"pic_{kind}.onnx")
    ex = torch.stack([tf(Image.open(man[0]["path"]).convert("RGB"))])
    torch.onnx.export(full, ex, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}}, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
    import onnxruntime as ort
    te_idx = np.where(te)[0]
    dom_idx = np.where(dom)[0]
    for name, p in (("onnx_fp32", onnx_path), ("onnx_int8", q_path)):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        s = ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
        preds, ms = [], []
        for i in te_idx:
            x = tf(Image.open(man[i]["path"]).convert("RGB")).numpy()[None]
            t = time.perf_counter()
            lg = s.run(None, {"input": x})[0]
            ms.append((time.perf_counter() - t) * 1000)
            preds.append(int(lg.argmax()))
        dpred = [int(s.run(None, {"input": tf(Image.open(man[i]["path"]).convert("RGB")).numpy()[None]})[0].argmax())
                 for i in dom_idx]
        ms.sort()
        res[name] = {"macro_f1": float(f1_score(lab[te], preds, average="macro")),
                     "report": classification_report(lab[te], preds, labels=list(range(len(CLASSES))),
                                                     target_names=CLASSES, output_dict=True, zero_division=0),
                     "agree_sklearn": float(np.mean(np.array(preds) == pred)),
                     "mb": os.path.getsize(p) / 1e6, "ms_p50": ms[len(ms) // 2], "ms_p95": ms[int(0.95 * (len(ms) - 1))],
                     "pred": preds, "domain_pred": dpred}
    res["test_ids"] = [int(man[i]["id"]) for i in te_idx]
    res["test_src"] = [man[i]["src"] for i in te_idx]
    res["domain_ids"] = [int(man[i]["id"]) for i in dom_idx]
    res["labels"] = lab[te].tolist()
    json.dump(res, open(out, "w"))
    print(kind, "macro-F1", res["sklearn"]["macro_f1"], res["onnx_fp32"]["macro_f1"], res["onnx_int8"]["macro_f1"])


if __name__ == "__main__":
    main()
