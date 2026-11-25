"""Round 8, test A: one arm of the picture-classifier comparison on all crops.
  docling : DocumentFigureClassifier-v2.5 (picture_classifier.onnx of docling.rs models-v1), preprocessed as
            docling.rs enrich.rs does (224x224 bilinear squash, RGB/255, its mean/std); stores the 26 probabilities.
  mnv3    : timm mobilenetv3_small_100.lamb_in1k, frozen; ONNX fp32 backbone features + logistic-regression head.
  dinov2  : facebook/dinov2-small (CLS + mean patch token), frozen; ONNX fp32 and dynamic-int8 backbone features
            + logistic-regression head (the int8 arm = int8 backbone with the same fp32 head).
Heads are scored leave-one-source-out: for every source in the pool, a head trained on all other sources
(variant "all") and one trained only on the other permissively licensed sources (variant "perm") predict it.
C is chosen inside each fold by grouped 3-fold CV over the training sources. Test-only sources (hand-labelled
real-document crops) are predicted by heads trained on the whole pool. Chart subtypes (bar/line/pie/scatter/box)
get their own leave-one-source-out head on chart crops whose source gives the subtype.
usage: p8_model.py KIND DATAROOT OUTDIR [--probe]"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p8_common import (CLASSES, D26, DOCLING_MEAN, DOCLING_STD, MEAN_IMAGENET, SOURCES, STD_IMAGENET,  # noqa: E402
                       SUBTYPES, load_manifest, to_tensor)

DOCLING_URL = "https://github.com/docling-project/docling.rs/releases/download/models-v1/picture_classifier.onnx"
CS = (0.01, 0.1, 1.0)
PROBE = False


def session(path, threads):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def load_img(m):
    from PIL import Image
    return Image.open(m["path"]).convert("RGB")


def timing(sess, man, mean, std, n=300):
    name = sess.get_inputs()[0].name
    ms = []
    for m in man[:: max(1, len(man) // n)][:n]:
        x = to_tensor(load_img(m), mean, std)[None]
        t = time.perf_counter()
        sess.run(None, {name: np.ascontiguousarray(x)})
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    return {"ms_p50": ms[len(ms) // 2], "ms_p95": ms[int(0.95 * (len(ms) - 1))], "n": len(ms)}


def run_docling(man, out):
    p = os.path.join(out, "picture_classifier.onnx")
    if not os.path.exists(p):
        urllib.request.urlretrieve(DOCLING_URL, p)
    s4 = session(p, 4)
    name = s4.get_inputs()[0].name
    rows = []
    t0 = time.time()
    for i, m in enumerate(man):
        x = to_tensor(load_img(m), DOCLING_MEAN, DOCLING_STD)[None]
        lg = s4.run(None, {name: np.ascontiguousarray(x)})[0][0].astype(np.float64)
        e = np.exp(lg - lg.max())
        pr = e / e.sum()
        rows.append({"id": m["id"], "p26": [round(float(v), 5) for v in pr]})
        if i % 2000 == 0:
            print("docling", i, len(man), f"{time.time() - t0:.0f} s", flush=True)
    t = timing(session(p, 1), man, DOCLING_MEAN, DOCLING_STD)
    info = {"kind": "docling", "mb": os.path.getsize(p) / 1e6, **t, "classes": D26}
    os.remove(p)
    return rows, info


def backbone(kind):
    import torch
    if kind == "mnv3":
        import timm
        m = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=True, num_classes=0).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        print("timm cfg", cfg, flush=True)
        mean, std = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)

        class F(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.m = m

            def forward(s, x):
                return s.m(x)
        return F().eval(), mean, std
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    class F(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = m

        def forward(s, x):
            h = s.m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
    return F().eval(), MEAN_IMAGENET, STD_IMAGENET


def export(net, path):
    import torch
    ex = torch.zeros(2, 3, 224, 224)
    torch.onnx.export(net, ex, path, input_names=["input"], output_names=["out"], opset_version=17,
                      dynamic_axes={"input": {0: "n"}, "out": {0: "n"}}, dynamo=False)


def quantize(src, dst, exclude_last=False):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    excl = []
    if exclude_last:
        # keep the linear head in fp32: the node that writes the graph output and, when that is the bias Add,
        # the MatMul / Gemm feeding it
        g = onnx.load(src).graph
        prod = {o: n for n in g.node for o in n.output}
        last = prod[g.output[0].name]
        excl = [last.name] + [prod[i].name for i in last.input if i in prod and prod[i].op_type in ("MatMul", "Gemm")]
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, nodes_to_exclude=excl)
    return excl


def features(path, man, mean, std, bs=32):
    s = session(path, 4)
    out = []
    t0 = time.time()
    for k in range(0, len(man), bs):
        x = np.stack([to_tensor(load_img(m), mean, std) for m in man[k:k + bs]])
        out.append(s.run(None, {"input": x})[0])
        if k % (bs * 100) == 0:
            print(os.path.basename(path), k, len(man), f"{time.time() - t0:.0f} s", flush=True)
    return np.concatenate(out).astype(np.float32)


def fit(X, y, g):
    """standardise + multinomial logistic regression, class_weight balanced; C by grouped 3-fold CV
    (groups = sources) when at least 3 sources are present"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    best = (None, CS[1])
    if len(set(g)) >= 3 and len(set(y)) > 1 and not PROBE:
        for C in CS:
            fs = []
            for a, b in GroupKFold(3).split(X, y, g):
                if len(set(y[a])) < 2:
                    continue
                sc = StandardScaler().fit(X[a])
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(X[a]), y[a])
                fs.append(f1_score(y[b], clf.predict(sc.transform(X[b])), average="macro"))
            if fs and (best[0] is None or np.mean(fs) > best[0]):
                best = (float(np.mean(fs)), C)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf, best[1]


def predict(model, X, names):
    sc, clf, _ = model
    p = clf.predict_proba(sc.transform(X))
    return [names[clf.classes_[i]] for i in p.argmax(1)]


def head_weights(model):
    """fold scaler + LR into one linear layer over raw features, all 9 classes (missing classes get -inf bias)"""
    sc, clf, _ = model
    W = np.zeros((len(CLASSES), clf.coef_.shape[1]), np.float32)
    b = np.full(len(CLASSES), -1e4, np.float32)
    coef = clf.coef_ if clf.coef_.shape[0] > 1 else np.vstack([-clf.coef_, clf.coef_])
    inter = clf.intercept_ if clf.coef_.shape[0] > 1 else np.array([-clf.intercept_[0], clf.intercept_[0]])
    for k, c in enumerate(clf.classes_):
        W[c] = coef[k] / sc.scale_
        b[c] = inter[k] - (coef[k] * sc.mean_ / sc.scale_).sum()
    return W, b


def run_head(kind, man, out):
    import torch
    from joblib import Parallel, delayed
    torch.set_num_threads(4)
    net, mean, std = backbone(kind)
    fp = os.path.join(out, f"{kind}_backbone.onnx")
    export(net, fp)
    paths = {"fp32": fp}
    if kind == "dinov2":
        q = os.path.join(out, f"{kind}_backbone_int8.onnx")
        quantize(fp, q)
        paths["int8"] = q
    F = {k: features(p, man, mean, std) for k, p in paths.items()}
    for k, v in F.items():
        print(kind, k, "features", v.shape, "finite", bool(np.isfinite(v).all()), flush=True)
    y = np.array([CLASSES.index(m["label"]) for m in man])
    src = np.array([m["src"] for m in man])
    role = np.array([SOURCES[m["src"]][3] for m in man])
    perm = np.array([SOURCES[m["src"]][2] for m in man])
    pool = role == "pool"
    pool_srcs = sorted(set(src[pool]))
    X = F["fp32"]

    def fold(s, variant):
        tr = pool & (src != s) & (perm if variant == "perm" else True)
        te = src == s
        if not tr.any():
            return s, variant, None, None
        model = fit(X[tr], y[tr], src[tr])
        res = {k: predict(model, F[k][te], CLASSES) for k in F}
        return s, variant, res, model[2]
    jobs = [(s, v) for s in pool_srcs for v in ("all", "perm")]
    t0 = time.time()
    outs = Parallel(n_jobs=4, prefer="threads", verbose=5)(delayed(fold)(s, v) for s, v in jobs)
    print("leave-one-source-out folds", len(jobs), f"{time.time() - t0:.0f} s", flush=True)
    pred = {m["id"]: {} for m in man}
    cs = {}
    for s, v, res, C in outs:
        cs[f"{s}/{v}"] = C
        if res is None:
            continue
        ids = [m["id"] for m in man if m["src"] == s]
        for k, ps in res.items():
            for i, p in zip(ids, ps):
                pred[i][f"{kind}_{k}_{v}"] = p
    # heads on the whole pool: test-only sources, and the exported graphs for size and speed
    info = {"kind": kind, "C": cs, "dims": int(X.shape[1])}
    test = role == "test"
    for v in ("all", "perm"):
        tr = pool & (perm if v == "perm" else True)
        model = fit(X[tr], y[tr], src[tr])
        info[f"C_final_{v}"] = model[2]
        for k in F:
            if test.any():
                for i, p in zip(np.where(test)[0], predict(model, F[k][test], CLASSES)):
                    pred[man[i]["id"]][f"{kind}_{k}_{v}"] = p
        if v != "perm":
            continue
        W, b = head_weights(model)

        class Full(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.f = net
                s.lin = torch.nn.Linear(W.shape[1], W.shape[0])
                s.lin.weight.data = torch.tensor(W)
                s.lin.bias.data = torch.tensor(b)

            def forward(s, x):
                return s.lin(s.f(x))
        full = os.path.join(out, f"{kind}_head_perm.onnx")
        export(Full().eval(), full)
        graphs = {"fp32": full}
        if kind == "dinov2":
            fq = full.replace(".onnx", "_int8.onnx")
            info["int8_excluded_nodes"] = quantize(full, fq, exclude_last=True)
            graphs["int8"] = fq
        for k, gp in graphs.items():
            s1 = session(gp, 1)
            t = timing(s1, man, mean, std)
            # the exported graph must reproduce the feature + head path
            idx = np.arange(0, len(man), max(1, len(man) // 200))[:200]
            g_pred = [int(s1.run(None, {"input": to_tensor(load_img(man[i]), mean, std)[None]})[0].argmax()) for i in idx]
            h_pred = [int((F[k][i] @ W.T + b).argmax()) for i in idx]
            info[f"graph_{k}"] = {"mb": os.path.getsize(gp) / 1e6, **t,
                                  "agree_with_feature_path": float(np.mean(np.array(g_pred) == np.array(h_pred)))}
    # chart subtypes
    sub_ok = np.array([m["label"] == "chart" and m.get("sub") in SUBTYPES for m in man])
    ys = np.array([SUBTYPES.index(m["sub"]) if sub_ok[j] else -1 for j, m in enumerate(man)])
    sub_srcs = sorted(set(src[sub_ok & pool]))
    for s in sub_srcs:
        tr = sub_ok & pool & (src != s)
        te = sub_ok & (src == s)
        if len(set(ys[tr])) < 2:
            continue
        model = fit(X[tr], ys[tr], src[tr])
        for k in F:
            for i, p in zip(np.where(te)[0], predict(model, F[k][te], SUBTYPES)):
                pred[man[i]["id"]][f"{kind}_{k}_sub"] = p
    for f in os.listdir(out):
        if f.endswith(".onnx"):
            os.remove(os.path.join(out, f))
    rows = [{"id": i, **p} for i, p in pred.items()]
    return rows, info


def main():
    global PROBE
    kind, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    PROBE = "--probe" in sys.argv
    os.makedirs(out, exist_ok=True)
    man, dropped = load_manifest(root)
    if kind == "docling":
        rows, info = run_docling(man, out)
    else:
        rows, info = run_head(kind, man, out)
    info["dropped_duplicates"] = dropped
    info["n_items"] = len(man)
    with open(os.path.join(out, f"p8-{kind}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(info, open(os.path.join(out, f"p8-{kind}-info.json"), "w"), indent=1)
    if kind == "docling":
        with open(os.path.join(out, "p8-items.jsonl"), "w", encoding="utf-8") as f:
            for m in man:
                f.write(json.dumps({k: m.get(k) for k in ("id", "src", "label", "group", "sub", "orig_type", "w", "h",
                                                          "licence")}) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "C"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
