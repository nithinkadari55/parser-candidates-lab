"""Train one pairwise order MLP on the dev split only (doc-separated validation inside dev).
usage: train6.py KIND OUT.npz OUT.json   KIND in gt, gt_stock, h8, h8_stock"""
import json, os, random, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats, load6, ro  # noqa: E402


def page_samples(p, kind, extras, rng, cap=3000):
    W, H = p["W"], p["H"]
    if kind.startswith("gt"):
        boxes, labels, texts = p["boxes"], p["labels"], p["texts"]
        c = feats.canon(boxes)
        rank = {k: c[k] for k in range(len(c))}
    else:
        dets = p["h8"]
        if len(dets) < 2:
            return None
        boxes = [d["box"] for d in dets]
        labels = [d["label"] for d in dets]
        texts = [""] * len(dets)
        c = feats.canon(boxes)
        r0 = load6.det_ranks(p["boxes"], [boxes[i] for i in c])
        rank = dict(r0)
    bc = [boxes[i] for i in c]
    lc = [labels[i] for i in c]
    tc = [texts[i] for i in c]
    o = ro.xycutpp(bc, W, H)
    xr = [0] * len(bc)
    for pos, i in enumerate(o):
        xr[i] = pos
    E, info = feats.element_features(bc, lc, tc, W, H, p["rects"], xr, extras)
    ks = [k for k in range(len(bc)) if k in rank]
    pairs = [(a, b) for a in ks for b in ks if a != b and rank[a] != rank[b]]
    if not pairs:
        return None
    if len(pairs) > cap:
        pairs = rng.sample(pairs, cap)
    ii = np.array([a for a, b in pairs])
    jj = np.array([b for a, b in pairs])
    X, _, _ = feats.pair_features(E, bc, W, H, info, xr, extras, pairs=(ii, jj))
    y = np.array([1.0 if rank[a] < rank[b] else 0.0 for a, b in pairs], np.float32)
    return X, y


def main():
    import torch
    kind, out, rep = sys.argv[1], sys.argv[2], sys.argv[3]
    extras = not kind.endswith("stock")
    torch.manual_seed(0)
    rng = random.Random(0)
    t0 = time.time()
    ps = [p for p in load6.pages() if p["split"] == "dev"]
    if kind.startswith("h8"):
        ps = [p for p in ps if p["have_h8"]]
    tr, va = [], []
    for p in ps:
        s = page_samples(p, kind, extras, rng)
        if s is None:
            continue
        (va if sum(map(ord, p["doc"])) % 7 == 0 else tr).append(s)
    Xtr = np.concatenate([s[0] for s in tr])
    ytr = np.concatenate([s[1] for s in tr])
    Xva = np.concatenate([s[0] for s in va])
    yva = np.concatenate([s[1] for s in va])
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    F = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(F, 2 * F), torch.nn.ReLU(), torch.nn.Linear(2 * F, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(ytr)
    Xv = torch.tensor((Xva - mu) / sd)
    yv = torch.tensor(yva)
    best, bad, bestw, hist = 1e9, 0, None, []
    for ep in range(60):
        net.train()
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 4096):
            b = perm[k:k + 4096]
            opt.zero_grad()
            l = lossf(net(Xt[b])[:, 0], yt[b])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = float(lossf(net(Xv)[:, 0], yv))
            acc = float(((net(Xv)[:, 0] > 0).float() == yv).float().mean())
        hist.append((ep, lv, acc))
        print(kind, "epoch", ep, "val loss %.4f acc %.4f" % (lv, acc), flush=True)
        if lv < best - 1e-4:
            best, bad = lv, 0
            bestw = [x.detach().clone() for x in net.parameters()]
        else:
            bad += 1
            if bad >= 6:
                break
    w1, b1, w2, b2 = [x.numpy() for x in bestw]
    np.savez(out, w1=w1, b1=b1, w2=w2, b2=b2, mu=mu, sd=sd, extras=extras)
    params = int(w1.size + b1.size + w2.size + b2.size)
    json.dump(dict(kind=kind, features=F, hidden=2 * F, params=params, fp32_kb=params * 4 / 1024, npz_bytes=os.path.getsize(out),
                   train_pages=len(tr), val_pages=len(va), train_pairs=int(len(Xtr)), val_pairs=int(len(Xva)), best_val_loss=best,
                   val_pair_acc=max(h[2] for h in hist), epochs=len(hist), seconds=time.time() - t0), open(rep, "w"), indent=1)
    print(open(rep).read())


if __name__ == "__main__":
    main()
