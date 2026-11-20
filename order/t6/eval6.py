"""Evaluate test-6 variants on one shard of pages (both splits; the report keeps dev and test apart).
usage: eval6.py MODELDIR SHARD NSHARDS OUT.jsonl"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats, load6, ro  # noqa: E402


def score(pred, n):
    s = ro.score(pred, list(range(n)))
    return {k: s[k] for k in ("tau", "ed", "exact")}


class Orderer:
    def __init__(self, boxes, labels, texts, W, H, rects):
        self.W, self.H, self.rects = W, H, rects
        self.c = feats.canon(boxes)
        self.b = [boxes[i] for i in self.c]
        self.l = [labels[i] for i in self.c]
        self.t = [texts[i] for i in self.c]
        self._xr = None
        self._P = {}

    def back(self, order):
        return [self.c[k] for k in order]

    def xy(self):
        o = ro.xycutpp(self.b, self.W, self.H)
        if self._xr is None:
            self._xr = [0] * len(o)
            for pos, i in enumerate(o):
                self._xr[i] = pos
        return o

    def P(self, model):
        if id(model) not in self._P:
            self.xy()
            self._P[id(model)] = model.matrix(self.b, self.l, self.t, self.W, self.H, self.rects, self._xr)
        return self._P[id(model)]

    def romlp(self, model):
        return feats.greedy_decode(self.P(model))

    def rr(self, model):
        o, st = feats.xycutpp_tree(self.b, self.W, self.H, rerank=feats.child_reranker(self.P(model)))
        return o, st

    def pdfpig(self):
        return feats.pdfpig_order(self.b, self.W, self.H)


def timed(fn):
    t = time.perf_counter()
    o = fn()
    return o, (time.perf_counter() - t) * 1000


def with_deco(n, deco, boxes, H, fn):
    keep = [i for i in range(n) if i not in deco]
    sub = fn(keep)
    top = sorted([i for i in deco if (boxes[i][1] + boxes[i][3]) / 2 < H / 2], key=lambda i: (boxes[i][1], boxes[i][0]))
    bot = sorted([i for i in deco if i not in top], key=lambda i: (boxes[i][1], boxes[i][0]))
    return top + sub + bot


def main():
    mdir, k, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    M = {name: feats.MLP(os.path.join(mdir, name + ".npz")) for name in ("gt", "gt_stock", "h8", "h8_stock") if os.path.exists(os.path.join(mdir, name + ".npz"))}
    ps = [p for i, p in enumerate(load6.pages()) if i % n == k]
    with open(out, "w", encoding="utf-8") as f:
        for p in ps:
            W, H = p["W"], p["H"]
            rec = {key: p[key] for key in ("id", "ds", "split", "layout", "source", "bucket", "tags", "doc")}
            nb = len(p["boxes"])
            rec["n"] = nb

            def put(name, fn, full=False, regs=None):
                try:
                    (o, ms) = timed(fn)
                except Exception as e:
                    rec[name] = dict(error=(type(e).__name__ + ": " + str(e))[:160])
                    return
                if regs is not None:
                    m = load6.match(p["boxes"], regs)
                    if len(m) < 2:
                        return
                    rank = {kk: i for i, kk in enumerate(o)}
                    gts = sorted(m)
                    pred = sorted(gts, key=lambda g: (rank[m[g]], p["boxes"][g][1], p["boxes"][g][0]))
                    s = ro.score(pred, gts)
                    rec[name] = dict(tau=s["tau"], ed=s["ed"], exact=s["exact"], cov=len(m) / nb, ms=ms)
                    return
                if full:
                    o = [p["fbody"][i] for i in o if p["fbody"][i] >= 0]
                if sorted(o) != list(range(nb)):
                    rec[name] = dict(error="bad permutation")
                    return
                s = score(o, nb)
                s["ms"] = ms
                if name in KEEP_PRED:
                    s["pred"] = o
                rec[name] = s

            # ---------------- ground-truth body boxes (headers/footers removed by GT labels)
            B = Orderer(p["boxes"], p["labels"], p["texts"], W, H, p["rects"])
            ref = ro.xycutpp(p["boxes"], W, H)
            put("xycutpp_gtorder_input", lambda: ref)
            put("xycutpp", lambda: B.back(B.xy()))
            tree0, _ = feats.xycutpp_tree(B.b, W, H)
            rec["tree_matches_xycutpp"] = tree0 == B.xy()
            put("pdfpig", lambda: B.back(B.pdfpig()))
            if "gt_stock" in M:
                put("romlp_stock", lambda: B.back(B.romlp(M["gt_stock"])))
            if "gt" in M:
                put("romlp", lambda: B.back(B.romlp(M["gt"])))
                st = {}

                def rr():
                    o, s = B.rr(M["gt"])
                    st.update(s)
                    return B.back(o)
                put("xycutpp_rr", rr)
                rec["uncertain_nodes"] = st.get("uncertain_nodes", 0)
            # ---------------- all GT boxes incl. headers/footers/page numbers (no labels for XY-Cut++)
            nf = len(p["fboxes"])
            if sum(1 for x in p["fbody"] if x >= 0) == nb and p["have_ro"]:
                deco = set(p["deco"])
                Fb, Fl, Ft = p["fboxes"], p["flabels"], p["ftexts"]

                def sub(fn, keep):
                    O = Orderer([Fb[i] for i in keep], [Fl[i] for i in keep], [Ft[i] for i in keep], W, H, p["rects"])
                    return [keep[i] for i in O.back(fn(O))]
                allk = list(range(nf))
                put("full:xycutpp", lambda: sub(lambda O: O.xy(), allk), full=True)
                put("full:xycutpp+deco", lambda: with_deco(nf, deco, Fb, H, lambda keep: sub(lambda O: O.xy(), keep)), full=True)
                put("full:pdfpig", lambda: sub(lambda O: O.pdfpig(), allk), full=True)
                put("full:pdfpig+deco", lambda: with_deco(nf, deco, Fb, H, lambda keep: sub(lambda O: O.pdfpig(), keep)), full=True)
                if "gt" in M:
                    put("full:romlp", lambda: sub(lambda O: O.romlp(M["gt"]), allk), full=True)
                    put("full:romlp+deco", lambda: with_deco(nf, deco, Fb, H, lambda keep: sub(lambda O: O.romlp(M["gt"]), keep)), full=True)
                    put("full:xycutpp_rr", lambda: sub(lambda O: O.rr(M["gt"])[0], allk), full=True)
                    put("full:xycutpp+deco+rr", lambda: with_deco(nf, deco, Fb, H, lambda keep: sub(lambda O: O.rr(M["gt"])[0], keep)), full=True)
            # ---------------- heron int8 boxes (real pipeline)
            if p["have_h8"] and p["h8"]:
                D = p["h8"]
                regs = [d["box"] for d in D]
                nof = [i for i, d in enumerate(D) if not d["furniture"]]

                def hsub(fn, keep):
                    O = Orderer([regs[i] for i in keep], [D[i]["label"] for i in keep], [""] * len(keep), W, H, p["rects"])
                    return [keep[i] for i in O.back(fn(O))] + [i for i in range(len(D)) if i not in keep]
                allk = list(range(len(D)))
                put("h8:xycutpp", lambda: hsub(lambda O: O.xy(), nof), regs=regs)
                put("h8:xycutpp_all", lambda: hsub(lambda O: O.xy(), allk), regs=regs)
                put("h8:pdfpig", lambda: hsub(lambda O: O.pdfpig(), nof), regs=regs)
                if "h8" in M:
                    put("h8:romlp_all", lambda: hsub(lambda O: O.romlp(M["h8"]), allk), regs=regs)
                    put("h8:romlp", lambda: hsub(lambda O: O.romlp(M["h8"]), nof), regs=regs)
                    put("h8:xycutpp_rr_all", lambda: hsub(lambda O: O.rr(M["h8"])[0], allk), regs=regs)
                    put("h8:xycutpp_rr", lambda: hsub(lambda O: O.rr(M["h8"])[0], nof), regs=regs)
                if "h8_stock" in M:
                    put("h8:romlp_stock", lambda: hsub(lambda O: O.romlp(M["h8_stock"]), nof), regs=regs)
            f.write(json.dumps(rec) + "\n")
    print("eval6", k, n, len(ps))


KEEP_PRED = {"full:xycutpp+deco", "full:xycutpp+deco+rr", "xycutpp", "xycutpp_rr", "romlp", "pdfpig"}

if __name__ == "__main__":
    main()
