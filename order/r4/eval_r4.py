"""Gap 13 evaluation, one shard. Pages, rects and decorations come from order/t6/load6.py (OmniDocBench English
+ DrDocBench dev, rects/decorations of order run RO_RUN, heron int8 boxes of LAYOUT_RUN). Two box sources:
  full  GT boxes incl. headers/footers/page numbers, repeated decorations moved out first (current pick
        full:xycutpp+deco = candidate 0 of config "base")
  h8    heron int8 detections (score >= 0.5), page header/footer labels moved out (current pick h8:xycutpp);
        each detection carries the GT text of the GT regions it covers, joined in top-left order
For every config the record keeps every candidate order's continuity score and its exact / tau / ed, so the
continuity margin is chosen on the dev split afterwards. Inputs are in canonical top-left order (no GT order).
usage: eval_r4.py SHARD NSHARDS OUT.jsonl"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ORDER, "t6"))
sys.path.insert(0, os.path.join(ORDER, "scripts"))
import load6  # noqa: E402
import r4order  # noqa: E402
import ro  # noqa: E402

HARD = {"newspaper", "magazine", "colorful_textbook", "exam_paper"}


def canon(boxes):
    return sorted(range(len(boxes)), key=lambda i: (round(boxes[i][1], 1), round(boxes[i][0], 1), i))


def score_full(order_full, p):
    o = [p["fbody"][i] for i in order_full if p["fbody"][i] >= 0]
    nb = len(p["boxes"])
    if sorted(o) != list(range(nb)):
        return None
    s = ro.score(o, list(range(nb)))
    return [round(s["tau"], 4), round(s["ed"], 4), s["exact"]]


def score_regs(order_regs, p, regs):
    m = load6.match(p["boxes"], regs)
    if len(m) < 2:
        return None
    rank = {k: i for i, k in enumerate(order_regs)}
    gts = sorted(m)
    pred = sorted(gts, key=lambda g: (rank[m[g]], p["boxes"][g][1], p["boxes"][g][0]))
    s = ro.score(pred, gts)
    return [round(s["tau"], 4), round(s["ed"], 4), s["exact"], round(len(m) / len(p["boxes"]), 3)]


def run_mode(boxes, labels, texts, W, H, rects, keep, tail_top, tail_bot, scorer):
    """keep: indices to order; tail_top/tail_bot: indices put before/after (decorations, furniture)."""
    c = canon([boxes[i] for i in keep])
    ks = [keep[k] for k in c]
    B = [boxes[i] for i in ks]
    Lb = [labels[i] for i in ks]
    Tx = [texts[i] or "" for i in ks]
    C = r4order.Cont(Tx)
    out = {}
    for cfg in r4order.CONFIGS:
        t = time.perf_counter()
        cands, trig = r4order.run_config(cfg, B, Lb, Tx, W, H, rects, list(range(len(B))))
        cands = r4order.with_repairs(cands, Lb, Tx, C)
        ms = (time.perf_counter() - t) * 1000
        rows = []
        seen = {}
        for o in cands:
            key = tuple(o)
            if sorted(o) != list(range(len(B))):
                rows.append(None)
                continue
            if key not in seen:
                full = tail_top + [ks[k] for k in o] + tail_bot
                seen[key] = [round(C.score(o, Lb, Tx), 3)] + (scorer(full) or [None])
            rows.append(seen[key])
        out[cfg] = {"c": rows, "trig": trig, "ms": round(ms, 2)}
    return out


def main():
    k, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ps = [p for i, p in enumerate(load6.pages()) if i % n == k]
    with open(out, "w", encoding="utf-8") as f:
        for p in ps:
            W, H = p["W"], p["H"]
            rec = {key: p[key] for key in ("id", "ds", "split", "layout", "source", "bucket", "tags", "doc")}
            rec["hard"] = (p.get("source") in HARD) or ("boxed_or_shaded_region" in p["tags"])
            rec["hard_src"] = p.get("source") in HARD
            rec["n"] = len(p["boxes"])
            nf = len(p["fboxes"])
            if sum(1 for x in p["fbody"] if x >= 0) == rec["n"] and p["have_ro"]:
                deco = set(p["deco"])
                Fb = p["fboxes"]
                keep = [i for i in range(nf) if i not in deco]
                top = sorted([i for i in deco if (Fb[i][1] + Fb[i][3]) / 2 < H / 2], key=lambda i: (Fb[i][1], Fb[i][0]))
                bot = sorted([i for i in deco if i not in top], key=lambda i: (Fb[i][1], Fb[i][0]))
                rec["full"] = run_mode(Fb, p["flabels"], p["ftexts"], W, H, p["rects"], keep, top, bot,
                                       lambda o: score_full(o, p))
            if p["have_h8"] and p["h8"]:
                D = p["h8"]
                regs = [d["box"] for d in D]
                m = load6.match(p["boxes"], regs)
                tx = {}
                for g, r in sorted(m.items(), key=lambda kv: (p["boxes"][kv[0]][1], p["boxes"][kv[0]][0])):
                    tx.setdefault(r, []).append(p["texts"][g] or "")
                texts = [" ".join(tx.get(r, [])) for r in range(len(D))]
                labels = [d["label"] for d in D]
                nof = [i for i, d in enumerate(D) if not d["furniture"]]
                furn = [i for i in range(len(D)) if D[i]["furniture"]]
                rec["h8"] = run_mode(regs, labels, texts, W, H, p["rects"], nof, [], furn,
                                     lambda o: score_regs(o, p, regs))
            f.write(json.dumps(rec) + "\n")
    print("eval_r4", k, n, len(ps))


if __name__ == "__main__":
    main()
